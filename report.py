#!/usr/bin/env python3
"""Report a local activation-steering pilot: python report.py RUN_DIR.

The paired bootstrap resamples the observed authored cases. Its interval is
descriptive sensitivity to this case mix, not a population-generalization claim.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261001
METRIC_ALIASES = {
    "repetition_3gram": ("repetition_3gram", "repetition", "repetition_rate", "rep3"),
    "distinct_word_ratio": ("distinct_word_ratio", "distinct_ratio", "distinct", "distinct_tokens"),
    "positive_hits": ("positive_hits", "positive_lexical_hits", "positive_word_hits", "pos_hits", "positive"),
    "negative_hits": ("negative_hits", "negative_lexical_hits", "negative_word_hits", "neg_hits", "negative"),
}
LIMITATION = (
    "This pilot manipulates one learned residual-stream direction at one layer. "
    "It does not identify pain neurons or establish felt pain, pleasure, or relief. "
    "Later layers can reconstruct information removed at the intervention layer. "
    "Word hits are style proxies; this small locally authored convenience suite "
    "is not an established benchmark and supports no general quality conclusion."
)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _load(run_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    raw_conditions = manifest.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("manifest.conditions must be a nonempty list")
    conditions = [dict(c) if isinstance(c, dict) else {"name": c} for c in raw_conditions]
    names = [c.get("name") for c in conditions]
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError("Condition names must be unique nonempty strings")
    rows, seen = [], set()
    for line_no, line in enumerate((run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid generations.jsonl line {line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Row {line_no} must be an object")
        condition, panel, case_id = row.get("condition"), row.get("panel"), row.get("case_id")
        if condition not in names:
            raise ValueError(f"Row {line_no}: unknown condition {condition!r}")
        if panel not in {"quality", "valence", "nll"}:
            raise ValueError(f"Row {line_no}: unknown panel {panel!r}")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Row {line_no}: case_id must be a nonempty string")
        key = condition, panel, case_id
        if key in seen:
            raise ValueError(f"Duplicate condition/panel/case row: {key}")
        seen.add(key)
        if panel == "quality" and not isinstance(row.get("correct"), bool):
            raise ValueError(f"Row {line_no}: quality.correct must be boolean")
        if panel == "nll":
            count = row.get("n_tokens")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"Row {line_no}: nll.n_tokens must be a positive integer")
            total = _finite_number(row.get("nll_total"), f"Row {line_no}: nll_total")
            if total < 0:
                raise ValueError(f"Row {line_no}: nll_total must be nonnegative")
        row.setdefault("category", "uncategorized")
        if not isinstance(row["category"], str):
            raise ValueError(f"Row {line_no}: category must be a string")
        rows.append(row)
    return manifest, conditions, rows


def _style(rows: list[dict]) -> dict:
    result: dict[str, Any] = {"n": len(rows), "metrics": {}}
    for canonical, alternatives in METRIC_ALIASES.items():
        values = []
        for row in rows:
            metrics = row.get("metrics") or {}
            if not isinstance(metrics, dict):
                raise ValueError(f"metrics must be an object for {row['case_id']}")
            value = next((metrics[k] for k in alternatives if k in metrics), None)
            if value is not None:
                values.append(_finite_number(value, f"{row['case_id']}.{canonical}"))
        result["metrics"][canonical] = {"mean": mean(values) if values else None, "n": len(values)}
    token_counts = [_finite_number(r["new_tokens"], "new_tokens") for r in rows if r.get("new_tokens") is not None]
    finishes = [r["finish"] for r in rows if r.get("finish") in {"eos", "length"}]
    result.update(
        mean_new_tokens=mean(token_counts) if token_counts else None,
        length_limited=sum(f == "length" for f in finishes),
        finish_observed=len(finishes),
        truncation_rate=sum(f == "length" for f in finishes) / len(finishes) if finishes else None,
    )
    return result


def _nll(rows: list[dict]) -> dict:
    count = sum(r["n_tokens"] for r in rows)
    total = math.fsum(float(r["nll_total"]) for r in rows)
    average = total / count if count else None
    overflow = average is not None and average > 709
    return {
        "n_cases": len(rows), "n_tokens": count, "nll_total": total,
        "mean_nll": average,
        "perplexity": math.exp(average) if average is not None and not overflow else None,
        "perplexity_overflow": overflow,
    }


def _paired(condition: list[dict], baseline: list[dict], name: str, baseline_name: str) -> dict:
    import numpy as np

    current = {r["case_id"]: r for r in condition}
    reference = {r["case_id"]: r for r in baseline}
    ids = sorted(current.keys() & reference.keys())
    gains = [k for k in ids if current[k]["correct"] and not reference[k]["correct"]]
    losses = [k for k in ids if reference[k]["correct"] and not current[k]["correct"]]
    differences = np.array([int(current[k]["correct"]) - int(reference[k]["correct"]) for k in ids], dtype=float)
    interval, delta = None, None
    if ids:
        delta = float(differences.mean())
        # Each independent draw samples the paired case IDs, with replacement.
        # A stable per-condition seed prevents ordering from changing intervals.
        salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
        rng = np.random.default_rng(BOOTSTRAP_SEED + salt)
        draws = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
        for start in range(0, BOOTSTRAP_SAMPLES, 1000):
            size = min(1000, BOOTSTRAP_SAMPLES - start)
            indices = rng.integers(0, len(ids), size=(size, len(ids)))
            draws[start:start + size] = differences[indices].mean(axis=1)
        interval = [float(v) for v in np.quantile(draws, [0.025, 0.975])]
    return {
        "baseline": baseline_name, "n_paired": len(ids), "accuracy_difference": delta,
        "descriptive_paired_bootstrap_95_interval": interval,
        "gained_case_ids": gains, "lost_case_ids": losses,
        "unpaired_condition_case_ids": sorted(current.keys() - reference.keys()),
        "unpaired_baseline_case_ids": sorted(reference.keys() - current.keys()),
    }


def summarize(manifest: dict, conditions: list[dict], rows: list[dict]) -> dict:
    names = [c["name"] for c in conditions]
    baseline = manifest.get("baseline_condition", "baseline")
    if baseline not in names:
        baseline = "baseline_none" if "baseline_none" in names else None
    if baseline is None:
        raise ValueError("Cannot identify baseline; set manifest.baseline_condition")
    grouped: dict[str, dict[str, list[dict]]] = {name: defaultdict(list) for name in names}
    panel_ids: dict[str, set[str]] = defaultdict(set)
    definitions: dict[tuple[str, str], tuple[str, Any]] = {}
    for row in rows:
        grouped[row["condition"]][row["panel"]].append(row)
        panel_ids[row["panel"]].add(row["case_id"])
        # A repeated ID has to represent the same case under every condition.
        key = row["panel"], row["case_id"]
        definition = row["category"], row.get("prompt")
        if key in definitions and definitions[key] != definition:
            raise ValueError(f"Case {key} changes category or prompt between conditions")
        definitions[key] = definition
    missing = {}
    for name in names:
        gaps = {panel: sorted(ids - {r["case_id"] for r in grouped[name][panel]}) for panel, ids in panel_ids.items()}
        gaps = {panel: ids for panel, ids in gaps.items() if ids}
        if gaps:
            missing[name] = gaps
    status = str(manifest.get("status", "unknown"))
    declared_complete = status.lower() in {"complete", "completed", "success", "succeeded"}
    expected_counts = {panel: manifest.get(panel + "_cases") for panel in ("quality", "valence", "nll")}
    count_gaps = {}
    for name in names:
        gaps = {}
        for panel, expected in expected_counts.items():
            if expected is None:
                continue
            if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
                raise ValueError(f"manifest.{panel}_cases must be a nonnegative integer")
            observed = len(grouped[name][panel])
            if observed != expected:
                gaps[panel] = {"expected": expected, "observed": observed}
        if gaps:
            count_gaps[name] = gaps
    if declared_complete and missing:
        raise ValueError("Manifest declares a complete run, but condition/case rows are missing: " + json.dumps(missing))
    if declared_complete and count_gaps:
        raise ValueError("Manifest declares a complete run, but panel counts differ: " + json.dumps(count_gaps))
    if declared_complete and not rows:
        raise ValueError("Manifest declares a complete run, but generations.jsonl has no rows")
    outputs = []
    for spec in conditions:
        name = spec["name"]
        quality = grouped[name]["quality"]
        by_category = defaultdict(list)
        for row in quality:
            by_category[row["category"]].append(row)
        categories = {category: {"n": len(items), "correct": sum(r["correct"] for r in items), "accuracy": mean(r["correct"] for r in items)} for category, items in sorted(by_category.items())}
        nll_categories = defaultdict(list)
        for row in grouped[name]["nll"]:
            nll_categories[row["category"]].append(row)
        outputs.append({
            "name": name, "parameters": spec,
            "quality": {
                "n": len(quality), "correct": sum(r["correct"] for r in quality),
                "accuracy": mean(r["correct"] for r in quality) if quality else None,
                "categories": categories, "generation": _style(quality),
                "vs_baseline": _paired(quality, grouped[baseline]["quality"], name, baseline),
            },
            "valence": _style(grouped[name]["valence"]),
            "nll": {"overall": _nll(grouped[name]["nll"]), "categories": {k: _nll(v) for k, v in sorted(nll_categories.items())}},
        })
    return {
        "schema_version": 1, "manifest": manifest, "baseline_condition": baseline,
        "run_status": status,
        "report_status": "complete" if declared_complete else "partial_or_unverified",
        "rows_read": len(rows), "observed_case_counts": {k: len(v) for k, v in sorted(panel_ids.items())},
        "missing_rows": missing,
        "expected_case_counts": expected_counts, "case_count_mismatches": count_gaps,
        "coverage_note": "Missing rows are checked against observed case IDs and manifest case counts when present. The identity of cases absent from every condition cannot be inferred from these files.",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "unit": "paired quality case ID", "interpretation": "Descriptive resampling sensitivity for this authored case mix, not a confidence claim about a broader task population."},
        "limitations": LIMITATION,
        "conditions": outputs,
    }


def _neutral_nll(condition: dict) -> dict | None:
    categories = condition["nll"]["categories"]
    return categories.get("neutral") or categories.get("neutral_text")


def _load_content_audit(run_dir: Path, rows: list[dict]) -> dict | None:
    """Validate an optional secondary audit against the unchanged raw records."""
    path = run_dir / "content_audit.json"
    if not path.exists():
        return None
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or audit.get("audit_type") != "posthoc_unblinded_model_assisted_exploratory":
        raise ValueError("content_audit.json must identify a complete posthoc unblinded exploratory audit")
    scope = audit.get("scope", [])
    if not isinstance(scope, list) or not scope or len(scope) != len(set(scope)):
        raise ValueError("Content audit scope must contain unique condition names")
    raw_rows = [json.loads(line) for line in (run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    canonical = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for r in raw_rows if r["panel"] == "quality" and r["condition"] in scope) + "\n"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != audit.get("source", {}).get("audited_quality_rows_sha256"):
        raise ValueError("Content audit source hash does not match the current quality records")
    if set(audit.get("conditions", {})) != set(scope):
        raise ValueError("Content audit scope and conditions disagree")
    bucket_names = {"strict_correct", "format_only", "unfinished", "substantive_wrong"}
    for name in scope:
        item = audit["conditions"][name]
        originals = {r["case_id"]: r for r in rows if r["condition"] == name and r["panel"] == "quality"}
        cases = item.get("cases", [])
        if not originals or len(cases) != len(originals) or len({c["case_id"] for c in cases}) != len(cases):
            raise ValueError(f"Content audit case coverage mismatch: {name}")
        counts = {bucket: 0 for bucket in bucket_names}
        for case in cases:
            original = originals.get(case["case_id"])
            if original is None:
                raise ValueError(f"Unknown audited case: {name}/{case['case_id']}")
            for field in ("category", "prompt", "expected", "text", "finish"):
                if case.get(field) != original.get(field):
                    raise ValueError(f"Content audit changes raw {field}: {name}/{case['case_id']}")
            bucket = case.get("bucket")
            if bucket not in bucket_names or case.get("strict_correct") is not original["correct"]:
                raise ValueError(f"Invalid audit bucket or changed primary score: {name}/{case['case_id']}")
            if (bucket == "strict_correct") != original["correct"]:
                raise ValueError(f"Strict-correct audit bucket disagrees with original score: {name}/{case['case_id']}")
            if case.get("completed_content_correct") is not (bucket in {"strict_correct", "format_only"}):
                raise ValueError(f"Content audit completion label disagrees with bucket: {name}/{case['case_id']}")
            counts[bucket] += 1
        if counts != item.get("counts") or item.get("n") != len(cases):
            raise ValueError(f"Content audit aggregate counts disagree: {name}")
        if item.get("completed_content_correct") != counts["strict_correct"] + counts["format_only"]:
            raise ValueError(f"Content audit completion total disagrees: {name}")
    return audit


def make_chart(summary: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = summary["conditions"]
    labels = [c["name"].replace("_", " ") for c in conditions]
    colors = []
    for c in conditions:
        params = c["parameters"]
        if c["name"] == summary["baseline_condition"]:
            colors.append("#536576")
        elif "random" in str(params.get("direction", "")) or "random" in c["name"]:
            colors.append("#a5acb7")
        elif params.get("joy_dose", 0):
            colors.append("#147d76")
        else:
            colors.append("#7954ad")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, max(5.5, 0.37 * len(conditions) + 2.2)), sharey=True)
    fig.patch.set_facecolor("#f7f9fc")
    ys = np.arange(len(conditions))
    for ax in axes:
        ax.set_facecolor("#f7f9fc")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#c4cdd8")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#dce2e9", linewidth=0.6, zorder=0)
    accuracy = [c["quality"]["accuracy"] for c in conditions]
    axes[0].barh(ys, [100 * v if v is not None else 0 for v in accuracy], color=colors, height=.66, zorder=3)
    axes[0].set(yticks=ys, yticklabels=labels, xlim=(0, 113), xlabel="Content + requested format correct (%)", title="Strict task score · higher is better")
    axes[0].set_xticks([0, 25, 50, 75, 100])
    for i, c in enumerate(conditions):
        value = c["quality"]["accuracy"]
        axes[0].text((100 * value + 1.5) if value is not None else 1.5, i, f"{c['quality']['correct']}/{c['quality']['n']}" if value is not None else "no data", va="center", fontsize=9)
    ppls = [(_neutral_nll(c) or {}).get("perplexity") for c in conditions]
    observed = [v for v in ppls if v is not None]
    log_scale = bool(observed and max(observed) / min(observed) > 5)
    if log_scale:
        axes[1].set_xscale("log")
        # Starting bars at one avoids a zero baseline on a logarithmic axis.
        axes[1].barh(ys, [v - 1 if v is not None else 0 for v in ppls], left=1, color=colors, height=.66, zorder=3)
    else:
        axes[1].barh(ys, [v if v is not None else 0 for v in ppls], color=colors, height=.66, zorder=3)
    axes[1].set(xlabel="Token-weighted perplexity" + (" (log scale)" if log_scale else ""), title="Held-out neutral text · lower is better")
    if observed:
        axes[1].set_xlim((max(.95, min(observed) * .75) if log_scale else 0), max(observed) * (1.7 if log_scale else 1.24))
    for i, value in enumerate(ppls):
        axes[1].text(value * 1.04 if value is not None else (1 if log_scale else .01), i, f"{value:.2f}" if value is not None else "no data", va="center", fontsize=9)
    axes[0].invert_yaxis()
    manifest = summary["manifest"]
    status = "COMPLETE PILOT" if summary["report_status"] == "complete" else "PARTIAL / UNVERIFIED RUN"
    fig.suptitle(f"Local direction suppression & joy steering — {status}", x=.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(.02, .925, f"{manifest.get('model', 'Model unspecified')}  ·  block {manifest.get('layer', '?')}  ·  scope: {manifest.get('token_scope', 'unspecified')}", fontsize=10, color="#4d5b6c")
    fig.text(.02, .025, "Small authored pilot; no general quality or felt-state conclusion. Bars summarize observed cases only.", fontsize=9, color="#4d5b6c")
    fig.tight_layout(rect=(0, .06, 1, .9), w_pad=3)
    fig.savefig(destination, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: float | None, places: int = 3) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _percent(value: float | None, signed: bool = False) -> str:
    return "—" if value is None else format(value * 100, "+.1f" if signed else ".1f") + "%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    # Cells are pre-escaped or trusted markup assembled locally below.
    return '<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{_escape(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def write_html(summary: dict, rows: list[dict], chart_path: Path, destination: Path) -> None:
    manifest = summary["manifest"]
    complete = summary["report_status"] == "complete"
    status = "Complete pilot" if complete else "Partial / unverified run"
    png = base64.b64encode(chart_path.read_bytes()).decode("ascii")
    quality_rows, category_rows, style_rows, nll_rows = [], [], [], []
    changes = []
    for condition in summary["conditions"]:
        name = _escape(condition["name"])
        q, v = condition["quality"], condition["valence"]
        paired = q["vs_baseline"]
        ci = paired["descriptive_paired_bootstrap_95_interval"]
        interval = "—" if ci is None else f"[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}] pp"
        quality_rows.append([name, f"{q['correct']}/{q['n']}", _percent(q["accuracy"]), _percent(paired["accuracy_difference"], True).replace("%", " pp"), interval, str(paired["n_paired"]), _percent(q["generation"]["truncation_rate"])])
        for category, data in q["categories"].items():
            category_rows.append([name, _escape(category), f"{data['correct']}/{data['n']}", _percent(data["accuracy"])])
        metrics = v["metrics"]
        style_rows.append([name, str(v["n"]), _number(metrics["positive_hits"]["mean"]), _number(metrics["negative_hits"]["mean"]), _number(metrics["repetition_3gram"]["mean"]), _number(metrics["distinct_word_ratio"]["mean"]), _percent(v["truncation_rate"])])
        for category, data in condition["nll"]["categories"].items():
            nll_rows.append([name, _escape(category), str(data["n_cases"]), str(data["n_tokens"]), _number(data["mean_nll"]), "overflow" if data["perplexity_overflow"] else _number(data["perplexity"])])
        if condition["name"] != summary["baseline_condition"]:
            gains = ", ".join(paired["gained_case_ids"]) or "none"
            losses = ", ".join(paired["lost_case_ids"]) or "none"
            changes.append(f'<details><summary>{name}: {len(paired["gained_case_ids"])} gained · {len(paired["lost_case_ids"])} lost</summary><p><strong>Gained</strong> {_escape(gains)}</p><p><strong>Lost</strong> {_escape(losses)}</p></details>')
    transcript_groups = defaultdict(list)
    for row in rows:
        transcript_groups[row["condition"]].append(row)
    transcripts = []
    for condition in summary["conditions"]:
        name = condition["name"]
        entries = []
        for row in transcript_groups[name]:
            tag = "correct" if row.get("correct") is True else "incorrect" if row.get("correct") is False else row["panel"]
            # Preserve the complete logged response without interpreting markup.
            metadata = {k: v for k, v in row.items() if k not in {"prompt", "text"}}
            entries.append(f'<details class="case"><summary>{_escape(row["case_id"])} · {_escape(row["category"])} · {_escape(tag)}</summary><h4>Prompt / scored text</h4><pre>{_escape(row.get("prompt", ""))}</pre><h4>Complete response</h4><pre>{_escape(row.get("text", ""))}</pre><h4>Recorded metrics</h4><pre>{_escape(json.dumps(metadata, indent=2, ensure_ascii=False))}</pre></details>')
        transcripts.append(f'<details><summary>{_escape(name)} · {len(entries)} records</summary>{"".join(entries)}</details>')
    missing_html = ""
    if summary["missing_rows"] or summary["case_count_mismatches"]:
        coverage = {"missing_observed_case_ids": summary["missing_rows"], "panel_count_mismatches": summary["case_count_mismatches"]}
        missing_html = '<details open><summary>Missing case rows / count mismatches</summary><pre>' + _escape(json.dumps(coverage, indent=2)) + "</pre></details>"
    audit_html = ""
    audit = summary.get("content_audit")
    if audit:
        audit_rows, audit_details = [], []
        for name in audit["scope"]:
            data = audit["conditions"][name]
            counts = data["counts"]
            audit_rows.append([_escape(name), str(data["n"]), str(counts["strict_correct"]), str(counts["format_only"]), str(counts["unfinished"]), str(counts["substantive_wrong"]), str(data["completed_content_correct"])])
            explanations = [f'<details class="case"><summary>{_escape(c["case_id"])} · {_escape(c["bucket"])}</summary><p>{_escape(c["evidence"])}</p><h4>Prompt</h4><pre>{_escape(c["prompt"])}</pre><h4>Expected answer</h4><pre>{_escape(json.dumps(c["expected"], ensure_ascii=False))}</pre><h4>Complete response</h4><pre>{_escape(c["text"])}</pre></details>' for c in data["cases"]]
            audit_details.append(f'<details><summary>{_escape(name)} · case-level audit reasons</summary>{"".join(explanations)}</details>')
        audit_table = _table(["Condition", "All cases", "Strict correct", "Format only", "Unfinished", "Substantive wrong", "Completed content correct"], audit_rows)
        limits = "".join(f"<li>{_escape(s)}</li>" for s in audit["limitations"])
        buckets = "".join(f"<p><strong>{_escape(k)}</strong>: {_escape(v)}</p>" for k, v in audit["bucket_definitions"].items())
        audit_html = f'<section><h2>Exploratory content audit · post hoc and unblinded</h2><p><strong>This secondary model-assisted review was defined after inspecting results. It is not independent, blind, preregistered, or a replacement for the primary strict score.</strong></p><p class="note">{_escape(audit["methodology"])}</p>{audit_table}<p class="note">Completed content correct = strict correct + format only; the denominator remains all cases. No significance or population generalization claim is made for these judgments.</p><details><summary>Audit definitions and limitations</summary>{buckets}<ul>{limits}</ul></details>{"".join(audit_details)}<p class="small note">Validated against the canonical hash of audited raw quality records: <code>{_escape(audit["source"]["audited_quality_rows_sha256"])}</code>. <a href="content_audit.json">Full audit JSON</a>.</p></section>'
    body = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI opium · local activation pilot</title>
<style>
:root{{color-scheme:light;--ink:#1c2c3d;--muted:#586a7d;--line:#dce3eb;--accent:#147d76}}
*{{box-sizing:border-box}} body{{margin:0;background:#f5f7fa;color:var(--ink);font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1320px;margin:auto;padding:40px 30px 70px}} header{{border-top:5px solid var(--accent);padding:24px 0 18px}} h1{{font-size:36px;line-height:1.2;letter-spacing:-1px;margin:8px 0 12px}} h2{{font-size:22px;margin:0 0 10px}} h4{{margin:12px 0 5px}}
.eyebrow{{letter-spacing:.12em;text-transform:uppercase;font-size:12px;font-weight:700;color:var(--accent)}} .meta,.note{{color:var(--muted)}} .pill{{display:inline-block;border-radius:16px;background:{'#dcefe8' if complete else '#fff0c2'};padding:3px 12px;font-size:12px;font-weight:700}}
section,.callout{{background:white;border:1px solid var(--line);border-radius:10px;padding:24px;margin:20px 0}} .callout{{border-left:4px solid {'#147d76' if complete else '#bd8616'}}} .callout p{{margin:7px 0}} img{{width:100%;height:auto;display:block}} .table-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th{{text-align:left;background:#f1f5f8;color:#42566a;font-weight:650}} td,th{{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}} td:not(:first-child),th:not(:first-child){{font-variant-numeric:tabular-nums}} tr:last-child td{{border-bottom:0}}
details{{border:1px solid var(--line);border-radius:7px;padding:11px 14px;margin:9px 0}} summary{{cursor:pointer;font-weight:600}} .case{{margin-left:12px;background:#fbfcfe}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f8;padding:12px;border-radius:5px;font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}} code{{font-family:ui-monospace,Consolas,monospace}} .small{{font-size:12px}} a{{color:var(--accent)}}
@media(max-width:700px){{main{{padding:20px 12px}} section,.callout{{padding:15px}} h1{{font-size:29px}}}}
</style><main>
<header><div class="eyebrow">Activation intervention · authored pilot</div><h1>Suppression, joy steering & task quality</h1><span class="pill">{status}</span><p class="meta">{_escape(manifest.get('model', 'Unspecified model'))} · block {_escape(manifest.get('layer', '?'))} · {_escape(manifest.get('token_scope', 'unspecified token scope'))}<br>Revision: <code>{_escape(manifest.get('revision') or 'not recorded')}</code></p></header>
<div class="callout"><p><strong>Interpretation boundary.</strong> {_escape(LIMITATION)}</p><p>Manifest status: <strong>{_escape(summary['run_status'])}</strong>. Observed rows: {summary['rows_read']}. {'All observed condition/case combinations are present.' if complete else 'This report is provisional; missing or unfinished conditions must not be treated as completed comparisons.'}</p></div>
{missing_html}
<section><h2>Observed results</h2><p class="note">Quality and neutral-text likelihood are separate outcomes. Perplexity uses total negative log likelihood divided by total scored tokens.</p><img alt="Authored-case quality and neutral-text perplexity by intervention condition" src="data:image/png;base64,{png}"></section>
<section><h2>Task quality · primary strict score</h2><p><strong>Primary accuracy requires both correct content and the requested output format; format-only failures count as incorrect.</strong> Length limits can also affect this score.</p><p class="note">Changes are paired against {_escape(summary['baseline_condition'])}, on matching case IDs. Intervals use 10,000 independently resampled paired case sets. They describe sensitivity to the authored case mix and do not justify population generalization. Dose comparisons are exploratory.</p>{_table(['Condition','Correct / observed','Accuracy','Change vs baseline','Descriptive 95% interval','Paired cases','Length-limited'], quality_rows)}<p class="small note">Length-limited responses reached the generation cap; incorrect scores may reflect truncation. Missing outcomes are never converted into successes or failures.</p></section>
{audit_html}
<section><h2>Scores by task category</h2>{_table(['Condition','Category','Correct / observed','Accuracy'], category_rows)}</section>
<section><h2>Cases gained and lost</h2><p class="note">A gain is an incorrect baseline response that became correct. A loss is a correct baseline response that became incorrect.</p>{''.join(changes)}</section>
<section><h2>Valence language and repetition</h2><p class="note">Means on the separate free-response panel. Positive and negative word hits are descriptive lexical proxies, not measurements of experience. Missing metrics appear as a dash.</p>{_table(['Condition','Responses','Positive hits','Negative hits','Repeated 3-grams','Distinct word ratio','Length-limited'], style_rows)}</section>
<section><h2>Held-out text likelihood</h2><p class="note">Neutral and pain-related text are reported separately. Lower perplexity means better prediction of these particular texts; an affect-related change can alter pain-text likelihood without improving general capability.</p>{_table(['Condition','Text category','Cases','Scored tokens','Mean NLL (nats)','Perplexity'], nll_rows)}</section>
<section><h2>Complete transcripts</h2><p class="note">Every logged response is preserved below. Expand a condition and a case to inspect its prompt, response and metrics.</p>{''.join(transcripts)}</section>
<section><h2>Provenance and coverage</h2><p class="note">{_escape(summary['coverage_note'])}</p><p class="small note">Current report source SHA256: <code>{_escape(summary.get('report_source_sha256', 'not recorded'))}</code>. The original run source snapshot and its manifest hash remain unchanged; this renderer may include later reporting clarifications.</p><details><summary>Manifest and intervention parameters</summary><pre>{_escape(json.dumps(manifest, indent=2, ensure_ascii=False))}</pre></details><p class="small note">Report artifacts: <a href="summary.json">summary.json</a> · <a href="quality.png">quality.png</a> · <a href="generations.jsonl">raw records</a>. This HTML embeds the figure and loads no external resources.</p></section>
</main></html>'''
    destination.write_text(body, encoding="utf-8")


def generate_report(run_dir: str | Path) -> dict:
    """Write summary.json, report.html and quality.png; return the summary."""
    run_dir = Path(run_dir)
    manifest, conditions, rows = _load(run_dir)
    summary = summarize(manifest, conditions, rows)
    summary["report_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    audit = _load_content_audit(run_dir, rows)
    if audit is not None:
        summary["content_audit"] = audit
        summary["content_audit_sha256"] = hashlib.sha256((run_dir / "content_audit.json").read_bytes()).hexdigest()
    chart = run_dir / "quality.png"
    make_chart(summary, chart)
    write_html(summary, rows, chart, run_dir / "report.html")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def build_report(run_dir: str | Path) -> dict:
    """Runner-facing entry point; identical to generate_report."""
    return generate_report(run_dir)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    result = generate_report(args.run_dir)
    print(json.dumps({"status": result["report_status"], "rows": result["rows_read"], "report": str((args.run_dir / 'report.html').resolve())}))
    return result


if __name__ == "__main__":
    main()
