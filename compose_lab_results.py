#!/usr/bin/env python3
"""Build comprehensive results from immutable, separately recorded study archives."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from html import escape
import json
from pathlib import Path

from lab.analysis import aggregate_groups
from lab.storage import atomic_json
from publish_lab_study import (ROOT, _relative_link, display_title, matched_condition_pairs,
                               published_json, read_json, render_dashboard, write_figures)


def episode_key(row):
    return tuple(row[name] for name in ("stage", "recipe", "condition", "thinking", "seed"))


def compose_4b(initial, core):
    """Use new core controls once; retain original core as verification records."""
    initial, core = Path(initial).resolve(), Path(core).resolve()
    older, older_hash = published_json(initial, "results.json")
    newer, newer_hash = published_json(core, "results.json")
    for value in (older, newer):
        if value.get("status") != "complete" or value.get("integrity_warning_count"):
            raise ValueError("Comprehensive primary results require complete, verified source archives")
    if not older["model"].get("fingerprint_sha256") or older["model"].get("fingerprint_sha256") != newer["model"].get("fingerprint_sha256"):
        raise ValueError("4B source archives use different model fingerprints")
    if not older["calibration"].get("vectors_sha256") or older["calibration"].get("vectors_sha256") != newer["calibration"].get("vectors_sha256"):
        raise ValueError("4B source archives use different calibration vectors")
    fresh = [row for row in newer["runs"] if row["stage"] == "core"]
    if len(fresh) != len(newer["runs"]):
        raise ValueError("The replacement core archive must contain only core episodes")
    identities = {episode_key(row): row for row in fresh}
    if len(identities) != len(fresh):
        raise ValueError("Duplicate core condition/seed records")
    verification = []
    for row in older["runs"]:
        if row["stage"] != "core":
            continue
        matched = identities.get(episode_key(row))
        if not matched:
            raise ValueError("A previous core control has no replacement")
        verification.append(dict(original_run=row["run_id"], primary_run=matched["run_id"],
            actions_identical=row["action_sequence_sha256"] == matched["action_sequence_sha256"] and row["actions"] == matched["actions"],
            tokens_identical=row["token_sequence_sha256"] == matched["token_sequence_sha256"] and row["tokens"] == matched["tokens"]))
    noncore = [row for row in older["runs"] if row["stage"] != "core"]
    rows = deepcopy(fresh + noncore)
    sources = {row["run_id"]: (core if row["stage"] == "core" else initial).relative_to(ROOT).as_posix() for row in rows}
    if len(sources) != len(rows):
        raise ValueError("Source archives contain colliding run IDs")
    all_recorded = len(older["runs"]) + len(newer["runs"])
    exact = sum(row["actions_identical"] and row["tokens_identical"] for row in verification)
    composition = dict(kind="two_batch_primary_condition_set", primary_episodes=len(rows),
        total_recorded_episodes=all_recorded, verification_repeats=len(verification), exact_verification_repeats=exact,
        selection_rule="Use all core episodes from the pain-inclusive batch and all noncore episodes from the original batch. Keep original core controls as verification repeats, not extra primary observations.",
        description=f"{len(rows)} primary episodes combine {len(fresh)} core episodes—including active, sham and pain-only arms—with {len(noncore)} other-stage episodes. "
        f"The {all_recorded} total recorded 4B episodes come from two separately frozen batches. {len(verification)} original core controls are retained as verification repeats; "
        f"{exact}/{len(verification)} exactly reproduce the primary controls' full action and token sequences. They are excluded from primary charts and totals. This is a composed analysis, not one {len(rows)}-episode execution.",
        source_archives=[dict(path=initial.relative_to(ROOT).as_posix(), results_sha256=older_hash,
                             protocol_sha256=older["protocol_sha256"], primary_episodes=len(noncore)),
                         dict(path=core.relative_to(ROOT).as_posix(), results_sha256=newer_hash,
                             protocol_sha256=newer["protocol_sha256"], primary_episodes=len(fresh))], verification=verification)
    result = deepcopy(newer)
    result.update(title="Opium Bench comprehensive results", publication_title="Qwen3-4B · comprehensive results",
                  planned_episodes=len(rows), recorded_episodes=len(rows), runs=rows, groups=aggregate_groups(rows),
                  condition_pairs=matched_condition_pairs(rows), composition=composition, run_sources=sources,
                  calibration_source=core.relative_to(ROOT).as_posix()+"/calibration",
                  status_counts=dict(Counter(row["status"] for row in rows)))
    for key in ("protocol_sha256", "receipt_status", "receipt_declared_episodes", "started_at", "finished_at", "comparison", "comparison_note", "historical_observation"):
        result.pop(key, None)
    result["totals"] = {field: sum(row[field] for row in rows) for field in older["totals"]}
    return result


def render_comprehensive_dashboard(result, output, dashboard, notes=None, replication=None):
    html = render_dashboard(result, output, dashboard, reasoning_notes=notes)
    archives = " · ".join(f"<a href='{escape(_relative_link(ROOT / item['path'], dashboard.parent))}/protocol.json'>Source batch {i+1} protocol</a>"
                           for i, item in enumerate(result["composition"]["source_archives"]))
    html = html.replace("Selection rule, verification repeats and source checksums</a></p>",
                        "Selection rule, verification repeats and source checksums</a></p><p>"+archives+"</p>", 1)
    if replication:
        other, path = replication
        other_html = render_dashboard(other, path, dashboard)
        # Separate model sections share the page, preserving their own denominators.
        body = other_html.split("<main>", 1)[1].rsplit("</main>", 1)[0]
        import re
        body = re.sub(r'\bid=([A-Za-z0-9_-]+)', r'id=model27-\1', body)
        body = re.sub(r'href=#([A-Za-z0-9_-]+)', r'href=#model27-\1', body)
        label = f"<section id=model27><h2>{escape(display_title(other))}</h2><p>Separate model configuration; its {other['recorded_episodes']} episodes are not pooled with the 4B results.</p></section>"
        html = html.replace("</main></html>", label+body+"</main></html>")
        html = html.replace("<nav>", "<nav><a href=#findings>Qwen3-4B</a><a href=#model27>Qwen3.8-27B</a>", 1)
    return html


def build(initial, core, output, dashboard, notes=None, replication=None, skip_figures=False):
    initial, core, output, dashboard = [Path(path).resolve() for path in (initial, core, output, dashboard)]
    replication = Path(replication).resolve() if replication else None
    protected = [initial, core, *([replication] if replication else [])]
    if any(output.is_relative_to(source) or dashboard.is_relative_to(source) for source in protected):
        raise ValueError("Comprehensive output must be outside immutable source archives")
    other = (published_json(replication, "results.json")[0], replication) if replication else None
    if other and (other[0].get("status") != "complete" or other[0].get("integrity_warning_count")):
        raise ValueError("Comprehensive model comparison requires a complete, verified replication archive")
    result = compose_4b(initial, core)
    output.mkdir(parents=True, exist_ok=True)
    if skip_figures:
        result["figures"], result["figure_note"] = [], "Figures skipped; all measurements remain in the tables."
    else:
        result["figures"], result["figure_note"] = write_figures(result, output/"figures")
    atomic_json(output/"composition.json", result["composition"])
    atomic_json(output/"results.json", result)
    html = render_comprehensive_dashboard(result, output, dashboard, notes, other)
    dashboard.parent.mkdir(parents=True, exist_ok=True)
    dashboard.write_text(html, encoding="utf-8")
    atomic_json(output/"checksums.json", {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "checksums.json"})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial",type=Path,default=ROOT/"studies/initial")
    parser.add_argument("--core",type=Path,default=ROOT/"studies/core-pain-4b")
    parser.add_argument("--output",type=Path,default=ROOT/"studies/comprehensive-4b")
    parser.add_argument("--dashboard",type=Path,default=ROOT/"docs/results.html")
    parser.add_argument("--reasoning-notes",type=Path,default=ROOT/"docs/comprehensive-thinking-notes.json")
    parser.add_argument("--replication",type=Path,help="Published 27B archive to include as a second model section")
    parser.add_argument("--skip-figures",action="store_true")
    args=parser.parse_args()
    result=build(args.initial,args.core,args.output,args.dashboard,read_json(args.reasoning_notes),args.replication,args.skip_figures)
    print(f"Comprehensive results: {result['recorded_episodes']} primary episodes; {result['composition']['total_recorded_episodes']} recorded; {result['composition']['verification_repeats']} verification repeats. {args.dashboard}")


if __name__ == "__main__":
    main()
