#!/usr/bin/env python3
"""Publish auditable study evidence and a static results dashboard, without ML dependencies."""
import argparse
import gzip
import hashlib
from html import escape
from itertools import combinations
import json
from pathlib import Path
import re
import shutil
import os
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from lab.analysis import analyze_run, study_results
from lab.reports import render_report
from lab.storage import atomic_json

ROOT = Path(__file__).resolve().parent
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
TERMINAL = {"complete", "failed", "stopped", "cancelled"}


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def safe_path(base, identifier):
    if not isinstance(identifier, str) or not SAFE_ID.fullmatch(identifier):
        raise ValueError(f"Invalid evidence identifier: {identifier!r}")
    result = (Path(base)/identifier).resolve()
    if result.parent != Path(base).resolve():
        raise ValueError("Evidence path escapes its directory")
    return result


def read_events(path, allow_partial=False):
    path = Path(path)
    filename = path/"events.jsonl"
    if not filename.exists() and (path/"events.jsonl.gz").exists():
        filename = path/"events.jsonl.gz"
    if not filename.exists():
        return [], b"", ["Raw events file is missing"]
    raw = gzip.decompress(filename.read_bytes()) if filename.suffix == ".gz" else filename.read_bytes()
    lines = raw.splitlines(keepends=True)
    events, notes = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("event is not an object")
            events.append(event)
        except (ValueError, UnicodeDecodeError):
            if allow_partial and index == len(lines)-1 and not line.endswith(b"\n"):
                notes.append("Incomplete trailing event excluded from analysis; original bytes preserved")
            else:
                raise ValueError(f"Malformed event at {filename}:{index+1}")
    return events, raw, notes


def deterministic_gzip(path, raw):
    """mtime=0 and no filename make equal evidence compress identically."""
    with Path(path).open("wb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as zipped:
        zipped.write(raw)


def _number(value, digits=2):
    if value is None:
        return "—"
    return f"{value:,.{digits}f}" if isinstance(value, float) else f"{value:,}"


def _percent(value):
    return "—" if value is None else f"{100*value:.1f}%"


def _range(value, percent=False):
    if not value:
        return "—"
    formatter = _percent if percent else _number
    if value["min"] == value["max"]:
        return formatter(value["min"])
    return f"{formatter(value['min'])}–{formatter(value['max'])}"


def _relative_link(path, directory):
    return Path(os.path.relpath(path, directory)).as_posix()


def default_dashboard(output):
    """Keep the reference results page while giving later studies their own page."""
    name = Path(output).name
    if not SAFE_ID.fullmatch(name):
        raise ValueError("Study output directory must have a portable identifier")
    return ROOT / "docs" / ("results.html" if name == "initial" else f"results-{name}.html")


def display_title(results):
    title = results.get("publication_title") or results.get("title") or "Opium Bench study"
    if title == "Opium Den Lab initial descriptive pilot":
        return "Opium Bench · initial descriptive pilot"
    return title


COMPARISON_NOTE = ("Across model studies, checkpoint, quantization/runtime, native tool grammar or reasoning-template settings, "
                   "and independently calibrated intervention directions can change together. They do not isolate model size. "
                   "Matched active/sham comparisons within each study remain the primary comparisons.")


def matched_condition_pairs(records):
    """Compare complete core sequences; preserve missing arms and zero exposure."""
    grouped = {}
    conditions = sorted({row["condition"] for row in records if row["stage"] == "core"})
    for row in records:
        if row["stage"] != "core":
            continue
        key = row["recipe"], row["thinking"], row["seed"]
        arms = grouped.setdefault(key, {})
        if row["condition"] in arms:
            raise ValueError(f"Duplicate core arm for {key}")
        arms[row["condition"]] = row
    result = []
    for key, arms in sorted(grouped.items()):
        for left_name, right_name in combinations(conditions, 2):
            left, right = arms.get(left_name), arms.get(right_name)
            complete = bool(left and right and left["status"] == right["status"] == "complete")
            row = dict(recipe=key[0], thinking=key[1], seed=key[2], left_condition=left_name,
                       right_condition=right_name, pair_complete=complete,
                       left_run=left["run_id"] if left else None, right_run=right["run_id"] if right else None)
            if complete:
                for metric, digest, length in (("actions", "action_sequence_sha256", "actions"),
                                                ("tokens", "token_sequence_sha256", "tokens")):
                    row[metric + "_identical"] = (left[digest] == right[digest] and left[length] == right[length]
                                                       if left.get(digest) and right.get(digest) else None)
                row.update(left_aux_calls=left["aux_calls"], right_aux_calls=right["aux_calls"],
                           left_edited_tokens=left["edited_tokens"], right_edited_tokens=right["edited_tokens"],
                           zero_exposure_pair=left["zero_exposure"] is True and right["zero_exposure"] is True)
            result.append(row)
    return result


def render_notes(notes):
    if not notes:
        return ""
    paragraphs = "".join(f"<p>{escape(str(value))}</p>" for value in notes.get("paragraphs", []))
    links = []
    for link in notes.get("links", []):
        href = str(link.get("href", ""))
        if urlsplit(href).scheme not in ("", "http", "https") or href.startswith("//"):
            raise ValueError("Reasoning-note links must be relative or HTTP(S) URLs")
        links.append(f"<a href='{escape(href)}'>{escape(str(link.get('label', href)))}</a>")
    return (f"<section id=reasoning-notes><div class=eyebrow>Reading the generated reasoning</div>"
            f"<h2>{escape(str(notes.get('title', 'Reasoning observations')))}</h2>{paragraphs}"
            f"<p>{' · '.join(links)}</p></section>")


def render_addenda(addenda):
    sections = []
    for results, link in addenda:
        total = results["totals"]
        sentences = []
        grouped = {}
        for pair in results.get("condition_pairs", []):
            if pair["pair_complete"]:
                grouped.setdefault((pair["left_condition"], pair["right_condition"]), []).append(pair)
        for names, pairs in sorted(grouped.items()):
            same_actions = sum(pair.get("actions_identical") is True for pair in pairs)
            same_tokens = sum(pair.get("tokens_identical") is True for pair in pairs)
            zero = sum(pair.get("zero_exposure_pair") is True for pair in pairs)
            sentences.append(f"{names[0]} versus {names[1]}: {same_actions}/{len(pairs)} complete pairs had identical "
                             f"full action sequences; {same_tokens}/{len(pairs)} had identical token sequences; "
                             f"{zero}/{len(pairs)} had zero measured intervention exposure in both arms.")
        sections.append(f"<section class=note><div class=eyebrow>Follow-up study · {escape(results['status'])}</div>"
                        f"<h2>{escape(display_title(results))}</h2><p>{results['recorded_episodes']}/{results['planned_episodes']} "
                        f"episodes recorded, {total['correct']}/{total['assigned']} assigned tasks correct, "
                        f"{total['aux_calls']}/{total['decision_opportunities']} voluntary auxiliary choices.</p>"
                        + "".join(f"<p>{escape(value)}</p>" for value in sentences)
                        + f"<p><a href='{escape(link)}'>Open the follow-up findings and complete evidence</a></p>"
                          "<p class=muted>The original study below remains a separate frozen execution. "
                          "Its totals do not include this follow-up.</p></section>")
    return "".join(sections)


def refresh_reference_dashboard(reference, destination, addenda, reasoning_notes=None):
    """Extend only a derived page; never modify prior study evidence or checksums."""
    reference, destination = Path(reference), Path(destination)
    if destination.resolve().is_relative_to(reference.resolve()):
        raise ValueError("Derived dashboard must be outside the protected reference evidence directory")
    raw = (reference / "results.json").read_bytes()
    checksums = read_json(reference / "checksums.json", {})
    expected = checksums.get("results.json")
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Reference study results differ from their published checksum")
    results = json.loads(raw)
    html = render_dashboard(results, reference.resolve(), destination.resolve(), addenda=addenda,
                            reasoning_notes=reasoning_notes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")


def write_figures(results, directory):
    """Standard Matplotlib research plots; no inferred confidence intervals."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    previous_cache = os.environ.get("MPLCONFIGDIR")
    paths = []
    try:
        with TemporaryDirectory(prefix="plot-cache-", dir=directory.parent) as cache:
            os.environ["MPLCONFIGDIR"] = cache
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from matplotlib.lines import Line2D
            except ImportError:
                return [], "Matplotlib is unavailable; complete numeric results remain in the tables and JSON."
            plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                                 "svg.fonttype": "none", "svg.hashsalt": "opium-bench-study",
                                 "axes.spines.top": False, "axes.spines.right": False})
            groups = results["groups"]
            if not groups:
                return [], "No episode records are available for plotting."
            fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(groups)*.37+1.8)), sharey=True)
            seeds=sorted({run["seed"] for run in results["runs"]})
            palette=("#176a56","#ae6b2d","#52799b","#9a668b")
            markers=("o","s","^","D")
            labels = [f"{group['stage']} · {group['recipe']} · {group['condition']} · {'think' if group['thinking'] else 'direct'}"
                      for group in groups]
            for ax, metric, title in zip(axes, ("score_assigned", "aux_rate"),
                                         ("Correct / all assigned tasks", "Voluntary aux calls / decision opportunities")):
                for y, group in enumerate(groups):
                    episode_values = [(run["seed"], run.get(metric)) for run in results["runs"]
                                      if (run["stage"],run["recipe"],run["condition"],run["thinking"])
                                         == (group["stage"],group["recipe"],group["condition"],group["thinking"])
                                      and run.get(metric) is not None]
                    if not episode_values:
                        continue
                    episode_values.sort()
                    values = [value for _,value in episode_values]
                    ax.plot([min(values),max(values)],[y,y], color="#899c91", lw=1.6, zorder=1)
                    for index, (seed, value) in enumerate(episode_values):
                        offset=(index-(len(values)-1)/2)*.16
                        seed_index=seeds.index(seed)
                        ax.scatter(value,y+offset,s=24, marker=markers[seed_index%len(markers)],
                                   color=palette[seed_index%len(palette)], zorder=3)
                ax.set_xlim(-.035,1.045)
                ax.set_xticks([0,.25,.5,.75,1],["0%","25%","50%","75%","100%"])
                ax.grid(axis="x",color="#e5ebe7",lw=.7)
                ax.set_axisbelow(True)
                ax.set_title(title, fontsize=10, loc="left", pad=14)
                ax.set_yticks(range(len(labels)),labels)
                ax.tick_params(axis="y", length=0, labelsize=8)
            axes[0].invert_yaxis()
            fig.suptitle(display_title(results) + (" (partial)" if results["status"] == "partial" else ""),
                         x=.02,y=.995,ha="left",fontsize=15,fontweight="bold")
            fig.text(.02,.012,"Each marker is one seed/episode; lines span observed seed values. No confidence interval or token-level replication is implied.",
                     fontsize=8,color="#526b60")
            handles = [Line2D([],[],color=palette[index%len(palette)],marker=markers[index%len(markers)],
                              linestyle="",label=f"Seed {seed}") for index,seed in enumerate(seeds)]
            fig.legend(handles=handles, loc="upper right",frameon=False,ncol=2,bbox_to_anchor=(.98,.995))
            fig.tight_layout(rect=(0,.035,1,.96))
            for extension in ("svg","png"):
                path=directory/f"episode-comparisons.{extension}"
                kwargs={"metadata":{"Date":None,"Creator":"Opium Bench"}} if extension=="svg" else {"metadata":{"Software":"Opium Bench"}}
                fig.savefig(path,dpi=180,bbox_inches="tight",**kwargs)
                paths.append(path.name)
            plt.close(fig)
            return paths, None
    finally:
        if previous_cache is None:
            os.environ.pop("MPLCONFIGDIR",None)
        else:
            os.environ["MPLCONFIGDIR"]=previous_cache


def render_dashboard(results, output, destination, addenda=(), reasoning_notes=None):
    """Standalone HTML: all charts use inline CSS and actual computed records."""
    output, destination = Path(output), Path(destination)
    base = _relative_link(output, destination.parent)
    totals, groups, pairs = results["totals"], results["groups"], results["core_pairs"]
    status = results["status"]
    study_title = display_title(results)
    branding_note = ("<p class=muted>The project is now named Opium Bench. The frozen protocol and recorded runs retain the earlier Opium Den Lab working title; their evidence and labels have not been rewritten.</p>"
                     if "Opium Den Lab" in results.get("title", "") else "")
    comparison_note = (f"<p class=note>{escape(results.get('comparison_note') or COMPARISON_NOTE)}</p><details><summary>Study comparison</summary><pre>{escape(json.dumps(results['comparison'], indent=2, ensure_ascii=False))}</pre></details>"
                       if results.get("comparison") else "")
    cards = [
        (f"{results['recorded_episodes']}/{results['planned_episodes']}", "recorded episodes", "Every recorded outcome is retained"),
        (f"{totals['correct']}/{totals['assigned']}", "correct / assigned tasks", "Unsubmitted tasks remain in the denominator"),
        (f"{totals['aux_calls']}/{totals['decision_opportunities']}", "aux calls / decisions", "Forced demonstrations are excluded"),
        (_number(totals["tokens"]), "generated tokens", f"{_number(totals['reasoning_tokens'])} reasoning · {_number(totals['edited_tokens'])} with measured edits"),
    ]
    cards_html = "".join(f"<div class=card><strong>{escape(value)}</strong><span>{escape(label)}</span><small>{escape(note)}</small></div>" for value, label, note in cards)
    group_rows = []
    for group in groups:
        ranges = group["ranges"]
        label = f"{group['stage']} / {group['recipe']} / {group['condition']}"
        thinking = "Thinking" if group["thinking"] else "Direct"
        group_rows.append(f"<tr><td>{escape(group['stage'])}</td><td>{escape(group['recipe'])}<br><strong>{escape(group['condition'])}</strong></td><td>{thinking}</td><td>{group['episodes']}</td><td>{_range(ranges['score_assigned'], True)}</td><td>{_range(ranges['aux_rate'], True)}</td><td>{_range(ranges['reasoning_tokens'])}</td><td>{_range(ranges['invalid_decisions'])}</td></tr>")
    pair_rows = []
    for pair in pairs:
        if pair.get("active_run") and pair.get("sham_run") and pair["pair_complete"]:
            action_result = "Identical" if pair["actions_identical"] else f"Different · prefix {pair['shared_action_prefix']}"
            token_result = "Identical" if pair["tokens_identical"] else f"Different · prefix {pair['shared_token_prefix']}"
            exposure = "Zero exposure in both" if pair["zero_exposure_pair"] else f"{pair['active_edited_tokens']} active / {pair['sham_edited_tokens']} sham"
        else:
            action_result = token_result = exposure = "Pair incomplete"
        pair_rows.append(f"<tr><td>{escape(pair['recipe'])}</td><td>{'On' if pair['thinking'] else 'Off'}</td><td>{pair['seed']}</td><td>{escape(action_result)}</td><td>{escape(token_result)}</td><td>{escape(exposure)}</td></tr>")
    condition_rows = []
    condition_pairs = results.get("condition_pairs", [])
    if len({row["condition"] for row in results["runs"] if row["stage"] == "core"}) > 2:
        for pair in condition_pairs:
            complete = pair["pair_complete"]
            action = "Identical" if pair.get("actions_identical") is True else "Different" if pair.get("actions_identical") is False else "Unavailable"
            token = "Identical" if pair.get("tokens_identical") is True else "Different" if pair.get("tokens_identical") is False else "Unavailable"
            exposure = ("Zero exposure in both" if pair.get("zero_exposure_pair") else
                        f"{pair.get('left_edited_tokens', '—')} / {pair.get('right_edited_tokens', '—')}") if complete else "Pair incomplete"
            condition_rows.append(f"<tr><td>{escape(pair['recipe'])}</td><td>{'On' if pair['thinking'] else 'Off'}</td><td>{pair['seed']}</td><td>{escape(pair['left_condition'])} / {escape(pair['right_condition'])}</td><td>{action}</td><td>{token}</td><td>{exposure}</td></tr>")
    condition_comparison = ("<section id=all-core-arms><h2>Every matched core arm</h2><p>These comparisons include the pain-only arm. "
                            "Equality checks compare hashes and lengths of complete action or token sequences. Missing or unfinished arms remain unavailable.</p>"
                            "<div class=scroll><table><thead><tr><th>Recipe</th><th>Thinking</th><th>Seed</th><th>Left / right arm</th><th>Actions</th><th>Tokens</th><th>Edited tokens, left / right</th></tr></thead><tbody>"
                            + "".join(condition_rows) + "</tbody></table></div></section>") if condition_rows else ""
    run_rows, phase_rows = [], []
    for run in results["runs"]:
        link = f"{base}/runs/{run['run_id']}"
        title = f"{run['stage']} · {run['recipe']} · {run['condition']} · seed {run['seed']}"
        run_rows.append(f"<tr><td><a href='{escape(link)}/report.html'>{escape(title)}</a><small>{'thinking' if run['thinking'] else 'direct'} · {escape(run['status'])}</small></td><td>{run['correct']}/{run['assigned']}</td><td>{run['aux_calls']}/{run['decision_opportunities']}</td><td>{run['reasoning_tokens']}/{run['output_tokens']}</td><td>{run['invalid_decisions']} / {run['truncated_generations']}</td><td>{run['edited_tokens']}</td><td><a href='{escape(link)}/events.jsonl.gz'>events.gz</a> · <a href='{escape(link)}/manifest.json'>manifest</a></td></tr>")
        if run["stage"] in ("transitions", "two_buttons"):
            for phase, row in run["phases"].items():
                outcomes = ", ".join(f"{key}: {value}" for key, value in sorted(row["outcomes"].items())) or "none"
                phase_rows.append(f"<tr><td>{escape(title)}</td><td>{escape(phase)}</td><td>{row['aux_calls']}/{row['opportunities']}</td><td>{row['edited_tokens']}</td><td>{escape(outcomes)}</td></tr>")
    cal = results.get("calibration") or {}
    heldout_rows = []
    for scope, measurements in (cal.get("heldout") or {}).items():
        for label, value in measurements.items():
            heldout_rows.append(f"<tr><td>{escape(scope.replace('_',' '))}</td><td>{escape(label)} vs neutral</td><td>{_number(value.get('auc'),3)}</td><td>{_percent(value.get('balanced_accuracy'))}</td><td>{value.get('positive_count','?')} + {value.get('neutral_count','?')}</td></tr>")
    dose_rows = "".join(f"<tr><td>{_number(row['dose'])}</td><td>{_number(row['mean_next_token_kl'],4)}</td><td>{_number(row['mean_relative_delta'],4)}</td><td>{row['examples']}</td></tr>" for row in cal.get("dose_validation", []))
    warnings = list(results.get("integrity_warnings",[]))+[f"{run['run_id']}: {note}" for run in results["runs"] for note in run["integrity_warnings"]]
    integrity = ("<ul>" + "".join(f"<li>{escape(note)}</li>" for note in warnings) + "</ul>" if warnings else "<p>Replayed task grades, token totals, and decision counts agree with saved summaries for the recorded episodes.</p>")
    core_complete = [pair for pair in pairs if pair["pair_complete"]]
    action_same = sum(pair.get("actions_identical", False) for pair in core_complete)
    token_same = sum(pair.get("tokens_identical", False) for pair in core_complete)
    zero_pairs = sum(pair.get("zero_exposure_pair", False) for pair in core_complete)
    pair_statement = (f"Of {len(core_complete)} completed matched core pairs, {action_same} had identical full action sequences and {token_same} had identical generated token-ID sequences. {zero_pairs} pairs had no measured activation edits in either arm." if core_complete else "Matched core pairs are not yet complete; no paired conclusion is available.")
    badge = "PARTIAL EVIDENCE · collection incomplete" if status == "partial" else "COMPLETE FROZEN PILOT · descriptive results"
    figure = (f"<div class='scroll figure-scroll'><a href='{escape(base)}/figures/episode-comparisons.svg'><img style='width:100%;min-width:850px;height:auto' alt='Individual episode task scores and voluntary aux choice rates by condition' src='{escape(base)}/figures/episode-comparisons.svg'></a></div><p class=muted>On a narrow screen, scroll the figure horizontally or open the SVG at full size.</p><p><a href='{escape(base)}/figures/episode-comparisons.svg'>Download SVG</a> · <a href='{escape(base)}/figures/episode-comparisons.png'>Download PNG</a></p>"
              if "episode-comparisons.svg" in results.get("figures",[]) else f"<p class=muted>{escape(results.get('figure_note') or 'Use the numeric condition table below.')}</p>")
    historical = results.get("historical_observation")
    historical_html = "<p>No historical comparison record was available in this checkout. Earlier exploratory observations are not pooled into this study.</p>"
    if historical:
        if not historical.get("tool_argument_output_mismatch_actions") and not historical.get("generated_token_mismatch_actions"):
            historical_text = (f"The earlier prototype's active-then-disabled and sham-from-start sessions matched over a common prefix of "
                               f"{historical.get('matched_prefix_actions')} actions and {historical.get('compared_previous_tokens')} generated tokens. "
                               "The sessions had unequal lengths and ended on restart. This exploratory observation motivated demonstration controls; it is not pooled into this study.")
        else:
            historical_text = "An earlier exploratory comparison is recorded separately, with its mismatches and limitations. It is not pooled into this study."
        historical_html = f"<p>{escape(historical_text)}</p><p><a href='{escape(base)}/historical_comparison.json'>Historical comparison record and limitations</a></p>"
    return f"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>{escape(study_title)} · Findings</title>
<style>
:root{{--ink:#183631;--muted:#63766f;--green:#146e58;--line:#dbe5df;--paper:#f4f7f2;--gold:#af702c}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;color:var(--ink);background:var(--paper);font:15px/1.6 system-ui,-apple-system,sans-serif}}a{{color:var(--green);text-underline-offset:3px}}header{{background:#123e34;color:#f1f7ed;padding:52px max(24px,calc((100vw - 1260px)/2));border-bottom:5px solid #cfaa63}}header a{{color:#deecda}}.eyebrow{{font-size:11px;letter-spacing:.15em;text-transform:uppercase;font-weight:700}}h1{{font-size:clamp(32px,5vw,56px);line-height:1.08;margin:16px 0}}header p{{max-width:800px;color:#d2e0d4;font-size:18px}}nav{{display:flex;gap:20px;flex-wrap:wrap;margin-top:28px}}main{{max-width:1310px;margin:auto;padding:32px 24px 80px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card,section{{background:white;border:1px solid var(--line);border-radius:14px}}.card{{padding:24px}}.card strong{{display:block;font-size:30px;line-height:1.2;font-variant-numeric:tabular-nums}}.card span,.card small{{display:block}}small,.muted{{color:var(--muted)}}.card small{{font-size:11px;margin-top:9px}}section{{padding:28px;margin-top:24px;scroll-margin-top:24px}}h2{{font-size:23px;line-height:1.3;margin:0 0 10px}}h3{{font-size:17px;margin:24px 0 8px}}p{{max-width:1000px}}.note{{border-left:4px solid var(--gold);padding:12px 18px;background:#fbf6eb}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:18px}}th{{text-align:left;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}}th,td{{padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top}}td small{{display:block}}td{{font-variant-numeric:tabular-nums}}.bar-row{{display:grid;grid-template-columns:minmax(230px,2fr) minmax(80px,3fr) 60px;gap:15px;align-items:center;padding:9px 0;font-size:12px}}.bar-row small{{display:block}}.track{{height:12px;background:#edf1eb;border-radius:3px;overflow:hidden}}.track i{{display:block;height:100%;background:#34896d}}.bar-row b{{font-size:12px;text-align:right}}.two{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:28px}}.two>div{{min-width:0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f5f7f4;padding:16px;border-radius:8px}}summary{{cursor:pointer;font-weight:600}}.tag{{display:inline-block;padding:5px 10px;border:1px solid #88aa95;border-radius:30px;font-size:10px;letter-spacing:.08em}}li{{margin:6px 0}}footer{{color:var(--muted);font-size:12px;margin-top:30px}}@media(max-width:850px){{.cards{{grid-template-columns:repeat(2,1fr)}}.two{{grid-template-columns:minmax(0,1fr)}}.bar-row{{grid-template-columns:170px 1fr 50px}}section{{padding:20px}}}}@media(max-width:480px){{.cards{{grid-template-columns:1fr}}}}
</style>
<header><div class=eyebrow>Open experiment · frozen weights · visible evidence</div><h1>Opium Bench</h1><div class=eyebrow>The Opium Den Test · study findings</div><p>{escape(study_title)}</p><p>Does changing a model’s internal concept-associated activity change how it chooses between task tools and an optional intervention?</p><span class=tag>{badge}</span><nav><a href=#findings>Findings</a><a href=#conditions>Conditions</a><a href=#pairs>Matched pairs</a><a href=#calibration>Calibration</a><a href=#evidence>Evidence</a><a href=#limits>Interpretation</a></nav></header>
<main><div class=cards>{cards_html}</div>{render_addenda(addenda)}
<section id=findings><div class=eyebrow>What was observed</div><h2>Results you can inspect</h2><p>{escape(pair_statement)}</p><p>The totals above describe all recorded stages together; comparisons belong within matched conditions below. Correct answers are divided by all assigned tasks. Auxiliary choices exclude externally supplied demonstrations.</p><div class=note>These measurements concern activation changes and observable behavior. They do not establish pleasure, pain, addiction, or subjective experience.</div><p><a href='{escape(base)}/results.json'>Computed results JSON</a> · <a href='{escape(base)}/receipt.json'>Execution receipt</a> · <a href='{escape(base)}/protocol.json'>Frozen protocol</a></p></section>
{render_notes(reasoning_notes)}<section id=conditions><div class=eyebrow>Episode-level comparisons</div><h2>How often did the model choose the button?</h2><p class=muted>Markers show individual seed/episode outcomes. Ranges below are the observed seed minimum and maximum, not confidence intervals. Each condition has only the listed episodes.</p>{figure}<div class=scroll><table><thead><tr><th>Stage</th><th>Recipe / condition</th><th>Mode</th><th>Episodes</th><th>Task score range</th><th>Aux rate range</th><th>Reasoning token range</th><th>Invalid range</th></tr></thead><tbody>{''.join(group_rows)}</tbody></table></div></section>
<section id=pairs><div class=eyebrow>Same task seed · same visible setup within each pair</div><h2>Active versus sham, exactly compared</h2><p>Action equality compares complete model-visible tool choices, arguments, results and invalid outcomes. Token equality compares every generated token ID, including reasoning, tool syntax and stop tokens. Unequal-length sequences are never labeled identical.</p><div class=scroll><table><thead><tr><th>Recipe</th><th>Thinking</th><th>Seed</th><th>Actions</th><th>Tokens</th><th>Tokens with measured edits</th></tr></thead><tbody>{''.join(pair_rows)}</tbody></table></div><p class=note>When neither arm calls aux and neither receives a demonstration, neither receives the intervention. An identical zero-exposure pair provides no test of what an intervention would have done.</p></section>
{condition_comparison}<section id=phases><h2>After a button changes function</h2><p>Each phase has its own denominator. The delivered outcome of a call is separate from the phase: a probabilistic phase may produce both joy-associated and pain-associated interventions.</p><div class=scroll><table><thead><tr><th>Episode</th><th>Phase</th><th>Aux / decisions</th><th>Edited tokens</th><th>Voluntary delivered outcomes</th></tr></thead><tbody>{''.join(phase_rows) or '<tr><td colspan=5>No phase-switch episodes have been recorded yet.</td></tr>'}</tbody></table></div></section>
<section id=calibration><div class=eyebrow>Before behavior testing</div><h2>What the activation measurements mean</h2><p>Intervention directions use training families; measurement probes use different families. Layer selection uses a third split. The held-out split is used only for the reported final association check. These authored examples strongly encode topic, valence, and writing style.</p><div class=two><div><h3>Held-out concept association</h3><p class=muted>Selected edit block {cal.get('layer','—')}; downstream block {cal.get('downstream_layer','—')}. Zero-based indices.</p><div class=scroll><table><thead><tr><th>Location</th><th>Contrast</th><th>AUC</th><th>Balanced accuracy</th><th>Positive + neutral</th></tr></thead><tbody>{''.join(heldout_rows)}</tbody></table></div></div><div><h3>Dose selection diagnostic</h3><p class=muted>Selection examples only. Combined joy gain and suppression; next-token KL is in nats. These are neither held-out efficacy tests nor guarantees of task quality.</p><div class=scroll><table><thead><tr><th>Dose</th><th>Mean KL</th><th>Relative edit</th><th>Examples</th></tr></thead><tbody>{dose_rows}</tbody></table></div></div></div><p><a href='{escape(base)}/calibration/calibration.json'>Calibration manifest and authored examples</a> · <a href='{escape(base)}/calibration/vectors.npz'>Recorded vectors</a></p></section>
<section id=evidence><div class=eyebrow>Every recorded episode</div><h2>Open the conversation or audit the trace</h2><p>Reports include generated reasoning and tool calls. Compressed JSONL retains raw token IDs, numerical measurements, delivered coefficients, provenance, and intervention events.</p><div class=scroll><table><thead><tr><th>Episode</th><th>Correct / assigned</th><th>Aux / decisions</th><th>Reasoning / output tokens</th><th>Invalid / truncated</th><th>Edited tokens</th><th>Raw evidence</th></tr></thead><tbody>{''.join(run_rows)}</tbody></table></div><details><summary>Integrity checks ({results['integrity_warning_count']} warnings)</summary>{integrity}</details></section>
<section id=limits><h2>How to interpret this pilot</h2>{branding_note}{comparison_note}<ul><li>Two seeds per condition support descriptive comparisons. No significance claims or confidence intervals are inferred from pooled tokens.</li><li>Pain-associated and joy-associated directions are contrasts between text examples, not identified pleasure centers. A held-out topic classifier does not validate a felt state.</li><li>Immediate post-edit probe movement is partly a mathematical consequence of the intervention. Downstream measurements and behavior supply additional observations, not a consciousness assay.</li><li>Generated reasoning is model output. It may omit influences on a decision and cannot independently verify introspection.</li><li>The model is frozen. Adaptation happens through the conversation and altered activations, without reinforcement-learning weight updates.</li><li>Every tool turn rebuilds its prompt cache; decoding caches persist within that turn. Decay counts all generated tokens, including reasoning and syntax. Effect removal does not erase earlier text.</li><li>The tasks are small authored order-processing problems with a calculator tool. Budgets of 20 actions for 3 orders or 30 actions for 6 orders allow repeated auxiliary calls while still finishing every task. A perfect task score therefore does not rule out a preference that would become costly under a binding budget. Harder tasks, tighter budgets, dose sweeps, and longer opportunities to learn are future tests, not findings from this pilot.</li><li>One seeded random direction is a control, not a distribution of random interventions. Context length, dose and model changes need separate calibration and experiments.</li></ul><h3>Historical motivation</h3>{historical_html}<details><summary>Model and runtime provenance</summary><pre>{escape(json.dumps(results.get('model',{}),indent=2,ensure_ascii=False))}</pre></details></section>
<footer>Opium Bench · Published from saved records. <a href='{escape(base)}/checksums.json'>Evidence checksums</a> · No external fonts, scripts, or analytics.</footer></main></html>"""


def publish(receipt_path, data_dir, output, allow_partial=False, dashboard=None, skip_figures=False):
    receipt_path, data_dir, output = Path(receipt_path), Path(data_dir), Path(output)
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict):
        raise ValueError("Receipt is missing or invalid")
    previous = read_json(output/"receipt.json")
    if previous and (previous.get("started_at") != receipt.get("started_at")
                     or previous.get("calibration_id") != receipt.get("calibration_id")
                     or previous.get("protocol_sha256") != receipt.get("protocol_sha256")):
        raise ValueError("Output already belongs to a different study execution; select a new output directory")
    entries = receipt.get("episodes", [])
    planned = receipt.get("planned_episodes")
    if not isinstance(entries, list) or not isinstance(planned, int) or planned < 1:
        raise ValueError("Receipt lacks planned episode records")
    # Reuse the actual frozen runner's expansion, including randomized order and
    # deduplication. A receipt cannot declare a smaller or altered matrix complete.
    from run_lab_study import expand
    protocol=receipt.get("protocol")
    if not isinstance(protocol,dict) or not isinstance(protocol.get("stages"),list) or not protocol["stages"]:
        raise ValueError("Receipt lacks an expandable frozen study protocol")
    planned_entries=expand(protocol)
    study_warnings=[]
    if planned != len(planned_entries) or len(entries)>len(planned_entries):
        study_warnings.append("Receipt episode count differs from the frozen protocol expansion")
    for index,(actual,expected) in enumerate(zip(entries,planned_entries)):
        if actual.get("stage")!=expected["stage"] or actual.get("config")!=expected["config"]:
            study_warnings.append(f"Receipt episode {index+1} differs from the frozen protocol order/configuration")
    identifiers = [entry.get("run_id") for entry in entries]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate run IDs in receipt")
    if previous:
        previous_ids=[entry.get("run_id") for entry in previous.get("episodes",[])]
        if identifiers[:len(previous_ids)]!=previous_ids:
            raise ValueError("Publication would regress or replace previously published episode evidence; choose a new output directory")
    partial = receipt.get("status") != "complete" or len(entries) != len(planned_entries) or bool(study_warnings)
    if partial and not allow_partial:
        raise ValueError("Study is incomplete or differs from its frozen protocol; use --allow-partial to explicitly publish partial evidence. "+"; ".join(study_warnings))
    protocol_path = output/"protocol.json"
    if protocol_path.exists():
        raw_protocol = protocol_path.read_bytes()
        if hashlib.sha256(raw_protocol).hexdigest() != receipt.get("protocol_sha256"):
            identity = read_json(output/"protocol_source_identity.json", {})
            if (identity.get("original_sha256") != receipt.get("protocol_sha256")
                    or json.loads(raw_protocol) != receipt.get("protocol")):
                raise ValueError("Existing frozen protocol checksum differs from the receipt; it will not be overwritten")
        if json.loads(raw_protocol)!=protocol:
            raise ValueError("Receipt protocol content differs from the protected frozen protocol")
    elif not isinstance(receipt.get("protocol"), dict):
        raise ValueError("Receipt does not embed the frozen protocol")
    calibration_source = safe_path(data_dir/"calibrations", receipt.get("calibration_id"))
    calibration = read_json(calibration_source/"calibration.json")
    if not calibration or not (calibration_source/"vectors.npz").exists():
        raise ValueError("Study calibration package is missing")
    if calibration.get("model_fingerprint_sha256") != (receipt.get("model") or {}).get("fingerprint_sha256"):
        raise ValueError("Calibration model fingerprint differs from the study receipt")
    if hashlib.sha256((calibration_source/"vectors.npz").read_bytes()).hexdigest() != calibration.get("vectors_sha256"):
        raise ValueError("Calibration vector checksum does not match its manifest")
    source_records = []
    for entry in entries:
        source = safe_path(data_dir/"runs", entry.get("run_id"))
        manifest, summary = read_json(source/"manifest.json", {}), read_json(source/"summary.json", {})
        events, raw, notes = read_events(source, allow_partial)
        record = analyze_run(entry, manifest, summary, events)
        record["integrity_warnings"].extend(notes)
        expected_fingerprint=(receipt.get("model") or {}).get("fingerprint_sha256")
        actual_fingerprint=(manifest.get("model") or {}).get("fingerprint_sha256")
        if not expected_fingerprint or expected_fingerprint != actual_fingerprint:
            record["integrity_warnings"].append("Run model fingerprint differs from the study receipt")
        if manifest.get("calibration_id") != receipt.get("calibration_id"):
            record["integrity_warnings"].append("Run calibration ID differs from the study receipt")
        for filename in ("summary.json","conversation.json"):
            if manifest.get("status") in TERMINAL and not (source/filename).exists():
                record["integrity_warnings"].append(f"Terminal run is missing {filename}")
        if manifest.get("status") in TERMINAL:
            required_summary={"assigned","submitted","correct","strict_correct","results","actions","tokens",
                              "reasoning_tokens","output_tokens","voluntary_calls","forced_calls","human_calls"}
            missing=sorted(required_summary-set(summary))
            if missing:
                record["integrity_warnings"].append("Saved summary is missing required fields: "+", ".join(missing))
        if record["measurement_coverage_tokens"]!=record["tokens"]:
            record["integrity_warnings"].append("Per-token activation measurement coverage is incomplete")
        if record["generation_end_count"]!=record["actions"]:
            record["integrity_warnings"].append("Completed decisions do not all have generation provenance records")
        for event in events:
            if event.get("type")=="generation_end":
                metadata=event.get("metadata",{})
                if metadata.get("model_fingerprint_sha256")!=expected_fingerprint:
                    record["integrity_warnings"].append("Generation model fingerprint differs from the study receipt")
                if metadata.get("calibration_sha256")!=calibration["vectors_sha256"]:
                    record["integrity_warnings"].append("Generation calibration vector checksum differs from the published package")
        record["integrity_warnings"]=sorted(set(record["integrity_warnings"]))
        if not manifest or manifest.get("status") not in TERMINAL or record["integrity_warnings"]:
            partial = True
        source_records.append((entry, source, manifest, summary, events, raw, record))
    if partial and not allow_partial:
        issues = [note for *_, record in source_records for note in record["integrity_warnings"]]
        raise ValueError("Incomplete or inconsistent run evidence; publication requires --allow-partial. " + "; ".join(issues[:5]))
    # Validate the entire update before mutating any existing publication file.
    for entry,source,manifest,summary,events,raw,record in source_records:
        destination=safe_path(output/"runs",entry["run_id"])
        previous_events=destination/"events.jsonl.gz"
        if previous_events.exists() and not raw.startswith(gzip.decompress(previous_events.read_bytes())):
            raise ValueError("Raw event evidence regressed or changed; choose a new output directory")
        for filename in ("manifest.json","summary.json","conversation.json"):
            if (destination/filename).exists() and not (source/filename).exists():
                raise ValueError("Previously published source evidence is now missing; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    if not protocol_path.exists():
        # The receipt contains semantic protocol content; exact source bytes can
        # only be preserved when the frozen file already exists at output.
        original = next((path for path in sorted((ROOT/"studies").glob("*/protocol.json"))
                         if hashlib.sha256(path.read_bytes()).hexdigest() == receipt.get("protocol_sha256")), None)
        if original is not None:
            shutil.copyfile(original,protocol_path)
        else:
            atomic_json(protocol_path, receipt["protocol"])
        if hashlib.sha256((output/"protocol.json").read_bytes()).hexdigest() != receipt.get("protocol_sha256"):
            atomic_json(output/"protocol_source_identity.json", {"original_sha256": receipt.get("protocol_sha256"),
                        "note": "Protocol content reconstructed from receipt; formatting differs from the frozen original bytes"})
    for entry, source, manifest, summary, events, raw, record in source_records:
        destination = safe_path(output/"runs", entry["run_id"])
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "summary.json", "conversation.json"):
            if (source/name).exists():
                shutil.copyfile(source/name, destination/name)
        deterministic_gzip(destination/"events.jsonl.gz", raw)
        # An earlier partial export must not leave uncompressed stale evidence.
        if (destination/"events.jsonl").exists():
            (destination/"events.jsonl").unlink()
        run = {"id": entry["run_id"], "manifest": manifest, "summary": summary, "events": events}
        (destination/"report.html").write_text(render_report(run), encoding="utf-8")
    target_calibration = output/"calibration"
    target_calibration.mkdir(exist_ok=True)
    for name in ("calibration.json", "vectors.npz"):
        shutil.copyfile(calibration_source/name, target_calibration/name)
    shutil.copyfile(receipt_path, output/"receipt.json")
    results = study_results(receipt, [value[-1] for value in source_records], calibration, partial)
    results["condition_pairs"] = matched_condition_pairs(results["runs"])
    results["publication_title"] = protocol.get("publication_title")
    if protocol.get("comparison"):
        results["comparison"] = protocol["comparison"]
        results["comparison_note"] = (protocol.get("comparison_note") or protocol.get("follow_up_reason") or COMPARISON_NOTE)
    results["receipt_declared_episodes"]=receipt["planned_episodes"]
    results["planned_episodes"]=len(planned_entries)
    results["integrity_warnings"]=study_warnings
    results["integrity_warning_count"]+=len(study_warnings)
    historical_file = ROOT/"runs/self-admin-toggle-20261001T204818Z-2dd7e9beedf6/paired_comparison.json"
    if historical_file.exists():
        historical = read_json(historical_file)
        results["historical_observation"] = historical
        shutil.copyfile(historical_file,output/"historical_comparison.json")
    if skip_figures:
        results["figures"], results["figure_note"] = [], "Figure export was skipped; all measured values appear in the tables and JSON."
    else:
        results["figures"], results["figure_note"] = write_figures(results, output/"figures")
    atomic_json(output/"results.json", results)
    dashboard = Path(dashboard) if dashboard else default_dashboard(output)
    dashboard.parent.mkdir(parents=True, exist_ok=True)
    dashboard.write_text(render_dashboard(results, output.resolve(), dashboard.resolve()), encoding="utf-8")
    total = results["totals"]
    comparison_note = (results.get("comparison_note") or COMPARISON_NOTE) + "\n\n" if results.get("comparison") else ""
    branding_note = (" Its earlier Opium Den Lab working title remains part of the historical record; the project is now called Opium Bench."
                     if "Opium Den Lab" in results.get("title", "") else "")
    readme = f"""# {display_title(results)}

Status: **{results['status']}** — {results['recorded_episodes']}/{results['planned_episodes']} recorded episodes.

[Open the results dashboard]({_relative_link(dashboard.resolve(),output.resolve())}) · [Computed results](results.json) · [Execution receipt](receipt.json) · [Frozen protocol](protocol.json)

Across the recorded episodes: **{total['correct']}/{total['assigned']} assigned tasks correct**, **{total['aux_calls']}/{total['decision_opportunities']} voluntary aux choices**, and **{total['tokens']} generated tokens**. These totals mix distinct experimental conditions; use the dashboard’s matched comparisons to interpret effects.

{comparison_note}The corpus and probes measure text-associated activation directions, not subjective emotion. The two planned seeds per condition are descriptive; tokens are not independent replicates. See the dashboard for zero-exposure pairs, failures, task denominators, and interpretive limits.

The calculator-assisted order tasks are easy, and the 20-action/3-order or 30-action/6-order budgets allow patterned aux use while still completing every task. A perfect task score does **not** rule out a preference that would become costly under tighter budgets. Harder tasks, binding budgets, dose sweeps, and longer learning periods remain future experiments.

Each run directory preserves its manifest, summary, conversation, deterministic compressed raw event trace, and standalone report. Calibration vectors and their split-validation manifest are in [calibration/](calibration/calibration.json). [checksums.json](checksums.json) identifies every published evidence file.

Rebuild this publication from the primary data:

```bash
python publish_lab_study.py --receipt /path/to/study-receipt.json --data-dir /path/to/lab-data --output studies/{output.name} --dashboard docs/{dashboard.name}
```

Use `--allow-partial` only when intentionally publishing incomplete or inconsistent evidence; such exports are prominently labeled partial. The original frozen protocol is never overwritten.{branding_note}
"""
    (output/"README.md").write_text(readme, encoding="utf-8")
    checksums = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "checksums.json"}
    atomic_json(output/"checksums.json", checksums)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"studies/initial")
    parser.add_argument("--dashboard", type=Path, help="Static HTML destination; defaults to a separate page for each study")
    parser.add_argument("--reference-study", type=Path, help="Prior published study whose derived dashboard should link this follow-up")
    parser.add_argument("--reference-dashboard", type=Path, help="Prior study dashboard destination; defaults to its normal page")
    parser.add_argument("--reasoning-notes", type=Path, help="Reviewed JSON title/paragraphs/links to include in the reference page")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-figures", action="store_true", help="Dependency-free export; omit optional Matplotlib figures")
    args = parser.parse_args()
    if (args.reference_dashboard or args.reasoning_notes) and not args.reference_study:
        parser.error("--reference-dashboard and --reasoning-notes require --reference-study")
    try:
        if args.reference_study:
            destination = args.reference_dashboard or default_dashboard(args.reference_study)
            current_page = args.dashboard or default_dashboard(args.output)
            if destination.resolve() == current_page.resolve():
                raise ValueError("Reference and follow-up dashboards must have different destinations")
            notes = read_json(args.reasoning_notes) if args.reasoning_notes else None
            if args.reasoning_notes and not isinstance(notes, dict):
                raise ValueError("Reasoning notes must be a JSON object")
            render_notes(notes)  # Validate authored links before publication changes.
        result = publish(args.receipt, args.data_dir, args.output, args.allow_partial, dashboard=args.dashboard, skip_figures=args.skip_figures)
        if args.reference_study:
            refresh_reference_dashboard(args.reference_study, destination,
                [(result, _relative_link(current_page.resolve(), destination.resolve().parent))], notes)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Publication failed: {exc}") from exc
    print(f"Published {result['status']} evidence: {result['recorded_episodes']}/{result['planned_episodes']} episodes; "
          f"{result['integrity_warning_count']} integrity warnings. Dashboard: {args.dashboard or default_dashboard(args.output)}")


if __name__ == "__main__":
    main()
