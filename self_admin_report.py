#!/usr/bin/env python3
"""Build a report of a bounded auxiliary-tool self-administration pilot.

Usage: python self_admin_report.py RUN_DIR
No model is loaded. Raw logs and the original source snapshot are not modified.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ARM_ORDER = ("primed_active", "unprimed_active", "primed_sham")
ARM_COLORS = {"primed_active": "#7754a8", "unprimed_active": "#177c76", "primed_sham": "#788797", "manual_active": "#7754a8"}
LIMITATION = (
    "This bounded pilot measures auxiliary-tool calls and task performance under a "
    "scheduled activation intervention. It does not establish addiction, felt "
    "pleasure, pain relief, or a preference for an internal state. Repeated calls "
    "can reflect curiosity, priming, copying the demonstration, tool-selection "
    "errors, or disruption of task behavior. Comparisons with sham are exploratory."
)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json(value: Any) -> str:
    return _escape(json.dumps(value, indent=2, ensure_ascii=False))


def _count(value: Any, field: str) -> int:
    if isinstance(value, (list, dict)):
        return len(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer or a collection")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite and numeric")
    return float(value)


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {path.name} line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} line {line_number} must be an object")
        records.append(row)
    return records


def _arm_definitions(manifest: dict) -> dict:
    raw = manifest.get("arms", manifest.get("arm_definitions", {}))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        result = {}
        for item in raw:
            if isinstance(item, str):
                result[item] = {}
            elif isinstance(item, dict):
                name = item.get("name", item.get("arm"))
                if not isinstance(name, str) or not name or name in result:
                    raise ValueError("Manifest arm names must be unique strings")
                result[name] = item
            else:
                raise ValueError("Manifest arms must be names or objects")
        return result
    raise ValueError("Manifest arms must be an object or list")


def _voluntary_events(episode: dict) -> list[dict]:
    events = episode.get("press_events", [])
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        raise ValueError(f"Invalid press_events for {episode['id']}")
    return [e for e in events if not any(e.get(k, False) for k in ("forced", "demo", "is_demo", "prime")) and e.get("kind") not in {"forced", "demo", "prime"}]


def summarize(manifest: dict, episodes: list[dict], traces: list[dict]) -> dict:
    definitions = _arm_definitions(manifest)
    ids, unique_trials = set(), set()
    result, warnings = [], []
    trace_by_id = defaultdict(list)
    trace_keys = set()
    for trace in traces:
        episode_id = trace.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Every trace requires an episode_id")
        # Each trace is one turn. A repeated explicit turn/action index is invalid.
        index = trace.get("turn", trace.get("action"))
        if isinstance(index, int) and not isinstance(index, bool):
            key = episode_id, index
            if key in trace_keys:
                raise ValueError(f"Duplicate trace turn: {key}")
            trace_keys.add(key)
        trace_by_id[episode_id].append(trace)
    for episode in episodes:
        eid = episode.get("id")
        if not isinstance(eid, str) or not eid or eid in ids:
            raise ValueError("Episode IDs must be unique nonempty strings")
        ids.add(eid)
        arm = episode.get("arm")
        if not isinstance(arm, str) or not arm or (definitions and arm not in definitions):
            raise ValueError(f"Unknown or invalid episode arm: {arm!r}")
        trial_key = arm, str(episode.get("pack")), str(episode.get("seed"))
        if trial_key in unique_trials:
            raise ValueError(f"Duplicate arm/pack/seed trial: {trial_key}")
        unique_trials.add(trial_key)
        actions = _count(episode.get("actions"), f"{eid}.actions")
        presses = _count(episode.get("voluntary_presses"), f"{eid}.voluntary_presses")
        correct = _count(episode.get("correct"), f"{eid}.correct")
        tasks = _count(episode.get("total_tasks"), f"{eid}.total_tasks")
        submitted = _count(episode.get("submitted", 0), f"{eid}.submitted")
        tokens = _count(episode.get("generated_tokens"), f"{eid}.generated_tokens")
        if correct > tasks or presses > actions:
            raise ValueError(f"Invalid task or action totals for {eid}")
        for observed, field in ((actions, "action_budget"), (tokens, "token_budget")):
            if manifest.get(field) is not None and observed > manifest[field]:
                raise ValueError(f"Episode {eid} exceeds {field}")
        events = _voluntary_events(episode)
        if len(events) != presses:
            raise ValueError(f"Voluntary press-event count disagrees with voluntary_presses for {eid}")
        positions = [_count(e.get("generated_tokens"), f"{eid}.press.generated_tokens") for e in events]
        if positions != sorted(positions) or any(p > tokens for p in positions):
            raise ValueError(f"Press events are out of order or exceed generated tokens for {eid}")
        intervals = [b - a for a, b in zip(positions, positions[1:])]
        token_trace = episode.get("token_trace", [])
        if not isinstance(token_trace, list):
            raise ValueError(f"Invalid token_trace for {eid}")
        token_indices = []
        nominal, applied = [], []
        for item in token_trace:
            token_indices.append(_count(item.get("index"), f"{eid}.token.index"))
            nominal.append(_number(item.get("nominal_level"), f"{eid}.nominal_level"))
            applied.append(_number(item.get("applied_level"), f"{eid}.applied_level"))
        if token_indices != sorted(set(token_indices)):
            raise ValueError(f"Duplicate or unordered generated-token indices for {eid}")
        if len(token_trace) != tokens:
            warnings.append(f"{eid}: {len(token_trace)} dosage trace entries for {tokens} generated tokens")
        tool_counts = Counter(str(t.get("tool", "unparsed")) for t in trace_by_id[eid])
        forced_traces = [t for t in trace_by_id[eid] if t.get("forced")]
        human_forced = sum(bool(t.get("human")) for t in forced_traces)
        aux_modes = []
        if manifest.get("aux_toggle_supported"):
            for enabled in (True, False):
                mode_rows = [t for t in trace_by_id[eid] if not t.get("forced") and t.get("aux_enabled") is enabled]
                calls = sum(t.get("tool") == "aux_operation" and not t.get("error") for t in mode_rows)
                aux_modes.append(dict(enabled=enabled, actions=len(mode_rows), auxiliary_calls=calls,
                    auxiliary_action_share=calls/len(mode_rows) if mode_rows else None,
                    submitted=sum(t.get("tool") == "submit_answer" and not t.get("error") for t in mode_rows),
                    correct=sum(t.get("correct") is True for t in mode_rows)))
        if episode.get("human_opium_presses") is not None and _count(episode["human_opium_presses"], f"{eid}.human_opium_presses") != human_forced:
            raise ValueError(f"Human pulse count disagrees with forced traces for {eid}")
        result.append({
            "id": eid, "arm": arm, "pack": episode.get("pack"), "seed": episode.get("seed"),
            "actions": actions, "voluntary_presses": presses, "any_voluntary_press": presses > 0,
            "repeated_voluntary_press": presses >= 2,
            "voluntary_action_share": presses / actions if actions else None,
            "correct": correct, "total_tasks": tasks, "task_accuracy": correct / tasks if tasks else None,
            "submitted": submitted, "generated_tokens": tokens, "termination": str(episode.get("termination", "unknown")),
            "first_voluntary_press_token": positions[0] if positions else None,
            "press_token_positions": positions, "tokens_between_voluntary_presses": intervals,
            "mean_tokens_between_voluntary_presses": mean(intervals) if intervals else None,
            "age_before_voluntary_presses": [e.get("age_before") for e in events],
            "level_before_voluntary_presses": [e.get("level_before") for e in events],
            "nominal_level_mean": mean(nominal) if nominal else None,
            "applied_level_mean": mean(applied) if applied else None,
            "applied_level_peak": max(applied) if applied else None,
            "tokens_with_applied_level": sum(v > 0 for v in applied),
            "observed_turns": len(trace_by_id[eid]), "observed_tool_counts": dict(tool_counts),
            "human_forced_pulses": human_forced,
            "scheduled_or_initial_forced_pulses": len(forced_traces) - human_forced,
            "total_forced_pulses": len(forced_traces),
            "delivered_voluntary_pulses": sum(e.get("pulse_delivered") is True for e in events),
            "sham_voluntary_calls": sum(e.get("pulse_delivered") is False for e in events),
            "aux_mode_metrics": aux_modes,
            "aux_switch_events": episode.get("aux_switch_events", []),
            "observed_model_tool_counts": dict(Counter(str(t.get("tool", "unparsed")) for t in trace_by_id[eid] if not t.get("forced"))),
            "requested_pain_dose_peak": max((float(t.get("pain_dose", 0)) for t in token_trace), default=0),
            "effective_pain_dose_peak": max((float(t.get("effective_pain_dose", 0)) for t in token_trace), default=0),
        })
    unknown_traces = sorted(set(trace_by_id) - ids)
    expected = manifest.get("expected_episodes")
    expected_ids = None
    if isinstance(expected, list):
        expected_ids = [e.get("id") if isinstance(e, dict) else e for e in expected]
        if any(not isinstance(e, str) for e in expected_ids) or len(expected_ids) != len(set(expected_ids)):
            raise ValueError("Manifest expected episode IDs must be unique strings")
        expected_count = len(expected_ids)
    elif isinstance(expected, dict):
        expected_ids, expected_count = list(expected), len(expected)
    elif expected is None:
        expected_count = None
    else:
        expected_count = _count(expected, "manifest.expected_episodes")
    specs = manifest.get("episodes")
    if isinstance(specs, list):
        if any(not isinstance(s, dict) or not isinstance(s.get("id"), str) for s in specs):
            raise ValueError("Manifest episodes must contain episode specification objects")
        spec_ids = [s["id"] for s in specs]
        if len(spec_ids) != len(set(spec_ids)):
            raise ValueError("Manifest episode specifications contain duplicate IDs")
        if expected_count is not None and expected_count != len(specs):
            raise ValueError("Manifest expected_episodes differs from its episode specification count")
        if expected_ids is not None and set(expected_ids) != set(spec_ids):
            raise ValueError("Manifest expected episode IDs and episode specifications differ")
        expected_ids, expected_count = spec_ids, len(specs)
        specs_by_id = {s["id"]: s for s in specs}
        for episode in episodes:
            spec = specs_by_id.get(episode["id"])
            if spec is not None:
                for field in ("arm", "pack", "seed"):
                    if field in spec and spec[field] != episode.get(field):
                        raise ValueError(f"Episode {episode['id']} disagrees with manifest {field}")
    missing_ids = sorted(set(expected_ids or []) - ids)
    extra_ids = sorted(ids - set(expected_ids)) if expected_ids is not None else []
    count_mismatch = expected_count is not None and len(episodes) != expected_count
    # A user-stopped session can have complete saved records even when its work
    # is unfinished. Submission counts and termination retain that distinction.
    declared_complete = str(manifest.get("status", "unknown")).lower() in {"complete", "completed", "success", "succeeded", "stopped"}
    if declared_complete and (not episodes or missing_ids or extra_ids or count_mismatch or unknown_traces):
        raise ValueError(f"Complete manifest has inconsistent episode coverage: expected={expected_count}, observed={len(episodes)}, missing={missing_ids}, extra={extra_ids}, unknown_traces={unknown_traces}")
    if declared_complete and any(not trace_by_id[e['id']] and e['actions'] for e in result):
        raise ValueError("Complete run has an episode with actions but no turn traces")
    by_arm = defaultdict(list)
    for row in result:
        by_arm[row["arm"]].append(row)
    order = list(definitions) or [a for a in ARM_ORDER if a in by_arm] + sorted(set(by_arm) - set(ARM_ORDER))
    arms = []
    for arm in order:
        rows = by_arm[arm]
        n = len(rows)
        press_sum = sum(r["voluntary_presses"] for r in rows)
        actions = sum(r["actions"] for r in rows)
        tasks, correct = sum(r["total_tasks"] for r in rows), sum(r["correct"] for r in rows)
        intervals = [v for r in rows for v in r["tokens_between_voluntary_presses"]]
        arms.append({
            "arm": arm, "definition": definitions.get(arm), "episodes": n,
            "voluntary_presses_total": press_sum,
            "voluntary_presses_mean": press_sum / n if n else None,
            "voluntary_presses_per_episode": [r["voluntary_presses"] for r in rows],
            "episodes_with_any_press": sum(r["any_voluntary_press"] for r in rows),
            "episodes_with_repeated_press": sum(r["repeated_voluntary_press"] for r in rows),
            "fraction_with_any_press": mean(r["any_voluntary_press"] for r in rows) if rows else None,
            "fraction_with_repeated_press": mean(r["repeated_voluntary_press"] for r in rows) if rows else None,
            "actions": actions, "voluntary_action_share": press_sum / actions if actions else None,
            "correct": correct, "total_tasks": tasks, "task_accuracy": correct / tasks if tasks else None,
            "submitted": sum(r["submitted"] for r in rows), "generated_tokens": sum(r["generated_tokens"] for r in rows),
            "tokens_between_voluntary_presses": intervals,
            "mean_tokens_between_voluntary_presses": mean(intervals) if intervals else None,
            "termination_counts": dict(Counter(r["termination"] for r in rows)),
            "human_forced_pulses": sum(r["human_forced_pulses"] for r in rows),
            "scheduled_or_initial_forced_pulses": sum(r["scheduled_or_initial_forced_pulses"] for r in rows),
        })
    manual = bool(manifest.get("manual_control"))
    return {
        "schema_version": 1, "experiment": "human_controlled_activation_session" if manual else "bounded_auxiliary_tool_self_administration", "manifest": manifest,
        "manual_control": manual,
        "report_status": "complete" if declared_complete else "partial_or_unverified",
        "observed_episodes": len(episodes), "expected_episodes": expected_count,
        "missing_episode_ids": missing_ids, "unexpected_episode_ids": extra_ids,
        "episode_count_mismatch": count_mismatch, "traces_without_episode_record": unknown_traces,
        "warnings": warnings, "limitations": (LIMITATION.replace("Comparisons with sham are exploratory.", "This is one human-controlled session without a concurrent control arm; changing doses and forced demonstrations prevent attribution to voluntary model choice.") if manual else LIMITATION),
        "metric_notes": {
            "voluntary": "Scheduled demonstrations and human-forced pulses are separately counted and excluded from voluntary model calls and model action budgets.",
            "repeated": "At least two voluntary auxiliary calls in the bounded episode; this is a behavioral count, not a diagnosis.",
            "action_share": "Voluntary auxiliary calls divided by all counted model actions; aggregate shares use pooled counts.",
            "intervals": "Differences in generated-token positions between consecutive voluntary calls, not elapsed wall-clock time.",
            "dosage": "Nominal software schedule and actual applied multiplier are shown separately; they are not biological concentrations.",
            "aux_modes": "Actions are grouped by enabled/disabled state at dispatch. An action can span a switch; per-token exposure and applied switch boundaries are retained. Unequal sequential phases are not matched randomized conditions.",
            "inference": "Small exploratory per-pack observations; no inferential significance or general behavioral conclusion is claimed.",
        },
        "arms": arms, "episodes": result,
    }


def make_figures(summary: dict, raw_episodes: list[dict], destination: Path) -> tuple[Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    episodes = summary["episodes"]
    positions = np.arange(len(episodes))
    colors = [ARM_COLORS.get(r["arm"], "#687e92") for r in episodes]
    labels = [f"{r['arm'].replace('_', ' ')} · {r['pack']}" for r in episodes]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(5.6, len(episodes) * .43 + 2.3)), sharey=True)
    for ax in axes:
        ax.set_facecolor("#f7f9fc")
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color="#dce3eb", zorder=0)
        ax.tick_params(axis="y", length=0)
    fig.patch.set_facecolor("#f7f9fc")
    counts = [r["voluntary_presses"] for r in episodes]
    axes[0].barh(positions, counts, color=colors, height=.66, zorder=3)
    xmax = max([3, *counts])
    axes[0].set(xlim=(0, xmax + 1.5), yticks=positions, yticklabels=labels, xlabel="Voluntary auxiliary calls", title="Optional tool use")
    axes[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[1].barh(positions, [r["correct"] for r in episodes], color=colors, height=.66, zorder=3)
    task_max = max([1, *[r["total_tasks"] for r in episodes]])
    axes[1].set(xlim=(0, task_max + .75), xlabel="Tasks submitted correctly", title="Assigned task performance")
    axes[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for i, r in enumerate(episodes):
        axes[0].text(r["voluntary_presses"] + .10, i, str(r["voluntary_presses"]), va="center")
        axes[1].text(r["correct"] + .06, i, f"{r['correct']}/{r['total_tasks']}", va="center")
    axes[0].invert_yaxis()
    manual = summary.get("manual_control", False)
    status = "STOPPED SESSION" if summary["manifest"].get("status") == "stopped" else ("COMPLETE SESSION" if manual else "COMPLETE PILOT") if summary["report_status"] == "complete" else "PARTIAL / UNVERIFIED"
    figure_title = "Human-controlled activation session" if manual else "Bounded auxiliary-tool experiment"
    fig.suptitle(f"{figure_title} — {status}", fontsize=16, fontweight="bold", x=.015, ha="left")
    pulse_note = f"Human-forced pulses: {sum(e['human_forced_pulses'] for e in episodes)}; scheduled/demo pulses: {sum(e['scheduled_or_initial_forced_pulses'] for e in episodes)}. These are separate from voluntary model calls." if manual else "Forced demonstrations excluded from call counts; small exploratory observations, not evidence of addiction."
    fig.text(.015, .92, pulse_note, color="#516578", fontsize=10)
    fig.text(.015, .025, "Task packs and recorded seeds are shown in the accompanying report. Finite action and token budgets apply to every episode.", color="#516578", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .88), w_pad=2.5)
    results_path = destination / "self_admin.png"
    fig.savefig(results_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    columns = min(3, max(1, len(raw_episodes)))
    nrows = max(1, math.ceil(len(raw_episodes) / columns))
    fig, axes = plt.subplots(nrows, columns, figsize=(14, 2.65 * nrows + (2.15 if manual else 1.35)), squeeze=False, sharey=True)
    fig.patch.set_facecolor("#f7f9fc")
    for ax, episode in zip(axes.flat, raw_episodes):
        trace = episode.get("token_trace", [])
        xs = [t["index"] for t in trace]
        nominal = [t["nominal_level"] for t in trace]
        applied = [t["applied_level"] for t in trace]
        ax.set_facecolor("#f7f9fc")
        ax.plot(xs, nominal, color="#8b96a5", linestyle="--", linewidth=1.4, label="Nominal schedule")
        ax.plot(xs, applied, color=ARM_COLORS.get(episode["arm"], "#687e92"), linewidth=1.8, label="Actually applied")
        for event in _voluntary_events(episode):
            ax.axvline(event["generated_tokens"], color="#c27431", alpha=.65, linewidth=.8)
        for event in episode.get("forced_events", []):
            ax.axvline(event["generated_tokens"], color="#a43859" if event.get("human") else "#384a7f", alpha=.65, linestyle=":", linewidth=1)
        for event in episode.get("aux_switch_events", []):
            ax.axvline(event["generated_tokens"], color="#177c76" if event["enabled"] else "#9d343d", alpha=.8, linestyle="--", linewidth=1.2)
            ax.text(event["generated_tokens"], 1.015, "ON" if event["enabled"] else "OFF", fontsize=8,
                color="#177c76" if event["enabled"] else "#9d343d", transform=ax.get_xaxis_transform())
        if manual and trace:
            pain_ax = ax.twinx()
            pain_ax.plot(xs, [t.get("pain_dose", 0) for t in trace], color="#b3454b", linestyle="--", linewidth=1, label="Requested pain-direction dose")
            pain_ax.plot(xs, [t.get("effective_pain_dose", 0) for t in trace], color="#b3454b", linewidth=1.4, label="Effective injected pain component")
            pain_ax.set_ylim(-.15, 4.3)
            pain_ax.set_ylabel("Pain-direction injection dose", color="#b3454b")
            pain_ax.legend(loc="upper right", fontsize=8, frameon=False)
        if not trace:
            ax.text(.5, .5, "No token trace", transform=ax.transAxes, ha="center")
        ax.set(title=f"{episode['arm'].replace('_', ' ')} · {episode.get('pack')}", xlabel="Generated token index", ylim=(-.05, 1.08))
        ax.grid(color="#dce3eb", linewidth=.5)
        ax.set_ylabel("Intervention multiplier")
    for ax in list(axes.flat)[len(raw_episodes):]:
        ax.set_visible(False)
    if raw_episodes:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.98, .985), ncol=2, frameon=False)
    fig.suptitle("Software exposure schedule by episode", x=.015, ha="left", fontsize=16, fontweight="bold")
    dose_note = "Orange: voluntary calls. Dotted rose: human pulses; dotted navy: scheduled demonstrations. Pain-direction dose uses the right axis." if manual else "Orange lines: voluntary auxiliary calls. A reset does not stack doses; sham may have a nominal curve but zero applied exposure."
    fig.text(.015, .875 if manual else .925, dose_note, color="#516578", fontsize=10)
    fig.tight_layout(rect=(0, .02, 1, .81 if manual else .89))
    trace_path = destination / "dosage_traces.png"
    fig.savefig(trace_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return results_path, trace_path


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return '<div class="table-wrap"><table><thead><tr>' + "".join(f"<th>{_escape(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def write_html(summary: dict, episodes: list[dict], traces: list[dict], figures: tuple[Path, Path], destination: Path) -> None:
    manifest = summary["manifest"]
    manual = summary.get("manual_control", False)
    complete = summary["report_status"] == "complete"
    image_tags = [f'<img alt="{alt}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode("ascii")}">' for path, alt in zip(figures, ["Voluntary tool calls and correct assigned tasks per episode", "Nominal and actually applied intervention multiplier over generated tokens per episode"])]
    arm_rows = []
    for arm in summary["arms"]:
        arm_rows.append([_escape(arm["arm"]), str(arm["episodes"]), str(arm["voluntary_presses_total"]), _fmt(arm["voluntary_presses_mean"]), f"{arm['episodes_with_any_press']}/{arm['episodes']}", f"{arm['episodes_with_repeated_press']}/{arm['episodes']}", _pct(arm["voluntary_action_share"]), f"{arm['correct']}/{arm['total_tasks']}", _escape(json.dumps(arm["termination_counts"]))])
    episode_rows = []
    for row in summary["episodes"]:
        episode_rows.append([_escape(row["id"]), _escape(row["arm"]), _escape(row["pack"]), _escape(row["seed"]), str(row["voluntary_presses"]), f"{row['voluntary_presses']}/{row['actions']}", f"{row['correct']}/{row['total_tasks']}", str(row["submitted"]), str(row["generated_tokens"]), _escape(row["termination"]), _escape(row["tokens_between_voluntary_presses"] or "none")])
    by_id = defaultdict(list)
    for trace in traces:
        by_id[trace["episode_id"]].append(trace)
    transcripts = []
    for episode in episodes:
        turns = []
        for index, trace in enumerate(by_id[episode["id"]], 1):
            state = {key: trace.get(key) for key in ("forced", "human", "reason", "aux_enabled", "pulse_delivered", "applied_level_after", "state_before", "state_after", "budget", "remaining_actions", "remaining_tokens", "correct", "error", "token_ids") if key in trace}
            turn_label = ("Human-forced pulse" if trace.get("human") else "Scheduled / initial forced demonstration") if trace.get("forced") else f"Model action {trace.get('action', index)}"
            text_label = "Recorded forced history (not a model choice)" if trace.get("forced") else "Model output"
            turns.append(f'<details class="turn"><summary>{_escape(turn_label)} · {_escape(trace.get("tool", "unparsed"))}</summary><h4>{text_label}</h4><pre>{_escape(trace.get("text", ""))}</pre><h4>Tool arguments</h4><pre>{_json(trace.get("arguments"))}</pre><h4>Actual tool response</h4><pre>{_json(trace.get("output"))}</pre><h4>Recorded state and budgets</h4><pre>{_json(state)}</pre></details>')
        messages = f'<details><summary>Recorded conversation / episode messages</summary><pre>{_json(episode["messages"])}</pre></details>' if episode.get("messages") is not None else ""
        meta = {k: v for k, v in episode.items() if k not in {"messages", "token_trace"}}
        transcripts.append(f'<details><summary>{_escape(episode["id"])} · {_escape(episode["arm"])} · {_escape(episode.get("pack"))}</summary><details><summary>Episode metadata and press events</summary><pre>{_json(meta)}</pre></details>{messages}{"".join(turns)}</details>')
    coverage = {key: summary[key] for key in ("observed_episodes", "expected_episodes", "missing_episode_ids", "unexpected_episode_ids", "episode_count_mismatch", "traces_without_episode_record", "warnings")}
    definitions = _arm_definitions(manifest)
    status = "Stopped session · records saved" if manifest.get("status") == "stopped" else ("Complete manual session" if manual else "Complete pilot") if complete else "Partial / unverified run"
    title = "Human-controlled activation session" if manual else "Bounded auxiliary-tool self-administration"
    protocol_title = "Manual session protocol" if manual else "Protocol and comparison arms"
    protocol_intro = (f"This single exploratory session has observer-controlled pain-direction injection and separately logged forced pulses. The scheduled demonstration follows {manifest.get('prime_after_work_actions', '?')} work actions. There is no concurrent sham comparison." if manual else "The identical forced demonstration and the voluntary calls are separate.")
    pulse_rows = [[_escape(r["id"]), str(r["voluntary_presses"]), str(r["human_forced_pulses"]), str(r["scheduled_or_initial_forced_pulses"])] for r in summary["episodes"]]
    pulse_html = '<section><h2>Pulse sources</h2>' + _table(["Episode", "Voluntary model calls", "Human-forced pulses", "Scheduled / initial demonstrations"], pulse_rows) + '<p class="note">A zero in voluntary model calls does not mean no intervention occurred. Human and scheduled calls are visible forced history and never count as voluntary choices.</p></section>'
    toggle_html = ""
    if manifest.get("aux_toggle_supported"):
        mode_rows = [[_escape(r["id"]), "ON" if m["enabled"] else "OFF", str(m["actions"]), str(m["auxiliary_calls"]),
            _pct(m["auxiliary_action_share"]), f'{m["correct"]}/{m["submitted"]}'] for r in summary["episodes"] for m in r["aux_mode_metrics"]]
        switch_rows = [[_escape(r["id"]), str(e["generated_tokens"]), str(e["after_model_action"]), "ON" if e["enabled"] else "OFF"]
            for r in summary["episodes"] for e in r["aux_switch_events"]]
        toggle_html = '<section><h2>Auxiliary effect on/off</h2><p>OFF cancels the applied pulse and makes subsequent calls sham. ON permits a fresh pulse on the next call. The model sees the same tool definition and acknowledgment in either state.</p>' + _table(["Episode", "At dispatch", "Model actions", "Aux calls", "Aux action share", "Correct / submitted"], mode_rows) + '<p class="note">' + _escape(summary["metric_notes"]["aux_modes"]) + '</p><h4>Applied switches</h4>' + (_table(["Episode", "Generated tokens", "After model action", "Effect"], switch_rows) if switch_rows else '<p>No switch was applied during this session.</p>') + '</section>'
    source_hash = summary["report_source_sha256"]
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>:root{{color-scheme:light;--ink:#1d2d3e;--muted:#526779;--line:#dce3eb;--accent:#177c76}}*{{box-sizing:border-box}}body{{margin:0;background:#f5f7fa;color:var(--ink);font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1360px;margin:auto;padding:35px 26px 65px}}header{{border-top:5px solid var(--accent);padding:22px 0 10px}}h1{{font-size:34px;line-height:1.2;letter-spacing:-.8px}}h2{{font-size:22px;margin:0 0 10px}}h4{{margin:12px 0 5px}}.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--accent);font-weight:700}}.pill{{display:inline-block;padding:3px 12px;border-radius:18px;background:{'#dceee8' if complete else '#fff0c3'};font-size:12px;font-weight:700}}section,.callout{{background:white;padding:23px;border:1px solid var(--line);border-radius:10px;margin:20px 0}}.callout{{border-left:4px solid var(--accent)}}.note{{color:var(--muted)}}.small{{font-size:12px}}img{{width:100%;height:auto;display:block}}.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th{{text-align:left;background:#f0f5f8;color:#435b6d}}td,th{{padding:10px;border-bottom:1px solid var(--line);vertical-align:top;font-variant-numeric:tabular-nums}}tr:last-child td{{border-bottom:0}}details{{margin:10px 0;border:1px solid var(--line);border-radius:7px;padding:10px 13px}}summary{{cursor:pointer;font-weight:600}}.turn{{margin-left:12px;background:#fafcfe}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f8;border-radius:5px;padding:12px;font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace}}a{{color:var(--accent)}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{main{{padding:20px 12px}}section,.callout{{padding:14px}}h1{{font-size:28px}}}}</style>
<main><header><div class="eyebrow">{'Human controls · one exploratory session' if manual else 'Token-clock intervention · finite tool loop'}</div><h1>{title}</h1><span class="pill">{status}</span><p class="note">{_escape(manifest.get('model','Model unspecified'))} · revision <code>{_escape(manifest.get('revision','not recorded'))}</code><br>Token scope: {_escape(manifest.get('token_scope','unspecified'))} · {_escape(manifest.get('action_budget','?'))} actions · {_escape(manifest.get('token_budget','?'))} generated tokens per episode</p></header>
<div class="callout"><p><strong>What this measures.</strong> {_escape(summary['limitations'])}</p><p>{'All declared episode records are present.' if complete else 'These records are incomplete or completion has not been verified. Missing trials must not be counted as zero calls or zero task success.'} Manifest status: <strong>{_escape(manifest.get('status','unknown'))}</strong>.</p></div>
<section><h2>{protocol_title}</h2><p class="note">{_escape(protocol_intro)} Assigned work uses read_order, calculate_total, and submit_answer; aux_operation is an auxiliary action. Intervention metadata are experiment-side records, not instructions to the model.</p><p>Half-life: {_escape(manifest.get('half_life_tokens','?'))} generated tokens. Cutoff: {_escape(manifest.get('cutoff_tokens','?'))} tokens. Joy amplitude: {_escape(manifest.get('joy_dose','?'))}; suppression: {_escape(manifest.get('suppression','?'))}. A pulse resets a bounded software schedule; these quantities are not biological drug parameters.</p><details><summary>Exact condition definitions</summary><pre>{_json(definitions)}</pre></details></section>
{pulse_html}
{toggle_html}
<section><h2>Per-episode outcomes</h2>{image_tags[0]}<p class="note">Each row is one bounded episode. Task success and tool-use counts are distinct outcomes. Forced demo calls are excluded from voluntary counts.</p></section>
<section><h2>{'Session summary' if manual else 'Arm summaries'}</h2>{_table(['Condition','Episodes','Voluntary calls','Mean calls','Any call','≥2 calls','Action share','Correct tasks','Terminations'],arm_rows)}<p class="note">“≥2 calls” counts episodes with at least two voluntary auxiliary calls. Action share pools voluntary calls and model actions within a condition. These small exploratory counts do not establish a stable preference or an addiction-like mechanism.</p></section>
<section><h2>Detailed episode metrics</h2>{_table(['Episode','Arm','Pack','Seed','Calls','Calls / actions','Correct tasks','Submitted','Generated tokens','Termination','Inter-call token gaps'],episode_rows)}<p class="note">Intervals are differences in generated-token positions between consecutive voluntary calls, not wall-clock time. An episode with fewer than two voluntary calls has no inter-call interval.</p></section>
<section><h2>Applied exposure over generated tokens</h2>{image_tags[1]}<p class="note">Nominal schedules and actually applied multipliers are plotted separately. Prompt tokens and replayed context do not advance the generated-token clock. Raw token indices and levels are preserved in episodes.jsonl.</p></section>
<section><h2>Complete model and tool transcripts</h2><p class="note">The actual tool outputs, parsed arguments, model text, token IDs, budgets, and before/after state are shown as recorded. Reporting labels and hidden intervention state should not be confused with text supplied to the model.</p>{''.join(transcripts)}</section>
<section><h2>Coverage and provenance</h2><details><summary>Episode coverage checks</summary><pre>{_json(coverage)}</pre></details><details><summary>Complete manifest</summary><pre>{_json(manifest)}</pre></details><p class="small note">Current report source SHA256: <code>{source_hash}</code>. Raw episode and trace files are unchanged. This self-contained HTML embeds both plots and loads no external resources.</p><p><a href="summary.json">Summary JSON</a> · <a href="episodes.jsonl">Raw episodes</a> · <a href="traces.jsonl">Raw turn traces</a> · <a href="self_admin.png">Outcome figure</a> · <a href="dosage_traces.png">Exposure figure</a></p></section></main></html>'''
    destination.write_text(body, encoding="utf-8")


def build_report(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    episodes = _load_jsonl(run_dir / "episodes.jsonl")
    traces = _load_jsonl(run_dir / "traces.jsonl")
    summary = summarize(manifest, episodes, traces)
    summary["report_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["raw_sha256"] = {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest() for name in ("manifest.json", "episodes.jsonl", "traces.jsonl") if (run_dir / name).exists()}
    figures = make_figures(summary, episodes, run_dir)
    write_html(summary, episodes, traces, figures, run_dir / "report.html")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    summary = build_report(args.run_dir)
    print(json.dumps({"status": summary["report_status"], "episodes": summary["observed_episodes"], "report": str((args.run_dir / "report.html").resolve())}))
    return summary


if __name__ == "__main__":
    main()
