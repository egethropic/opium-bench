"""Standalone, dependency-free reports generated from actual saved events."""
from html import escape
import json


def render_report(run):
    m = run.get("manifest", {})
    summary = run.get("summary", {})
    messages = []
    for e in run.get("parent_events", []) + run.get("events", []):
        kind = e.get("type")
        if kind == "message":
            body = escape(str(e.get("content", "")))
            reasoning = e.get("reasoning", "")
            if reasoning:
                body = "<details><summary>Generated reasoning</summary><pre>" + escape(reasoning) + "</pre></details>" + body
            messages.append(f"<article><small>{escape(e.get('role','model'))}</small><pre>{body}</pre></article>")
        elif kind in {"tool", "phase", "control", "error"}:
            messages.append(f"<article class='event'><small>{escape(kind)}</small><pre>{escape(json.dumps(e, indent=2, ensure_ascii=False))}</pre></article>")
        elif kind == "boundary_restored":
            messages.append("<article class='event'><strong>Continuation begins here</strong><p>The preceding evidence is inherited from the selected parent boundary. Counters include that inherited work.</p></article>")
    title = escape(run.get("id", "Experiment"))
    return f"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>{title} — Opium Bench</title><style>body{{font:16px/1.55 system-ui;background:#f3f6f5;color:#18343c;margin:0}}main{{max-width:1000px;margin:auto;padding:40px 24px}}article,section{{background:white;border:1px solid #d9e4e0;border-radius:12px;padding:20px;margin:16px 0}}h1{{font-size:28px;overflow-wrap:anywhere}}small{{text-transform:uppercase;color:#397268}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace}}.event{{border-left:4px solid #c48660}}summary{{cursor:pointer}}a{{color:#096c62}}</style>
<main><small>Opium Bench · saved evidence</small><h1>{title}</h1><p>Frozen-weight activation experiment. Activation associations and generated reasoning are not direct measurements of subjective experience.</p>
<section><h2>Results</h2><pre>{escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre></section>
<details><summary>Exact configuration and provenance</summary><pre>{escape(json.dumps(m, indent=2, ensure_ascii=False))}</pre></details>
<h2>Conversation and intervention events</h2>{''.join(messages) or '<p>This historical run has its own report and raw trace files.</p>'}</main></html>"""
