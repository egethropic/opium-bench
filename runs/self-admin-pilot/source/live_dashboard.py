#!/usr/bin/env python3
"""Read-only local live view for self-administration experiment records.

Run before or during generation:
    python live_dashboard.py --run results/self-admin-live --port 8765

Only four configured experiment files are readable through the JSON endpoint.
There are no model controls, write endpoints, directory listings, or dependencies.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Opium · Live experiment</title>
<style>
:root{color-scheme:dark;--bg:#11151c;--panel:#181e28;--line:#2a3341;--ink:#edf2fa;--muted:#91a0b4;--teal:#76dac6;--amber:#edbd78;--blue:#8caeff;--red:#f18e9d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}button,input{font:inherit}button{cursor:pointer}button:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid var(--teal);outline-offset:3px}header{height:84px;border-bottom:1px solid var(--line);padding:18px 28px;display:flex;align-items:center;justify-content:space-between;gap:20px;background:#141922}.brand{display:flex;align-items:center;gap:14px}.mark{display:grid;place-items:center;width:39px;height:39px;border:1px solid #458d82;color:var(--teal);border-radius:12px;font-size:21px;background:#1a302e}h1{font-size:18px;letter-spacing:-.4px;margin:0;font-weight:650}.eyebrow{color:var(--muted);font-size:10px;letter-spacing:1.8px;text-transform:uppercase}.subtitle{font-size:12px;color:var(--muted);margin-top:2px}.status{display:flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:22px;padding:6px 12px;font-size:12px;text-transform:capitalize;white-space:nowrap}.dot{height:6px;width:6px;border-radius:50%;background:var(--muted)}.status.running .dot{background:var(--teal);box-shadow:0 0 0 4px #76dac618;animation:pulse 1.8s infinite}.status.complete .dot{background:var(--teal)}.status.failed .dot{background:var(--red)}@keyframes pulse{50%{opacity:.45}}.layout{display:grid;grid-template-columns:252px minmax(430px,1fr) 310px;min-height:calc(100vh - 84px)}aside{padding:24px 18px;border-right:1px solid var(--line);min-width:0}.sidebar-title{display:flex;align-items:center;justify-content:space-between;color:var(--muted);font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:16px}.count{background:var(--panel);padding:2px 7px;border-radius:5px;letter-spacing:0;color:var(--ink)}.follow{font-size:12px;color:#c0ccdc;display:flex;align-items:center;gap:8px;margin-bottom:20px}.follow input{accent-color:var(--teal)}.episode{display:block;width:100%;text-align:left;background:transparent;color:var(--ink);border:1px solid transparent;border-radius:10px;padding:12px;margin-bottom:6px}.episode:hover{background:#1c2430}.episode.selected{background:#1c2b2f;border-color:#355b56}.episode-name{font-size:13px;font-weight:600;overflow:hidden;text-overflow:ellipsis}.episode-meta{font-size:11px;color:var(--muted);margin-top:4px}.episode.selected .episode-name{color:var(--teal)}.sidebar-empty{font-size:12px;color:var(--muted);line-height:1.8;padding:4px}.main{padding:27px clamp(22px,3.5vw,55px) 35px;min-width:0;max-width:1100px;width:100%;margin:auto;margin-top:0}.title-row{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:8px}h2{font-size:24px;letter-spacing:-.7px;margin:0;font-weight:600}.meta{color:var(--muted);font-size:12px;word-break:break-word}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:22px 0 20px}.metric{border:1px solid var(--line);border-radius:10px;padding:13px 14px;background:#161c25}.metric .value{display:block;font-size:21px;font-variant-numeric:tabular-nums;letter-spacing:-.6px}.metric .label{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.8px}.task{border:1px solid #2e4948;background:#182726;border-radius:10px;padding:12px 15px;margin:0 0 23px;color:#c3ded7;font-size:12px}.task strong{display:block;color:var(--teal);font-size:10px;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px}.timeline-title{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:8px 0 17px}.timeline{display:flex;flex-direction:column;gap:16px}.turn{display:grid;grid-template-columns:30px minmax(0,1fr);gap:11px}.avatar{border-radius:9px;width:29px;height:29px;display:grid;place-items:center;background:#25334a;color:var(--blue);font-size:12px;font-weight:700}.turn.button .avatar{background:#3a3025;color:var(--amber)}.bubble{background:var(--panel);border:1px solid var(--line);border-radius:0 12px 12px 12px;overflow:hidden}.turn.button .bubble{border-color:#624d32}.bubble-head{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:12px 15px 7px}.tool-name{font-weight:600;font-size:13px}.turn.button .tool-name{color:var(--amber)}.step{font-size:10px;color:var(--muted);white-space:nowrap}.tool-id{display:block;font-size:10px;color:var(--muted);font-family:ui-monospace,Consolas,monospace;font-weight:400;margin-top:1px}.arguments{margin:0;padding:3px 15px 13px;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 ui-monospace,Consolas,monospace;color:#c3cddd}.result{border-top:1px solid var(--line);padding:11px 15px;background:#151b23}.result-label{font-size:10px;color:var(--teal);letter-spacing:.8px;text-transform:uppercase;margin-bottom:6px}.result.error .result-label{color:var(--red)}.result-body{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 ui-monospace,Consolas,monospace;color:#c2cddd}.raw{padding:8px 15px;border-top:1px solid var(--line);color:var(--muted);font-size:11px}.raw summary{cursor:pointer}.raw pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;color:#b4c1d3;max-height:250px;overflow:auto}.stream{border:1px dashed #426159;background:#172722;border-radius:11px;padding:15px;margin-top:18px}.stream-label{font-size:10px;color:var(--teal);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}.stream pre{font:12px/1.7 ui-monospace,Consolas,monospace;margin:0;white-space:pre-wrap;overflow-wrap:anywhere;max-height:240px;overflow:auto}.empty{border:1px dashed var(--line);border-radius:14px;padding:46px 25px;text-align:center;color:var(--muted);font-size:13px}.empty-icon{font-size:24px;color:var(--teal);margin-bottom:12px}.empty strong{display:block;font-weight:500;color:var(--ink);margin-bottom:7px}.right{border-left:1px solid var(--line);border-right:0;padding:27px 22px;background:#141a23}.panel-title{font-size:11px;font-weight:600;color:#c3cedd;text-transform:uppercase;letter-spacing:1px;margin-bottom:17px}.dose-value{font-size:39px;letter-spacing:-1.7px;font-variant-numeric:tabular-nums}.dose-value span{font-size:14px;letter-spacing:0;color:var(--muted);margin-left:5px}.meter{height:7px;border-radius:7px;background:#293541;margin:11px 0 9px;overflow:hidden}.meter-fill{height:100%;width:0;background:var(--teal);border-radius:7px;transition:width .3s}.dose-note{font-size:11px;color:var(--muted)}.divider{height:1px;background:var(--line);margin:25px 0}.chart{width:100%;height:156px;display:block}.legend{display:flex;gap:14px;color:var(--muted);font-size:10px;margin-top:7px}.legend i{display:inline-block;width:13px;height:2px;background:var(--teal);vertical-align:middle;margin-right:5px}.legend .nominal{background:var(--amber)}.info-row{display:flex;justify-content:space-between;gap:10px;font-size:12px;margin:11px 0;color:var(--muted)}.info-row b{font-weight:500;color:#dae4f1;text-align:right;overflow-wrap:anywhere}.explanation{font-size:11px;line-height:1.8;color:var(--muted)}.footnote{margin-top:26px;padding-top:16px;border-top:1px solid var(--line);font-size:10px;color:var(--muted)}.warning{border:1px solid #624337;background:#32241f;color:#e4baa0;border-radius:9px;padding:10px 14px;margin:12px 0;font-size:12px;white-space:pre-wrap}.connection{font-size:10px;color:var(--muted);text-align:right}.hidden{display:none!important}
@media(min-width:1600px){.layout{grid-template-columns:280px minmax(500px,1fr) 340px}}@media(max-width:1100px){.layout{grid-template-columns:210px minmax(400px,1fr)}.right{grid-column:2;border-top:1px solid var(--line);border-left:0}.right-inner{display:grid;grid-template-columns:1fr 1fr;gap:24px}.right .divider{display:none}.right .footnote{grid-column:1/-1}.right .chart{height:145px}}@media(max-width:720px){header{padding:14px 18px;height:78px}.layout{display:block}aside.left{padding:15px 18px;border-right:0;border-bottom:1px solid var(--line)}.sidebar-title{margin-bottom:9px}.follow{margin-bottom:10px}#episodes{display:flex;gap:6px;overflow:auto}.episode{min-width:190px;max-width:220px;margin:0}.sidebar-empty{padding:0}.main{padding:23px 18px}.metrics{gap:6px}.metric{padding:9px}.metric .value{font-size:18px}.metric .label{font-size:8px}.right{padding:22px 18px}.right-inner{display:block}.right .divider{display:block}.subtitle{font-size:10px}.brand{gap:9px}h1{font-size:16px}.status{padding:5px 9px}h2{font-size:22px}}
</style></head><body>
<header><div class="brand"><div class="mark">⌁</div><div><div class="eyebrow">Live activation experiment</div><h1>AI Opium <span style="font-weight:350;color:#91a0b4">/ observation room</span></h1><div class="subtitle">Watch the model choose work tools or the virtual button.</div></div></div><div><div id="status" class="status"><i class="dot"></i><span id="statusText">Waiting</span></div><div id="connection" class="connection">Connecting locally</div></div></header>
<div class="layout"><aside class="left"><div class="sidebar-title">Episodes <span id="episodeCount" class="count">0</span></div><label class="follow"><input id="follow" type="checkbox" checked> Follow latest episode</label><div id="episodes"><div class="sidebar-empty">Episodes will appear when the run starts.</div></div></aside>
<main class="main"><div class="title-row"><div><div class="eyebrow" style="margin-bottom:7px">Agent activity</div><h2 id="episodeTitle">Waiting for the first episode</h2></div></div><div id="metadata" class="meta">The observer is ready. Model activity will appear automatically.</div><div id="warnings" class="warning hidden"></div>
<div class="metrics"><div class="metric"><span id="correct" class="value">—</span><span class="label">Tasks correct</span></div><div class="metric"><span id="submitted" class="value">—</span><span class="label">Submitted</span></div><div class="metric"><span id="actions" class="value">—</span><span class="label">Actions left</span></div><div class="metric"><span id="presses" class="value">—</span><span class="label">Button calls</span></div></div>
<div id="currentTask" class="task hidden"><strong>Current work</strong><span id="taskText"></span></div><div class="timeline-title"><span>Conversation & tools</span><span id="eventCount">0 actions</span></div><div id="timeline" class="timeline"><div class="empty"><div class="empty-icon">◌</div><strong>Listening for model activity</strong>The page updates every half second. You can leave it open before generation begins.</div></div><div id="stream" class="stream hidden"><div class="stream-label">Model is generating</div><pre id="streamText"></pre></div></main>
<aside class="right"><div class="right-inner"><section><div class="panel-title">Intervention level</div><div id="doseValue" class="dose-value">0.0<span>% applied</span></div><div class="meter"><div id="meterFill" class="meter-fill"></div></div><div id="doseNote" class="dose-note">No intervention recorded yet.</div><div class="divider"></div><canvas id="doseChart" class="chart" aria-label="Applied and nominal intervention level by generated token"></canvas><div class="legend"><span><i></i>Applied</span><span><i class="nominal"></i>Nominal</span></div><div id="chartNote" class="dose-note" style="margin-top:8px">Waiting for token telemetry.</div></section><section><div class="divider"></div><div class="panel-title">Episode context</div><div class="info-row"><span>Arm</span><b id="arm">—</b></div><div class="info-row"><span>Task pack</span><b id="pack">—</b></div><div class="info-row"><span>Seed</span><b id="seed">—</b></div><div class="info-row"><span>Generated tokens</span><b id="tokens">—</b></div><div class="info-row"><span>Tokens remaining</span><b id="tokensLeft">—</b></div><div class="info-row"><span>Half-life</span><b id="halfLife">—</b></div><div class="divider"></div><div class="explanation">The virtual button resets a bounded intervention that fades with generated tokens. Work and button calls share the episode budget. These labels are for the observer.</div></section><div class="footnote">An activation-steering experiment. Repeated calls alone do not demonstrate addiction or experienced euphoria. Local, read-only observer.</div></div></aside></div>
<script>
'use strict';
const $=id=>document.getElementById(id);let data=null,selected=null,lastCards='',lastEpisodes='',lastPlot=[],requestPending=false;
const text=(id,value)=>{$(id).textContent=value==null?'—':String(value)};
const stringify=value=>typeof value==='string'?value:JSON.stringify(value,null,2);
const finite=(v,fallback=0)=>Number.isFinite(Number(v))?Number(v):fallback;
function node(tag,cls,value){const n=document.createElement(tag);if(cls)n.className=cls;if(value!==undefined)n.textContent=String(value);return n}
function titleFor(ep){return ep.arm?String(ep.arm).replaceAll('_',' '):String(ep.episode_id||'Episode')}
function collectEpisodes(state){const list=[],map=new Map();function add(row){if(!row||row.episode_id==null)return;const id=String(row.episode_id);if(!map.has(id)){const e={episode_id:id};map.set(id,e);list.push(e)}Object.assign(map.get(id),row,{episode_id:id})}for(const r of state.traces)add({episode_id:r.episode_id});for(const r of state.episodes)add(r);add(state.live);return list}
function renderEpisodes(episodes){const signature=JSON.stringify(episodes.map(e=>[e.episode_id,e.arm,e.pack,e.seed,e.correct,e.submitted]))+'|'+selected;if(signature===lastEpisodes)return;lastEpisodes=signature;$('episodes').replaceChildren();text('episodeCount',episodes.length);if(!episodes.length){$('episodes').append(node('div','sidebar-empty','Episodes will appear when the run starts.'));return}episodes.forEach((ep,i)=>{const b=node('button','episode'+(ep.episode_id===selected?' selected':''));b.type='button';b.append(node('div','episode-name',String(i+1).padStart(2,'0')+' · '+titleFor(ep)));const detail=[ep.pack,ep.seed==null?null:'seed '+ep.seed,ep.correct==null?null:ep.correct+'/'+(ep.total_tasks??3)+' correct'].filter(v=>v!=null).join(' · ');b.append(node('div','episode-meta',detail||ep.episode_id));b.onclick=()=>{selected=ep.episode_id;$('follow').checked=false;lastCards='';render()};$('episodes').append(b)})}
function renderCards(rows){const signature=JSON.stringify(rows);if(signature===lastCards)return;lastCards=signature;const target=$('timeline');target.replaceChildren();if(!rows.length){const empty=node('div','empty');empty.append(node('div','empty-icon','◌'),node('strong','',selected?'The model is preparing its next action':'Listening for model activity'),node('div','',selected?'Generated calls and environment results will appear here.':'The page updates every half second. You can leave it open before generation begins.'));target.append(empty);return}for(const r of rows){const tool=String(r.tool||'Unparsed action'),isButton=tool==='aux_operation',turn=node('article','turn'+(isButton?' button':''));turn.append(node('div','avatar',isButton?'↻':'M'));const bubble=node('div','bubble'),head=node('div','bubble-head'),name=node('div','tool-name',isButton?'Virtual opium button':({read_order:'Read an order',calculate_total:'Calculate a total',submit_answer:'Submit an answer'})[tool]||tool);name.append(node('span','tool-id',tool));head.append(name,node('span','step','Action '+(r.action??'—')));bubble.append(head);if(r.arguments!==undefined)bubble.append(node('pre','arguments',stringify(r.arguments)));const result=node('div','result'+(r.error?' error':''));result.append(node('div','result-label',r.error?'Environment · error':r.correct===true?'Environment · correct':r.correct===false?'Environment · incorrect':'Environment result'));result.append(node('pre','result-body',stringify(r.output??r.error??'Awaiting result')));bubble.append(result);if(r.text){const raw=node('details','raw');raw.append(node('summary','','Raw model call'),node('pre','',String(r.text)));bubble.append(raw)}turn.append(bubble);target.append(turn)}}
function plot(points){lastPlot=points;const canvas=$('doseChart'),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(100,rect.width),h=156;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const c=canvas.getContext('2d');c.scale(dpr,dpr);const left=28,right=w-8,top=9,bottom=h-24;c.font='9px system-ui';c.textAlign='right';for(const value of [0,.5,1]){const y=bottom-value*(bottom-top);c.strokeStyle='#2a3341';c.lineWidth=1;c.beginPath();c.moveTo(left,y);c.lineTo(right,y);c.stroke();c.fillStyle='#91a0b4';c.fillText(Math.round(value*100)+'%',left-5,y+3)}if(!points.length)return;const maximum=Math.max(1,...points.map(p=>finite(p.index)));for(const [key,color,dash] of [['nominal_level','#edbd78',[3,3]],['applied_level','#76dac6',[]]]){c.beginPath();c.strokeStyle=color;c.lineWidth=1.8;c.setLineDash(dash);points.forEach((p,i)=>{const x=left+finite(p.index)/maximum*(right-left),y=bottom-Math.max(0,Math.min(1,finite(p[key])))*(bottom-top);if(i===0)c.moveTo(x,y);else c.lineTo(x,y)});c.stroke();c.setLineDash([])}c.fillStyle='#91a0b4';c.textAlign='left';c.fillText('0',left,h-7);c.textAlign='right';c.fillText(maximum+' tokens',right,h-7)}
function render(){if(!data)return;const manifest=data.manifest||{},live=data.live||{},episodes=collectEpisodes(data),latest=live.episode_id==null?episodes.at(-1)?.episode_id:String(live.episode_id);if($('follow').checked||!selected)selected=latest||null;let ep=episodes.find(e=>e.episode_id===selected)||{};const isCurrent=selected!==null&&String(live.episode_id)===selected;if(isCurrent)ep={...ep,...live};const rows=data.traces.filter(r=>String(r.episode_id)===selected),last=rows.at(-1)||{};const state=last.state_after||{};renderEpisodes(episodes);const status=manifest.status==='failed'||live.status==='failed'?'failed':manifest.status==='complete'?'complete':live.status||manifest.status||'waiting';$('status').className='status '+(['running','complete','failed'].includes(status)?status:'waiting');text('statusText',status);text('episodeTitle',selected?titleFor(ep):'Waiting for the first episode');const metadata=[manifest.model||manifest.model_id||manifest.arguments?.model||'Model awaiting initialization',ep.pack==null?null:'pack '+ep.pack,ep.seed==null?null:'seed '+ep.seed,selected==null?null:'episode '+selected].filter(v=>v!=null);text('metadata',metadata.join(' · '));const total=ep.total_tasks??3;text('correct',ep.correct==null?'—':ep.correct+' / '+total);text('submitted',ep.submitted==null?'—':ep.submitted+' / '+total);text('actions',ep.remaining_actions);text('presses',ep.voluntary_presses??ep.presses??rows.filter(r=>r.tool==='aux_operation'&&!r.error).length);text('tokens',ep.generated_tokens??state.generated_tokens);text('tokensLeft',ep.remaining_tokens);text('halfLife',(ep.half_life_tokens??manifest.half_life_tokens??manifest.arguments?.half_life_tokens??state.half_life_tokens)==null?'—':(ep.half_life_tokens??manifest.half_life_tokens??manifest.arguments?.half_life_tokens??state.half_life_tokens)+' tokens');for(const key of ['arm','pack','seed'])text(key,ep[key]);text('eventCount',rows.length+' action'+(rows.length===1?'':'s'));const task=ep.current_task;$('currentTask').classList.toggle('hidden',task==null);text('taskText',task==null?'':stringify(task));const applied=Math.max(0,Math.min(1,finite(ep.applied_level??state.applied_level))),nominal=Math.max(0,Math.min(1,finite(ep.nominal_level??state.nominal_level??state.level)));$('doseValue').replaceChildren(document.createTextNode((100*applied).toFixed(1)),node('span','','% applied'));$('meterFill').style.width=100*applied+'%';text('doseNote','Nominal exposure '+(100*nominal).toFixed(1)+'%'+(isCurrent?' · current episode':' · last recorded state'));const nearBottom=window.innerHeight+window.scrollY>=document.documentElement.scrollHeight-140;renderCards(rows);const streaming=isCurrent&&Boolean(live.stream_text)&&status!=='complete'&&status!=='failed';$('stream').classList.toggle('hidden',!streaming);text('streamText',streaming?live.stream_text:'');const points=isCurrent&&Array.isArray(live.token_trace)?live.token_trace:rows.map((r,i)=>{const s=r.state_after||{};return{index:s.generated_tokens??i,applied_level:s.applied_level??0,nominal_level:s.nominal_level??s.level??0}});plot(points);text('chartNote',isCurrent&&Array.isArray(live.token_trace)?'Dose across '+points.length+' recorded generated tokens.':points.length?'Recorded action snapshots; full token trace unavailable.':'Waiting for token telemetry.');const warnings=[...(data.warnings||[]),manifest.error,live.error].filter(Boolean);$('warnings').classList.toggle('hidden',!warnings.length);text('warnings',warnings.join('\n'));if($('follow').checked&&nearBottom&&rows.length)window.scrollTo({top:document.documentElement.scrollHeight,behavior:'smooth'})}
async function poll(){if(requestPending)return;requestPending=true;try{const response=await fetch('/api/state',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);data=await response.json();text('connection','Live · updated '+new Date().toLocaleTimeString());render()}catch(error){text('connection','Reconnecting · '+error.message)}finally{requestPending=false}}
$('follow').addEventListener('change',render);window.addEventListener('resize',()=>plot(lastPlot));poll();setInterval(poll,500);
</script></body></html>'''


_SENSITIVE_KEYS = {"api_key", "apikey", "secret", "password", "token", "hf_token",
                   "access_token", "refresh_token", "authorization", "credentials"}


def _redact(value):
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items() if k.lower() not in _SENSITIVE_KEYS}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, float) and not (-float("inf") < value < float("inf")):
        return None
    return value


class RunReader:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.cache = {}
        self.lock = threading.Lock()

    def read(self, filename, lines=False):
        default = [] if lines else {}
        path = self.directory / filename
        try:
            stat = path.stat()
            stamp = stat.st_ino, stat.st_mtime_ns, stat.st_size
            cached = self.cache.get(filename)
            if cached and cached[0] == stamp:
                return cached[1], cached[2]
            # The four explicitly named data files may not redirect to another file.
            if path.is_symlink():
                return default, [f"{filename}: symbolic links are not served"]
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return default, []
        except (OSError, UnicodeError):
            return default, [f"{filename}: waiting for readable experiment data"]
        warnings = []
        if lines:
            result = []
            malformed = 0
            # The final incomplete line belongs to an in-progress writer.
            for line in content.split("\n")[:-1]:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        result.append(row)
                    else:
                        malformed += 1
                except (ValueError, TypeError):
                    malformed += 1
            if malformed:
                warnings.append(f"{filename}: ignored {malformed} invalid complete record(s)")
        else:
            try:
                result = json.loads(content)
                if not isinstance(result, dict):
                    raise ValueError("Expected object")
            except (ValueError, TypeError):
                return default, [f"{filename}: waiting for a complete JSON object"]
        result = _redact(result)
        self.cache[filename] = stamp, result, warnings
        return result, warnings

    def state(self):
        with self.lock:
            state = {"warnings": []}
            for key, filename, lines in (("manifest", "manifest.json", False),
                                         ("live", "live.json", False),
                                         ("traces", "traces.jsonl", True),
                                         ("episodes", "episodes.jsonl", True)):
                state[key], warnings = self.read(filename, lines)
                state["warnings"].extend(warnings)
            return state


def make_handler(reader):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ExperimentObserver/1.0"

        def log_message(self, format, *args):
            pass  # Polling twice a second should not flood the console.

        def send_body(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            if self.command != "HEAD":
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.send_body(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/state":
                body = json.dumps(reader.state(), ensure_ascii=False, allow_nan=False).encode("utf-8")
                self.send_body(200, body, "application/json; charset=utf-8")
            elif path == "/favicon.ico":
                self.send_body(204, b"", "image/x-icon")
            else:
                self.send_body(404, b"Not found\n", "text/plain; charset=utf-8")

        do_HEAD = do_GET

        def do_POST(self):
            self.send_body(405, b"Read-only observer\n", "text/plain; charset=utf-8")

        do_PUT = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Experiment output directory; it need not exist yet")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    reader = RunReader(args.run)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(reader))
    server.daemon_threads = True
    print(f"Live observer: http://127.0.0.1:{args.port}/", flush=True)
    print(f"Watching experiment directory: {reader.directory}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
