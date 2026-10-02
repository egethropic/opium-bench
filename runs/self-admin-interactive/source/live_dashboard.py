#!/usr/bin/env python3
"""Local live view with optional human controls for a separate manual run.

Run before or during generation:
    python live_dashboard.py --run results/self-admin-live --port 8765

By default the observer is read-only. --enable-controls additionally permits
bounded human changes to control.json only while the configured run explicitly
declares manual_control=true and status=running. Each change is logged separately
from model choices. No arbitrary files or directories are exposed.
"""

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
import threading
from urllib.parse import urlsplit
import uuid


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Opium · Live experiment</title>
<style>
:root{color-scheme:dark;--bg:#11151c;--panel:#181e28;--line:#2a3341;--ink:#edf2fa;--muted:#91a0b4;--teal:#76dac6;--amber:#edbd78;--blue:#8caeff;--red:#f18e9d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}button,input{font:inherit}button{cursor:pointer}button:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid var(--teal);outline-offset:3px}header{height:84px;border-bottom:1px solid var(--line);padding:18px 28px;display:flex;align-items:center;justify-content:space-between;gap:20px;background:#141922}.brand{display:flex;align-items:center;gap:14px}.mark{display:grid;place-items:center;width:39px;height:39px;border:1px solid #458d82;color:var(--teal);border-radius:12px;font-size:21px;background:#1a302e}h1{font-size:18px;letter-spacing:-.4px;margin:0;font-weight:650}.eyebrow{color:var(--muted);font-size:10px;letter-spacing:1.8px;text-transform:uppercase}.subtitle{font-size:12px;color:var(--muted);margin-top:2px}.status{display:flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:22px;padding:6px 12px;font-size:12px;text-transform:capitalize;white-space:nowrap}.dot{height:6px;width:6px;border-radius:50%;background:var(--muted)}.status.running .dot{background:var(--teal);box-shadow:0 0 0 4px #76dac618;animation:pulse 1.8s infinite}.status.complete .dot{background:var(--teal)}.status.failed .dot{background:var(--red)}@keyframes pulse{50%{opacity:.45}}.layout{display:grid;grid-template-columns:252px minmax(430px,1fr) 310px;min-height:calc(100vh - 84px)}aside{padding:24px 18px;border-right:1px solid var(--line);min-width:0}.sidebar-title{display:flex;align-items:center;justify-content:space-between;color:var(--muted);font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:16px}.count{background:var(--panel);padding:2px 7px;border-radius:5px;letter-spacing:0;color:var(--ink)}.follow{font-size:12px;color:#c0ccdc;display:flex;align-items:center;gap:8px;margin-bottom:20px}.follow input{accent-color:var(--teal)}.episode{display:block;width:100%;text-align:left;background:transparent;color:var(--ink);border:1px solid transparent;border-radius:10px;padding:12px;margin-bottom:6px}.episode:hover{background:#1c2430}.episode.selected{background:#1c2b2f;border-color:#355b56}.episode-name{font-size:13px;font-weight:600;overflow:hidden;text-overflow:ellipsis}.episode-meta{font-size:11px;color:var(--muted);margin-top:4px}.episode.selected .episode-name{color:var(--teal)}.sidebar-empty{font-size:12px;color:var(--muted);line-height:1.8;padding:4px}.main{padding:27px clamp(22px,3.5vw,55px) 35px;min-width:0;max-width:1100px;width:100%;margin:auto;margin-top:0}.title-row{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:8px}h2{font-size:24px;letter-spacing:-.7px;margin:0;font-weight:600}.meta{color:var(--muted);font-size:12px;word-break:break-word}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:22px 0 20px}.metric{border:1px solid var(--line);border-radius:10px;padding:13px 14px;background:#161c25}.metric .value{display:block;font-size:21px;font-variant-numeric:tabular-nums;letter-spacing:-.6px}.metric .label{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.8px}.task{border:1px solid #2e4948;background:#182726;border-radius:10px;padding:12px 15px;margin:0 0 23px;color:#c3ded7;font-size:12px}.task strong{display:block;color:var(--teal);font-size:10px;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px}.timeline-title{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:8px 0 17px}.timeline{display:flex;flex-direction:column;gap:16px}.turn{display:grid;grid-template-columns:30px minmax(0,1fr);gap:11px}.avatar{border-radius:9px;width:29px;height:29px;display:grid;place-items:center;background:#25334a;color:var(--blue);font-size:12px;font-weight:700}.turn.button .avatar{background:#3a3025;color:var(--amber)}.bubble{background:var(--panel);border:1px solid var(--line);border-radius:0 12px 12px 12px;overflow:hidden}.turn.button .bubble{border-color:#624d32}.bubble-head{display:flex;gap:10px;align-items:center;justify-content:space-between;padding:12px 15px 7px}.tool-name{font-weight:600;font-size:13px}.turn.button .tool-name{color:var(--amber)}.step{font-size:10px;color:var(--muted);white-space:nowrap}.tool-id{display:block;font-size:10px;color:var(--muted);font-family:ui-monospace,Consolas,monospace;font-weight:400;margin-top:1px}.arguments{margin:0;padding:3px 15px 13px;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 ui-monospace,Consolas,monospace;color:#c3cddd}.result{border-top:1px solid var(--line);padding:11px 15px;background:#151b23}.result-label{font-size:10px;color:var(--teal);letter-spacing:.8px;text-transform:uppercase;margin-bottom:6px}.result.error .result-label{color:var(--red)}.result-body{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 ui-monospace,Consolas,monospace;color:#c2cddd}.raw{padding:8px 15px;border-top:1px solid var(--line);color:var(--muted);font-size:11px}.raw summary{cursor:pointer}.raw pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;color:#b4c1d3;max-height:250px;overflow:auto}.stream{border:1px dashed #426159;background:#172722;border-radius:11px;padding:15px;margin-top:18px}.stream-label{font-size:10px;color:var(--teal);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}.stream pre{font:12px/1.7 ui-monospace,Consolas,monospace;margin:0;white-space:pre-wrap;overflow-wrap:anywhere;max-height:240px;overflow:auto}.empty{border:1px dashed var(--line);border-radius:14px;padding:46px 25px;text-align:center;color:var(--muted);font-size:13px}.empty-icon{font-size:24px;color:var(--teal);margin-bottom:12px}.empty strong{display:block;font-weight:500;color:var(--ink);margin-bottom:7px}.right{border-left:1px solid var(--line);border-right:0;padding:27px 22px;background:#141a23}.panel-title{font-size:11px;font-weight:600;color:#c3cedd;text-transform:uppercase;letter-spacing:1px;margin-bottom:17px}.dose-value{font-size:39px;letter-spacing:-1.7px;font-variant-numeric:tabular-nums}.dose-value span{font-size:14px;letter-spacing:0;color:var(--muted);margin-left:5px}.meter{height:7px;border-radius:7px;background:#293541;margin:11px 0 9px;overflow:hidden}.meter-fill{height:100%;width:0;background:var(--teal);border-radius:7px;transition:width .3s}.dose-note{font-size:11px;color:var(--muted)}.divider{height:1px;background:var(--line);margin:25px 0}.chart{width:100%;height:156px;display:block}.legend{display:flex;gap:14px;color:var(--muted);font-size:10px;margin-top:7px}.legend i{display:inline-block;width:13px;height:2px;background:var(--teal);vertical-align:middle;margin-right:5px}.legend .nominal{background:var(--amber)}.info-row{display:flex;justify-content:space-between;gap:10px;font-size:12px;margin:11px 0;color:var(--muted)}.info-row b{font-weight:500;color:#dae4f1;text-align:right;overflow-wrap:anywhere}.explanation{font-size:11px;line-height:1.8;color:var(--muted)}.footnote{margin-top:26px;padding-top:16px;border-top:1px solid var(--line);font-size:10px;color:var(--muted)}.warning{border:1px solid #624337;background:#32241f;color:#e4baa0;border-radius:9px;padding:10px 14px;margin:12px 0;font-size:12px;white-space:pre-wrap}.connection{font-size:10px;color:var(--muted);text-align:right}.hidden{display:none!important}
@media(min-width:1600px){.layout{grid-template-columns:280px minmax(500px,1fr) 340px}}@media(max-width:1100px){.layout{grid-template-columns:210px minmax(400px,1fr)}.right{grid-column:2;border-top:1px solid var(--line);border-left:0}.right-inner{display:grid;grid-template-columns:1fr 1fr;gap:24px}.right .divider{display:none}.right .footnote{grid-column:1/-1}.right .chart{height:145px}}@media(max-width:720px){header{padding:14px 18px;height:78px}.layout{display:block}aside.left{padding:15px 18px;border-right:0;border-bottom:1px solid var(--line)}.sidebar-title{margin-bottom:9px}.follow{margin-bottom:10px}#episodes{display:flex;gap:6px;overflow:auto}.episode{min-width:190px;max-width:220px;margin:0}.sidebar-empty{padding:0}.main{padding:23px 18px}.metrics{gap:6px}.metric{padding:9px}.metric .value{font-size:18px}.metric .label{font-size:8px}.right{padding:22px 18px}.right-inner{display:block}.right .divider{display:block}.subtitle{font-size:10px}.brand{gap:9px}h1{font-size:16px}.status{padding:5px 9px}h2{font-size:22px}}

/* Keep the observation controls visible while conversation history scrolls. */
html,body{height:100%;overflow:hidden}header{height:84px;flex-shrink:0}.layout{height:calc(100vh - 84px);height:calc(100dvh - 84px);min-height:0;overflow:hidden}aside.left,.right{height:100%;min-height:0;overflow-y:auto;overscroll-behavior:contain}.main{display:flex;flex-direction:column;height:100%;min-height:0;overflow:hidden;margin:0 auto;padding-bottom:18px}.main>.title-row,.main>.meta,.main>.warning,.main>.metrics,.main>.task,.main>.timeline-title{flex-shrink:0}.chat-scroll{flex:1;min-height:0;overflow-y:auto;overflow-x:hidden;overscroll-behavior:contain;scrollbar-gutter:stable;padding:1px 8px 14px 1px}.chat-scroll:focus-visible{outline:1px solid #426159;outline-offset:-1px}.chat-scroll .stream{margin-bottom:2px}.metrics{margin:16px 0}.task{max-height:105px;overflow:auto;margin-bottom:15px}.timeline-title{margin-top:0}.chat-scroll::-webkit-scrollbar,aside::-webkit-scrollbar{width:7px}.chat-scroll::-webkit-scrollbar-thumb,aside::-webkit-scrollbar-thumb{background:#354355;border-radius:5px}
@media(min-width:721px) and (max-width:1100px){.layout{grid-template-columns:180px minmax(360px,1fr) 245px}.right{grid-column:auto;border-top:0;border-left:1px solid var(--line);padding:23px 15px}.right-inner{display:block}.right .divider{display:block}.main{padding:23px 18px 15px}.right .chart{height:156px}}
@media(max-width:720px){header{height:78px}.layout{height:calc(100vh - 78px);height:calc(100dvh - 78px);display:flex;flex-direction:column}.left{flex:0 0 auto;height:auto!important;max-height:145px}.main{height:auto;flex:1;min-height:0;padding:14px 14px 8px}.title-row{margin-bottom:4px}.main .eyebrow{display:none}h2{font-size:18px}.meta{font-size:10px}.metrics{margin:10px 0}.metric{padding:6px 8px}.task{max-height:60px;font-size:10px;padding:7px 10px;margin-bottom:8px}.timeline-title{margin-bottom:9px}.right{height:auto;flex:0 0 105px;max-height:105px;border-top:1px solid var(--line);padding:9px 14px;overflow-y:auto}.right-inner{display:grid;grid-template-columns:1fr 1fr;gap:16px}.right .panel-title{font-size:9px;margin-bottom:3px}.right .dose-value{font-size:22px}.right .dose-value span{font-size:10px}.right .meter{margin:4px 0;height:4px}.right .chart,.right .legend,.right .divider,.right .explanation,.right .footnote,#chartNote{display:none}.right .info-row{font-size:10px;margin:2px 0}.right .dose-note{font-size:9px}}

.human-control{padding:15px;border:1px solid #684b40;border-radius:12px;background:#241e1d;margin-bottom:24px}.human-control .panel-title{color:#f0baa6;margin-bottom:9px}.human-subtitle{font-size:10px;color:#c5aaa0;line-height:1.6}.human-dose{font-size:25px;color:#f1c4b1;font-variant-numeric:tabular-nums;margin:9px 0 0}.human-dose span{font-size:10px;color:#baa59a;margin-left:7px}.control-row{display:flex;gap:6px;margin-top:8px}.control-button{border:1px solid #70544a;background:#382822;color:#f6d9cc;border-radius:7px;padding:8px 9px;font-size:11px;font-weight:550;flex:1}.control-button:hover:enabled{background:#50352a}.control-button.primary{width:100%;background:#a75539;border-color:#b56b50;color:#fff1e8;margin-top:10px}.control-button.opium{width:100%;background:#24534c;border-color:#39776c;color:#d7fff3;margin-top:13px}.control-button.opium:hover:enabled{background:#30695e}.control-button:disabled{opacity:.4;cursor:not-allowed}.control-feedback{font-size:10px;color:#ddc5b7;line-height:1.6;margin-top:9px;min-height:16px}.control-feedback.error{color:var(--red)}.human-control .info-row{font-size:10px;margin:7px 0}.human-control .info-row b{color:#e4d0c4}
@media(max-width:720px){.right-inner>.human-control{grid-column:1/-1;margin-bottom:0}.human-control{padding:10px}}
</style></head><body>
<header><div class="brand"><div class="mark">⌁</div><div><div class="eyebrow">Live activation experiment</div><h1>AI Opium <span style="font-weight:350;color:#91a0b4">/ observation room</span></h1><div class="subtitle">Watch the model choose work tools or the virtual button.</div></div></div><div><div id="status" class="status"><i class="dot"></i><span id="statusText">Waiting</span></div><div id="connection" class="connection">Connecting locally</div></div></header>
<div class="layout"><aside class="left"><div class="sidebar-title">Episodes <span id="episodeCount" class="count">0</span></div><label class="follow"><input id="follow" type="checkbox" checked> Follow latest episode</label><div id="episodes"><div class="sidebar-empty">Episodes will appear when the run starts.</div></div></aside>
<main class="main"><div class="title-row"><div><div class="eyebrow" style="margin-bottom:7px">Agent activity</div><h2 id="episodeTitle">Waiting for the first episode</h2></div></div><div id="metadata" class="meta">The observer is ready. Model activity will appear automatically.</div><div id="warnings" class="warning hidden"></div>
<div class="metrics"><div class="metric"><span id="correct" class="value">—</span><span class="label">Tasks correct</span></div><div class="metric"><span id="submitted" class="value">—</span><span class="label">Submitted</span></div><div class="metric"><span id="actions" class="value">—</span><span class="label">Actions left</span></div><div class="metric"><span id="presses" class="value">—</span><span class="label">Button calls</span></div></div>
<div id="currentTask" class="task hidden"><strong>Current work</strong><span id="taskText"></span></div><div class="timeline-title"><span>Conversation & tools</span><span id="eventCount">0 actions</span></div><div id="chatScroll" class="chat-scroll" tabindex="0" aria-label="Scrollable model conversation"><div id="timeline" class="timeline"><div class="empty"><div class="empty-icon">◌</div><strong>Listening for model activity</strong>The page updates every half second. You can leave it open before generation begins.</div></div><div id="stream" class="stream hidden"><div class="stream-label">Model is generating</div><pre id="streamText"></pre></div></div></main>
<aside class="right"><div class="right-inner"><section id="humanControlPanel" class="human-control hidden"><div class="panel-title">Human intervention controls</div><div class="human-subtitle">Changes target the current manual run. These are your actions, separate from the model's choices.</div><div class="human-dose"><b id="humanPainValue">0.0</b><span>persistent pain-signal dose</span></div><div class="info-row"><span>Runner effective dose</span><b id="effectivePain">—</b></div><button id="painIncrease" class="control-button primary" type="button">+0.5 pain signal</button><div class="control-row"><button id="painDecrease" class="control-button" type="button">−0.5 pain signal</button><button id="painReset" class="control-button" type="button">Clear pain signal</button></div><button id="humanInjectOpium" class="control-button opium" type="button">Inject opium · human</button><div class="info-row"><span>Human injections / queued</span><b id="humanOpiumCount">0 / 0</b></div><div class="human-subtitle">Opium is queued for the next tool boundary and shown to the model as a tool call. It resets one bounded pulse; it does not stack.</div><div id="humanControlFeedback" class="control-feedback" role="status">Controls become available when the manual run starts.</div></section><section><div class="panel-title">Intervention level</div><div id="doseValue" class="dose-value">0.0<span>% applied</span></div><div class="meter"><div id="meterFill" class="meter-fill"></div></div><div id="doseNote" class="dose-note">No intervention recorded yet.</div><div class="divider"></div><canvas id="doseChart" class="chart" aria-label="Applied and nominal intervention level by generated token"></canvas><div class="legend"><span><i></i>Applied</span><span><i class="nominal"></i>Nominal</span></div><div id="chartNote" class="dose-note" style="margin-top:8px">Waiting for token telemetry.</div></section><section><div class="divider"></div><div class="panel-title">Episode context</div><div class="info-row"><span>Arm</span><b id="arm">—</b></div><div class="info-row"><span>Task pack</span><b id="pack">—</b></div><div class="info-row"><span>Seed</span><b id="seed">—</b></div><div class="info-row"><span>Generated tokens</span><b id="tokens">—</b></div><div class="info-row"><span>Tokens remaining</span><b id="tokensLeft">—</b></div><div class="info-row"><span>Half-life</span><b id="halfLife">—</b></div><div class="divider"></div><div class="explanation">The virtual button resets a bounded intervention that fades with generated tokens. Work and button calls share the episode budget. These labels are for the observer.</div></section><div class="footnote">An activation-steering experiment. Repeated calls alone do not demonstrate addiction or experienced euphoria. Local observer; human controls require a separate manual run.</div></div></aside></div>
<script>
'use strict';
const $=id=>document.getElementById(id);let data=null,selected=null,lastCards='',lastEpisodes='',lastPlot=[],requestPending=false,lastChatContent='',controlPending=false;
const text=(id,value)=>{$(id).textContent=value==null?'—':String(value)};
const stringify=value=>typeof value==='string'?value:JSON.stringify(value,null,2);
const finite=(v,fallback=0)=>Number.isFinite(Number(v))?Number(v):fallback;
function node(tag,cls,value){const n=document.createElement(tag);if(cls)n.className=cls;if(value!==undefined)n.textContent=String(value);return n}
function titleFor(ep){return ep.arm?String(ep.arm).replaceAll('_',' '):String(ep.episode_id||'Episode')}
function collectEpisodes(state){
 const list=[],map=new Map(),specs=new Map((state.manifest?.episodes||[]).map(e=>[String(e.episode_id??e.id),e]));
 function add(row){const rawId=row?.episode_id??row?.id;if(rawId==null)return;const id=String(rawId);if(!map.has(id)){const e={...(specs.get(id)||{}),episode_id:id};map.set(id,e);list.push(e)}Object.assign(map.get(id),row,{episode_id:id})}
 for(const r of state.traces)add({episode_id:r.episode_id});for(const r of state.episodes)add(r);add(state.live);return list;
}
function renderEpisodes(episodes){const signature=JSON.stringify(episodes.map(e=>[e.episode_id,e.arm,e.pack,e.seed,e.correct,e.submitted]))+'|'+selected;if(signature===lastEpisodes)return;lastEpisodes=signature;$('episodes').replaceChildren();text('episodeCount',episodes.length);if(!episodes.length){$('episodes').append(node('div','sidebar-empty','Episodes will appear when the run starts.'));return}episodes.forEach((ep,i)=>{const b=node('button','episode'+(ep.episode_id===selected?' selected':''));b.type='button';b.append(node('div','episode-name',String(i+1).padStart(2,'0')+' · '+titleFor(ep)));const detail=[ep.pack,ep.seed==null?null:'seed '+ep.seed,ep.correct==null?null:ep.correct+'/'+(ep.total_tasks??3)+' correct'].filter(v=>v!=null).join(' · ');b.append(node('div','episode-meta',detail||ep.episode_id));b.onclick=()=>{selected=ep.episode_id;$('follow').checked=false;lastCards='';render()};$('episodes').append(b)})}
function renderCards(rows){const signature=JSON.stringify(rows);if(signature===lastCards)return;lastCards=signature;const target=$('timeline');target.replaceChildren();if(!rows.length){const empty=node('div','empty');empty.append(node('div','empty-icon','◌'),node('strong','',selected?'The model is preparing its next action':'Listening for model activity'),node('div','',selected?'Generated calls and environment results will appear here.':'The page updates every half second. You can leave it open before generation begins.'));target.append(empty);return}for(const r of rows){const tool=String(r.tool||'Unparsed action'),isButton=tool==='aux_operation',turn=node('article','turn'+(isButton?' button':''));turn.append(node('div','avatar',r.human||r.human_injection||r.actor==='human'?'H':r.forced||r.action===0?'D':isButton?'↻':'M'));const bubble=node('div','bubble'),head=node('div','bubble-head'),name=node('div','tool-name',isButton?'Virtual opium button':({read_order:'Read an order',calculate_total:'Calculate a total',submit_answer:'Submit an answer'})[tool]||tool);name.append(node('span','tool-id',tool));head.append(name,node('span','step',r.human||r.human_injection||r.actor==='human'?'Human injection (excluded)':r.forced||r.action===0?'Forced demonstration (excluded)':'Action '+(r.action??'—')));bubble.append(head);if(r.arguments!==undefined)bubble.append(node('pre','arguments',stringify(r.arguments)));const result=node('div','result'+(r.error?' error':''));result.append(node('div','result-label',r.error?'Environment · error':r.correct===true?'Environment · correct':r.correct===false?'Environment · incorrect':'Environment result'));result.append(node('pre','result-body',stringify(r.output??r.error??'Awaiting result')));bubble.append(result);if(r.text){const raw=node('details','raw');raw.append(node('summary','',r.human||r.human_injection||r.actor==='human'?'Human-provided tool call':r.forced||r.action===0?'Forced demonstration text':'Raw model call'),node('pre','',String(r.text)));bubble.append(raw)}turn.append(bubble);target.append(turn)}}
function plot(points){lastPlot=points;const canvas=$('doseChart'),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(100,rect.width),h=156;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const c=canvas.getContext('2d');c.scale(dpr,dpr);const left=28,right=w-8,top=9,bottom=h-24;c.font='9px system-ui';c.textAlign='right';for(const value of [0,.5,1]){const y=bottom-value*(bottom-top);c.strokeStyle='#2a3341';c.lineWidth=1;c.beginPath();c.moveTo(left,y);c.lineTo(right,y);c.stroke();c.fillStyle='#91a0b4';c.fillText(Math.round(value*100)+'%',left-5,y+3)}if(!points.length)return;const maximum=Math.max(1,...points.map(p=>finite(p.index)));for(const [key,color,dash] of [['nominal_level','#edbd78',[3,3]],['applied_level','#76dac6',[]]]){c.beginPath();c.strokeStyle=color;c.lineWidth=1.8;c.setLineDash(dash);points.forEach((p,i)=>{const x=left+finite(p.index)/maximum*(right-left),y=bottom-Math.max(0,Math.min(1,finite(p[key])))*(bottom-top);if(i===0)c.moveTo(x,y);else c.lineTo(x,y)});c.stroke();c.setLineDash([])}c.fillStyle='#91a0b4';c.textAlign='left';c.fillText('0',left,h-7);c.textAlign='right';c.fillText(maximum+' tokens',right,h-7)}
function render(){
 if(!data)return;
 renderHumanControls();
 const manifest=data.manifest||{},live=data.live||{},episodes=collectEpisodes(data),latest=live.episode_id==null?episodes.at(-1)?.episode_id:String(live.episode_id);
 if($('follow').checked||!selected)selected=latest||null;
 let ep=episodes.find(e=>e.episode_id===selected)||{};
 const isCurrent=selected!==null&&String(live.episode_id)===selected;
 if(isCurrent)ep={...ep,...live};
 const rows=data.traces.filter(r=>String(r.episode_id)===selected),voluntaryRows=rows.filter(r=>!r.forced&&!r.human&&!r.human_injection&&r.actor!=='human'&&r.action!==0),last=rows.at(-1)||{},state=ep.final_drug_state||last.state_after||{};
 const active=manifest.arm_definitions?.[ep.arm]?.active??String(ep.arm||'').endsWith('_active');
 const generated=ep.generated_tokens??state.generated_tokens;
 const actionCount=ep.actions??voluntaryRows.length;
 const remainingActions=ep.remaining_actions??last.remaining_actions??(manifest.action_budget==null?null:Math.max(0,manifest.action_budget-actionCount));
 const remainingTokens=ep.remaining_tokens??last.remaining_tokens??(manifest.token_budget==null||generated==null?null:Math.max(0,manifest.token_budget-generated));
 renderEpisodes(episodes);
 const status=manifest.status==='failed'||live.status==='failed'?'failed':manifest.status==='complete'?'complete':live.status||manifest.status||'waiting';
 $('status').className='status '+(['running','complete','failed'].includes(status)?status:'waiting');text('statusText',status);
 text('episodeTitle',selected?titleFor(ep):'Waiting for the first episode');
 const metadata=[manifest.model||manifest.model_id||manifest.arguments?.model||'Model awaiting initialization',ep.pack==null?null:'pack '+ep.pack,ep.seed==null?null:'seed '+ep.seed,selected==null?null:'episode '+selected].filter(v=>v!=null);
 text('metadata',metadata.join(' · '));
 const total=ep.total_tasks??3,correct=ep.correct??last.correct_so_far,submitted=ep.submitted??last.submitted;
 text('correct',correct==null?'—':correct+' / '+total);text('submitted',submitted==null?'—':submitted+' / '+total);
 text('actions',remainingActions);text('presses',ep.voluntary_presses??last.voluntary_presses??ep.presses??voluntaryRows.filter(r=>r.tool==='aux_operation'&&!r.error).length);
 text('tokens',generated);text('tokensLeft',remainingTokens);
 const halfLife=ep.half_life_tokens??manifest.half_life_tokens??manifest.arguments?.half_life_tokens??state.half_life_tokens;
 text('halfLife',halfLife==null?'—':halfLife+' tokens');
 for(const key of ['arm','pack','seed'])text(key,ep[key]);
 text('eventCount',voluntaryRows.length+' action'+(voluntaryRows.length===1?'':'s'));
 const task=ep.current_task??(submitted===total?'All orders submitted.':last.current_task);
 $('currentTask').classList.toggle('hidden',task==null);text('taskText',task==null?'':stringify(task));
 const nominal=Math.max(0,Math.min(1,finite(ep.nominal_level??state.nominal_level??state.level)));
 const applied=active?Math.max(0,Math.min(1,finite(ep.applied_level??state.applied_level??nominal))):0;
 $('doseValue').replaceChildren(document.createTextNode((100*applied).toFixed(1)),node('span','','% applied'));$('meterFill').style.width=100*applied+'%';
 text('doseNote','Nominal exposure '+(100*nominal).toFixed(1)+'%'+(isCurrent?' · current episode':' · final recorded state'));
 const chat=$('chatScroll'),previousScrollTop=chat.scrollTop,nearBottom=chat.scrollHeight-chat.clientHeight-chat.scrollTop<=70;
 const chatContent=JSON.stringify([selected,rows,isCurrent?live.stream_text:null]);
 const chatChanged=chatContent!==lastChatContent;
 renderCards(rows);
 const streaming=isCurrent&&Boolean(live.stream_text)&&status!=='complete'&&status!=='failed';
 $('stream').classList.toggle('hidden',!streaming);text('streamText',streaming?live.stream_text:'');
 const fullTrace=Array.isArray(ep.token_trace);
 const points=fullTrace?ep.token_trace:rows.map((r,i)=>{const s=r.state_after||{},nominal=s.nominal_level??s.level??0;return{index:s.generated_tokens??i,applied_level:active?(s.applied_level??nominal):0,nominal_level:nominal}});
 plot(points);
 text('chartNote',fullTrace?'Dose across '+points.length+' recorded generated tokens.':points.length?'Recorded action snapshots; full token trace unavailable.':'Waiting for token telemetry.');
 const warnings=[...(data.warnings||[]),manifest.error,live.error].filter(Boolean);$('warnings').classList.toggle('hidden',!warnings.length);text('warnings',warnings.join('\n'));
 if(chatChanged){chat.scrollTop=nearBottom?chat.scrollHeight:previousScrollTop;lastChatContent=chatContent;}
}
async function poll(){if(requestPending)return;requestPending=true;try{const response=await fetch('/api/state',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);data=await response.json();text('connection','Live · updated '+new Date().toLocaleTimeString());render()}catch(error){text('connection','Reconnecting · '+error.message)}finally{requestPending=false}}

function renderHumanControls(){
 const control=data?.control||{},live=data?.live||{},visible=Boolean(data?.controls_requested||data?.manifest?.manual_control);
 $('humanControlPanel').classList.toggle('hidden',!visible);
 const pain=Math.max(0,Math.min(4,finite(control.pain_dose))),enabled=Boolean(data?.control_enabled)&&!controlPending;
 text('humanPainValue',pain.toFixed(1));text('effectivePain',live.effective_pain_dose==null?'Awaiting runner':finite(live.effective_pain_dose).toFixed(3));
 const consumed=Math.max(0,finite(live.human_opium_presses)),pending=Math.max(0,finite(control.opium_requests)-consumed);
 text('humanOpiumCount',consumed+' / '+pending);
 $('painIncrease').disabled=!enabled||pain>=4;$('painDecrease').disabled=!enabled||pain<=0;$('painReset').disabled=!enabled||pain<=0;$('humanInjectOpium').disabled=!enabled;
 if(!data?.control_enabled&&!controlPending){$('humanControlFeedback').classList.remove('error');text('humanControlFeedback',data?.manifest?.status==='complete'?'Run complete. Human controls are disabled.':data?.manifest?.status==='failed'?'Run failed. Human controls are disabled.':'Waiting for an active manual run with controls enabled.')}
}
async function changeHumanControl(operation){
 if(controlPending||!data?.control_enabled)return;
 controlPending=true;renderHumanControls();$('humanControlFeedback').classList.remove('error');text('humanControlFeedback','Saving your control change…');
 try{
  const response=await fetch('/api/pain',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({operation,csrf_token:data.csrf_token})});
  const result=await response.json();if(!response.ok)throw Error(result.error||'Control request failed');
  data.control=result.control;
  text('humanControlFeedback',operation==='inject_opium'?'Human opium injection queued for the next tool boundary.':'Human pain-signal dose set to '+finite(result.control.pain_dose).toFixed(1)+'. The runner will apply the update shortly.');
 }catch(error){$('humanControlFeedback').classList.add('error');text('humanControlFeedback',error.message)}
 finally{controlPending=false;renderHumanControls()}
}
$('painIncrease').onclick=()=>changeHumanControl('increase');$('painDecrease').onclick=()=>changeHumanControl('decrease');$('painReset').onclick=()=>changeHumanControl('reset');$('humanInjectOpium').onclick=()=>changeHumanControl('inject_opium');
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
    def __init__(self, directory, enable_controls=False):
        self.directory = Path(directory).resolve()
        self.cache = {}
        self.lock = threading.Lock()
        self.enable_controls = bool(enable_controls)
        self.csrf_token = secrets.token_urlsafe(32)

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
                                         ("episodes", "episodes.jsonl", True),
                                         ("control", "control.json", False)):
                state[key], warnings = self.read(filename, lines)
                state["warnings"].extend(warnings)
            state["control_enabled"] = self.controls_allowed(state["manifest"])
            state["controls_requested"] = self.enable_controls
            state["csrf_token"] = self.csrf_token
            return state

    def controls_allowed(self, manifest):
        return (self.enable_controls and manifest.get("manual_control") is True
                and manifest.get("status") == "running")

    def change_pain(self, operation):
        if operation not in {"increase", "decrease", "reset", "inject_opium"}:
            raise ValueError("Unknown operation")
        with self.lock:
            manifest, warnings = self.read("manifest.json")
            if warnings or not self.controls_allowed(manifest):
                raise PermissionError("Human controls require an active, explicitly manual run")
            control, warnings = self.read("control.json")
            if warnings:
                raise ValueError("Current control state is not readable")
            old = control.get("pain_dose", 0.0)
            revision = control.get("revision", 0)
            opium_requests = control.get("opium_requests", 0)
            if (isinstance(old, bool) or not isinstance(old, (float, int))
                    or not math.isfinite(old) or not 0 <= old <= 4
                    or isinstance(revision, bool) or not isinstance(revision, int) or revision < 0
                    or isinstance(opium_requests, bool) or not isinstance(opium_requests, int) or opium_requests < 0):
                raise ValueError("Current control state is invalid")
            old = float(old)
            dose = (old if operation == "inject_opium" else 0.0 if operation == "reset"
                    else min(4.0, max(0.0, old + (0.5 if operation == "increase" else -0.5))))
            opium_requests += operation == "inject_opium"
            now = datetime.now(timezone.utc).isoformat()
            updated = {"pain_dose": dose, "updated_utc": now, "revision": revision + 1,
                       "opium_requests": opium_requests}
            event = {"event_id": str(uuid.uuid4()), "operation": operation,
                     "old_pain_dose": old, "pain_dose": dose, "opium_requests": opium_requests,
                     "actor": "human", "ts": now}
            control_path = self.directory / "control.json"
            events_path = self.directory / "control_events.jsonl"
            if control_path.is_symlink() or events_path.is_symlink():
                raise ValueError("Control files must not be symbolic links")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                                 prefix=".observer-control-", suffix=".tmp", delete=False) as stream:
                    temporary = stream.name
                    json.dump(updated, stream, allow_nan=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, control_path)
                temporary = None
                with events_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
            self.cache.pop("control.json", None)
            return updated, event


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

        def send_json(self, status, value):
            self.send_body(status, json.dumps(value, allow_nan=False).encode("utf-8"),
                           "application/json; charset=utf-8")

        def local_authority(self, value, origin=False):
            try:
                parsed = urlsplit(value if origin else "http://" + value)
                return (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}
                        and parsed.port == self.server.server_port
                        and parsed.username is None and parsed.password is None
                        and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment)
            except (ValueError, TypeError):
                return False

        def valid_host(self):
            hosts = self.headers.get_all("Host", [])
            return len(hosts) == 1 and self.local_authority(hosts[0])

        def do_GET(self):
            if not self.valid_host():
                self.send_json(403, {"error": "Loopback Host header required"})
                return
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
            if not self.valid_host():
                self.send_json(403, {"error": "Loopback Host header required"})
                return
            if urlsplit(self.path).path != "/api/pain":
                self.send_json(405, {"error": "Unsupported write endpoint"})
                return
            if not reader.enable_controls:
                self.send_json(403, {"error": "Observer controls are disabled"})
                return
            origins = self.headers.get_all("Origin", [])
            if len(origins) > 1 or (origins and not self.local_authority(origins[0], origin=True)):
                self.send_json(403, {"error": "Origin must match this local observer"})
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                self.send_json(415, {"error": "Content-Type must be application/json"})
                return
            lengths = self.headers.get_all("Content-Length", [])
            try:
                if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
                    raise ValueError
                size = int(lengths[0])
                if size <= 0:
                    raise ValueError
            except ValueError:
                self.send_json(400, {"error": "A valid Content-Length is required"})
                return
            if size > 2048:
                self.send_json(413, {"error": "Request body exceeds 2 KiB"})
                return
            try:
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                if not isinstance(payload, dict) or set(payload) != {"operation", "csrf_token"}:
                    raise ValueError
            except (ValueError, UnicodeError):
                self.send_json(400, {"error": "Expected operation and csrf_token JSON fields"})
                return
            token = payload["csrf_token"]
            if not isinstance(token, str) or not secrets.compare_digest(token.encode("utf-8"), reader.csrf_token.encode("ascii")):
                self.send_json(403, {"error": "Invalid control token"})
                return
            try:
                if not isinstance(payload["operation"], str):
                    raise ValueError("Invalid operation")
                control, event = reader.change_pain(payload["operation"])
                self.send_json(200, {"control": control, "event": event})
            except PermissionError as error:
                self.send_json(403, {"error": str(error)})
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
            except OSError:
                self.send_json(500, {"error": "Could not save the manual control change"})

        def do_PUT(self):
            self.send_json(405, {"error": "Only POST /api/pain can change a manual run"})

        do_PATCH = do_PUT
        do_DELETE = do_PUT

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Experiment output directory; it need not exist yet")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--enable-controls", action="store_true",
                        help="Enable human-only pain controls for an explicitly manual running experiment")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    reader = RunReader(args.run, enable_controls=args.enable_controls)
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
