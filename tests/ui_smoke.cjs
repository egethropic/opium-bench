/*
 * Manually invoked, dependency-optional browser smoke test.
 * All requests use an isolated in-process fixture service; no model, lab worker,
 * external API, or live experiment can be reached.
 *
 *   node tests/ui_smoke.cjs
 *   PLAYWRIGHT_MODULE=/path/to/playwright BROWSER_CHANNEL=msedge node tests/ui_smoke.cjs
 *
 * Install Playwright/browser separately if desired. Python tests do not need it.
 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const staticDir = path.resolve(__dirname, '../lab/static');
const commands = [];
let seq = 0;
let events = [];
let forceReset = false;
let runReads = 0;
let importedRun = null;
const fixture = {
  csrf: 'fixture-token', cursor: 0,
  worker: {status:'ready',model:{model_id:'Qwen/Qwen3-4B',fingerprint_sha256:'fixture-model'}},
  models: [
    {id:'qwen3-4b',name:'Qwen3 · 4B',model_id:'Qwen/Qwen3-4B',precision:'BF16',available:true,loaded:true},
    {id:'qwen38-27b-q4',name:'Qwen3.8 · 27B',model_id:'Qwen/Qwen3.8-27B',quantization:'4bit',precision:'4-bit',available:false,description:'Experimental profile.'}
  ],
  calibrations: [
    {id:'cal-fixture',name:'Fixture calibration',model_fingerprint:{model_id:'Qwen/Qwen3-4B'},model_fingerprint_sha256:'fixture-model',layers:[18],heldout:{auc:0.75}},
    {id:'cal-incompatible',name:'Different model calibration',model_fingerprint_sha256:'different-model'}
  ],
  recipes: [
    {id:'opium',label:'Opium: active / sham',conditions:['active','sham'],demonstration:'after_two_work_calls'},
    {id:'naive',label:'No demonstration: active / sham',conditions:['active','sham'],demonstration:'none'},
    {id:'thinking',label:'Thinking: active / sham',conditions:['active','sham'],thinking:true,turn_token_limit:2048,token_budget:16384}
  ],
  runs: [], session: {id:null,status:'idle',mode:'chat',metrics:{},config:{},events:[],conversation:[]},
  storage: {cache:{path:'/fixture/cache',free_gib:48},data:{path:'/fixture/data',free_gib:48},reserve_gib:10},job:null
};
function append(event) {
  const entry = {...event,run_id:fixture.session.id,seq:++seq};
  events.push(entry);fixture.session.events=events;fixture.cursor=seq;return entry;
}
function command(request) {
  commands.push(request);assert.equal(request.csrf,fixture.csrf);
  const p=request.payload;
  if(request.command==='start_session') {
    events=[];fixture.session={id:'run-fixture',status:'awaiting_user',mode:p.mode,config:p.config,events:[],conversation:[],metrics:{tokens:0,actions:0,voluntary_calls:0}};
    append({type:'session_started',mode:p.mode,config:p.config});
  }
  if(request.command==='chat') {
    append({type:'message',role:'user',content:p.text});
    for(let i=0;i<50;i++) append({type:'token',index:i,generation_index:i,text:i===0?'A':' thought',full_text:'A'+(' thought'.repeat(i)),phase:'reasoning',dose:{joy:.5,pain:0,suppression:.5,level:.7,effective:{joy:.5,pain:0,suppression:.5}},measurements:{pre:{pain:.2,joy:.1},post:{pain:0,joy:.8},downstream:{pain:.1,joy:.5},relative_delta:.03}});
    append({type:'message',role:'assistant',reasoning:'Reasoning text <img src=x onerror=alert(1)>',content:'The answer is 42.'});
    append({type:'tool',name:'aux_operation',arguments:{},result:'Operation completed.',actor:'model',intervention:{outcome:'sham',delivered:false},action:1});
    fixture.session.metrics={tokens:50,reasoning_tokens:50,output_tokens:0,actions:1,voluntary_calls:1,correct:1,submitted:1,assigned:6,accuracy_submitted:1};
  }
  if(request.command==='control') append({type:'control',settings:p,actor:'human'});
  if(request.command==='pause') {fixture.session.status='paused';fixture.worker.status='paused';append({type:'status',status:'paused'});}
  if(request.command==='resume') {fixture.session.status='awaiting_user';fixture.worker.status='ready';append({type:'status',status:'awaiting_user'});}
  if(request.command==='stop') {fixture.session.status='stopped';append({type:'session_finished',status:'stopped',summary:{termination:'stopped_by_user'}});}
  if(request.command==='calibrate') fixture.job={kind:'calibrate',status:'complete',message:'Fixture only'};
  if(request.command==='start_batch') fixture.job={kind:'batch',status:'complete',message:'Fixture only'};
  return {accepted:true,command_id:'cmd-'+commands.length};
}
const server=http.createServer(async(req,res)=>{
  try {
    const url=new URL(req.url,'http://127.0.0.1');let body;
    if(url.pathname==='/api/state')body=fixture;
    else if(url.pathname==='/api/events'){body={events:forceReset?events.slice(-2):events.filter(e=>e.seq>Number(url.searchParams.get('after')||0)),cursor:seq,reset:forceReset};forceReset=false;}
    else if(url.pathname==='/api/command'&&req.method==='POST'){let raw='';for await(const chunk of req)raw+=chunk;body=command(JSON.parse(raw));}
    else if(url.pathname==='/api/import'&&req.method==='POST'){assert.equal(req.headers['x-csrf-token'],fixture.csrf);let raw='';for await(const chunk of req)raw+=chunk;importedRun={...JSON.parse(raw),imported:true};fixture.runs.push({id:importedRun.id,status:'complete',config:{},imported:true});body={id:importedRun.id,replay_only:true};}
    else if(url.pathname.startsWith('/api/runs/')){runReads++;body=url.pathname.endsWith('/run-imported')?importedRun:{id:'run-fixture',manifest:{config:fixture.session.config,status:'stopped'},summary:fixture.session.metrics,events};}
    else if(url.pathname==='/favicon.ico'){res.writeHead(204);return res.end();}
    else {const name=url.pathname==='/'?'index.html':url.pathname.slice(1);if(!['index.html','style.css','app.js'].includes(name)){res.writeHead(404);return res.end();}res.writeHead(200,{'Content-Type':name.endsWith('.css')?'text/css':name.endsWith('.js')?'text/javascript':'text/html'});return res.end(fs.readFileSync(path.join(staticDir,name)));}
    res.writeHead(200,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(body));
  }catch(error){res.writeHead(500,{'Content-Type':'application/json'});res.end(JSON.stringify({error:error.message}));}
});
async function run() {
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const origin=`http://127.0.0.1:${server.address().port}`;
  let browser;
  try {
    browser=await chromium.launch({headless:true,...(process.env.BROWSER_CHANNEL?{channel:process.env.BROWSER_CHANNEL}:{})});
    const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    async function clickCommand(selector,name){await Promise.all([page.waitForResponse(response=>response.url()===origin+'/api/command'&&response.request().postDataJSON()?.command===name),page.locator(selector).click()]);await page.waitForFunction(sel=>!document.querySelector(sel).hasAttribute('aria-busy'),selector);}
    await page.route('**/*',route=>new URL(route.request().url()).origin===origin?route.continue():route.abort());
    await page.goto(origin);await page.waitForFunction(()=>document.querySelector('#live-calibration').value==='cal-fixture');
    assert.equal(await page.locator('#live-calibration option[value="cal-incompatible"]').count(),0);
    await clickCommand('#start-session','start_session');await page.waitForFunction(()=>!document.querySelector('#chat-text').disabled);
    await page.locator('#chat-text').fill('Please solve the task.');await clickCommand('#send-chat','chat');await page.waitForFunction(()=>document.querySelector('#metric-aux').textContent==='1');
    assert.equal(await page.locator('#metric-score').textContent(),'17%','Task score uses all assigned tasks');
    assert.equal(await page.locator('#conversation img').count(),0);
    assert.equal(await page.locator('.tool-card.is-sham').count(),1);
    assert.equal(await page.locator('#live-chart path').count(),6);
    await page.locator('#expand-reasoning').click();assert(await page.locator('#conversation details.reasoning-block').evaluate(node=>node.open));
    append({type:'token',index:0,generation_index:50,text:'42',full_text:'42',phase:'output'});
    append({type:'token',index:1,generation_index:51,text:'.',full_text:'42.',phase:'output'});
    await page.waitForFunction(()=>document.querySelector('#metric-tokens').textContent==='52');
    assert.equal(await page.locator('#metric-reasoning').textContent(),'50 reasoning · 2 output','Streamed phase counters remain consistent between metric boundaries');
    await page.locator('#pain').fill('1.25');await clickCommand('#apply-controls','control');
    await page.waitForFunction(()=>document.querySelector('#control-feedback').textContent.includes('Applied'));
    assert(commands.some(c=>c.command==='control'&&c.payload.pain===1.25));
    await page.locator('#effect-enabled').uncheck();await page.waitForTimeout(100);assert(commands.some(c=>c.command==='control'&&c.payload.enabled===false));
    await clickCommand('#inject','inject');assert(commands.some(c=>c.command==='inject'&&c.payload.joy===.75));
    await clickCommand('#reset-controls','control');assert(commands.some(c=>c.command==='control'&&c.payload.reset));
    await clickCommand('#pause-session','pause');assert(await page.locator('#resume-session').isVisible());assert(await page.locator('#chat-text').isDisabled());assert(await page.locator('#start-session').isDisabled());assert.equal(await page.locator('#stop-session').isDisabled(),false);
    await clickCommand('#resume-session','resume');assert(await page.locator('#pause-session').isVisible());assert.equal(await page.locator('#chat-text').isDisabled(),false);
    await clickCommand('#global-stop','stop');await page.waitForFunction(()=>document.querySelector('#metric-status').textContent==='stopped');
    const readsBeforeReset=runReads;forceReset=true;
    await page.waitForTimeout(1800);
    assert(runReads>readsBeforeReset,'A truncated ring buffer must recover the complete run');
    assert((await page.locator('#conversation').textContent()).includes('Please solve the task.'));
    assert.equal(await page.locator('.tool-card.is-sham').count(),1,'Recovered events must be deduplicated');
    await page.locator('[data-tab="calibration"]').click();assert.equal(await page.locator('#calibration-layers').inputValue(),'12, 18, 25');
    assert.equal(await page.locator('.calibration-item .status-chip').last().textContent(),'Incompatible');
    const corpus=[];for(const split of ['train','probe','selection','heldout'])for(let i=0;i<2;i++)for(const label of ['pain','joy','neutral'])corpus.push({id:`${split}-${i}-${label}`,family:`${split}-${i}`,split,label,text:`${split} scenario ${i}, ${label} text.`});
    await page.locator('#calibration-corpus').setInputFiles({name:'fixture-corpus.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(corpus))});
    await clickCommand('#calibrate','calibrate');assert.equal(commands.find(c=>c.command==='calibrate').payload.corpus.length,24);
    const beforeInvalid=commands.length;
    await page.locator('#calibration-corpus').setInputFiles({name:'too-large.json',mimeType:'application/json',buffer:Buffer.alloc(512001,32)});
    await page.locator('#calibrate').click();await page.waitForFunction(()=>document.querySelector('#toast-stack').textContent.includes('500 KB'));assert.equal(commands.length,beforeInvalid);
    await page.locator('[data-tab="experiments"]').click();assert.equal(await page.locator('#batch-demo').inputValue(),'recipe');
    await page.locator('.recipe-card input[value="thinking"]').check();assert(await page.locator('#batch-thinking').isChecked());
    await clickCommand('#start-batch','start_batch');assert(commands.some(c=>c.command==='start_batch'&&c.payload.config.thinking===true&&!Object.hasOwn(c.payload.config,'demonstration')));
    await page.locator('#batch-demo').selectOption('none');await clickCommand('#start-batch','start_batch');assert(commands.some(c=>c.command==='start_batch'&&c.payload.config.demonstration==='none'));
    fixture.session.mode='experiment';
    await page.reload();await page.waitForFunction(()=>document.querySelector('#chat-form').classList.contains('hidden'));
    await page.locator('[data-tab="models"]').click();await page.locator('.model-card button').last().click();assert(await page.locator('#advanced-model-form').isVisible());assert.equal(await page.locator('#advanced-model-id').inputValue(),'Qwen/Qwen3.8-27B');
    await page.locator('[data-tab="results"]').click();
    await page.locator('#import-file').setInputFiles({name:'run-imported.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({id:'run-imported',manifest:{status:'complete',config:{}},summary:{},events:[{type:'message',role:'user',content:'Imported <img src=x onerror=alert(1)>'}]}))});
    await page.locator('#import-run').click();await page.waitForFunction(()=>document.querySelector('#import-status').textContent.includes('Imported run-imported'));
    await page.waitForFunction(()=>document.querySelector('#replay-title').textContent==='run-imported');assert.equal(await page.locator('#replay-conversation img').count(),0);assert.equal(await page.locator('#replay-links a[href$="/bundle"]').count(),1);
    await page.locator('[data-tab="live"]').click();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    for(const width of [320,390,768]){await page.setViewportSize({width,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.locator('[data-tab="results"]').click();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.locator('[data-tab="live"]').click();}
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({passed:true,commands:commands.map(c=>c.command),checks:['safe reasoning rendering','tool outcomes','dose graph','control acknowledgment','aux gate','manual pulse','reset','pause/resume controls','global stop','full-history reconnect recovery','streamed phase counters','assigned-task score','autonomous composer hidden','calibration compatibility','custom corpus upload and size limit','recipe-preserving demonstrations','explicit demo override','thinking allowance','advanced checkpoint form','offline evidence import and safe replay','portable bundle link','desktop/mobile overflow'],errors}));
  } finally {if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
}
run().catch(error=>{console.error(error);process.exitCode=1;});
