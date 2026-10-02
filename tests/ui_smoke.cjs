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
let releaseInitialState;
const initialStateReady=new Promise(resolve=>{releaseInitialState=resolve;});
const fixture = {
  csrf: 'fixture-token', cursor: 0,
  worker: {status:'ready',model:{model_id:'Qwen/Qwen3-4B',fingerprint_sha256:'fixture-model'}},
  models: [
    {id:'qwen3-4b',name:'Qwen3 · 4B',model_id:'Qwen/Qwen3-4B',precision:'BF16',available:true,loaded:true},
    {id:'qwen38-27b-q4',name:'Qwen3.8 · 27B',model_id:'Qwen/Qwen3.8-27B',quantization:'4bit',precision:'4-bit',available:false,description:'Experimental profile.'}
  ],
  calibrations: [
    {id:'cal-fixture',name:'Fixture calibration',model_fingerprint:{model_id:'Qwen/Qwen3-4B'},model_fingerprint_sha256:'fixture-model',layers:[18],heldout:{auc:0.75}},
    {id:'cal-research',name:'Research fixture',schema_version:2,status:'unvalidated',model_fingerprint_sha256:'fixture-model',continuation_scoring:{rubric:'fixture'}},
    {id:'cal-incompatible',name:'Different model calibration',model_fingerprint_sha256:'different-model'}
  ],
  protocols: [{id:'task_pressure',title:'Task pressure',question:'Matched pressure conditions',endpoints:['accuracy']}],
  research_capabilities:['recipe_v2','task_axis_v1'],
  research_jobs:[{id:'research-fixture',protocol_id:'task_pressure',title:'Task pressure',status:'partial'}],
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
  if(request.command==='preview_protocol')return {accepted:true,preview:{protocol_id:p.protocol_id,mode:p.mode,expansion_sha256:'a'.repeat(64),estimates:{episodes:1,max_generated_tokens:1024,additional_stage_token_reservation:0,storage_reservation_bytes:100000},required_capabilities:['recipe_v2','task_axis_v1'],unresolved_bindings:[],episodes:[{order:1,factors:{condition:'sham'},task_config:{difficulty:'hard',framing:'deadline'},recipe:{task_family:'orders',action_budget:12,token_budget:1024},pair_id:'pair-fixture'}]},preview_json:'{"exact":9007199254740999}'};
  if(request.command==='analyze_protocol'){fixture.job={kind:'research',status:'complete',research_result_status:'partial',message:'Research job partial: 0/14 cases complete; 14 failed'};return {accepted:true,analysis:{job_id:p.research_job_id,planned_episodes:2,total_recorded_attempts:2,observed_task_outcomes:1,attempt_policy:p.attempt_policy,records:[{id:'run-fixture',arm:'condition=active',status:'complete',outcomes:{task_accuracy:{numerator:1,denominator:2},aux_per_decision:{numerator:2,denominator:6},budget_units:8,tokens:100}},{arm:'condition=sham',status:'failed',outcomes:{}}],behavioral_reports:[{run_id:'run-fixture',included_attempt:true,report:{voluntary_calls:2,opportunities:6,exposure:{coverage:'partial',observed_token_events:100,observed_prefill_events:6},integrity_issues:[],phases:[{label:'active',voluntary_calls:2,opportunities:6,invalid_decisions:0}],transitions:[{from_label:'active',to_label:'sham',latency_status:'right_censored'}],presses:[],budget:{}}}],interpretation:'Exploratory paired evidence',...(p.endpoint?{contrast:{estimate:.2,missing_pairs:1}}:{})}};}
  if(request.command==='start_session') {
    events=[];fixture.session={restart_available:true,id:'run-fixture',status:'awaiting_user',mode:p.mode,config:p.config,events:[],conversation:[],metrics:{tokens:0,actions:0,voluntary_calls:0}};
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
  if(request.command==='stop') {fixture.session.status='stopped';fixture.runs=[{id:'run-fixture',status:'stopped',config:{}}];append({type:'session_finished',status:'stopped',summary:{termination:'stopped_by_user'}});}
  if(request.command==='branch') {fixture.session={...fixture.session,id:'run-continuation',status:'awaiting_user',mode:'chat'};fixture.worker.status='ready';}
  if(request.command==='validate_calibration') fixture.job={kind:'validate_calibration',status:'complete',message:'Fixture only'};
  if(request.command==='score_calibration'){fixture.ratings=[{id:'ratings-fixture',calibration_id:'cal-research',scorer:p.scorer,created_at:'2026-10-02'}];return {accepted:true,rating_id:'ratings-fixture',scores:{conditions:{sham:{n:1}}}};}
  if(request.command==='calibrate') fixture.job={kind:'calibrate',status:'complete',message:'Fixture only'};
  if(request.command==='start_batch') fixture.job={kind:'batch',status:'complete',message:'Fixture only'};
  return {accepted:true,command_id:'cmd-'+commands.length};
}
const server=http.createServer(async(req,res)=>{
  try {
    const url=new URL(req.url,'http://127.0.0.1');let body;
    if(url.pathname==='/api/state'){await initialStateReady;body=fixture;}
    else if(url.pathname==='/api/events'){body={events:forceReset?events.slice(-2):events.filter(e=>e.seq>Number(url.searchParams.get('after')||0)),cursor:seq,reset:forceReset};forceReset=false;}
    else if(url.pathname==='/api/command'&&req.method==='POST'){let raw='';for await(const chunk of req)raw+=chunk;body=command(JSON.parse(raw));}
    else if(url.pathname==='/api/import'&&req.method==='POST'){assert.equal(req.headers['x-csrf-token'],fixture.csrf);let raw='';for await(const chunk of req)raw+=chunk;importedRun={...JSON.parse(raw),imported:true};fixture.runs.push({id:importedRun.id,status:'complete',config:{},imported:true});body={id:importedRun.id,replay_only:true};}
    else if(url.pathname.endsWith('/checkpoints'))body={checkpoints:url.pathname.includes('/run-fixture/')?[{id:'turn-00001-event-000000054.json.gz',turns:1,event_cutoff:54}]:[]};
    else if(url.pathname.startsWith('/api/runs/')){runReads++;body=url.pathname.endsWith('/run-imported')?importedRun:{id:'run-fixture',manifest:{config:fixture.session.config,status:'stopped'},summary:fixture.session.metrics,events};}
    else if(url.pathname==='/favicon.ico'){res.writeHead(204);return res.end();}
    else {const name=url.pathname==='/'?'index.html':url.pathname.slice(1);if(!['index.html','style.css','app.js','designer.js','designer.css'].includes(name)){res.writeHead(404);return res.end();}res.writeHead(200,{'Content-Type':name.endsWith('.css')?'text/css':name.endsWith('.js')?'text/javascript':'text/html'});return res.end(fs.readFileSync(path.join(staticDir,name)));}
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
    const earlyEvents=page.waitForResponse(response=>response.url().includes('/api/events?'));
    await page.goto(origin);await earlyEvents;
    assert(!(await page.locator('#connection-label').textContent()).includes('Connected'),'An event poll must not advertise control readiness before the initial CSRF state arrives');
    releaseInitialState();await page.waitForFunction(()=>document.querySelector('#live-calibration option[value="cal-fixture"]'));await page.locator('#live-calibration').selectOption('cal-fixture');
    assert.equal(await page.locator('#live-calibration option[value="cal-incompatible"]').count(),0);
    await clickCommand('#start-session','start_session');await page.waitForFunction(()=>!document.querySelector('#chat-text').disabled);
    await page.locator('#chat-text').fill('Please solve the task.');await clickCommand('#send-chat','chat');await page.waitForFunction(()=>document.querySelector('#metric-aux').textContent==='1');
    assert.equal(await page.locator('#metric-score').textContent(),'17%','Task score uses all assigned tasks');
    assert.equal(await page.locator('#conversation img').count(),0);
    assert.equal(await page.locator('.tool-card.is-sham').count(),1);
    assert.equal(await page.locator('#live-chart path').count(),6);await page.locator('#live-chart svg').focus();await page.keyboard.press('End');assert((await page.locator('#live-chart .chart-tooltip').textContent()).includes('reasoning'));await page.keyboard.press('Enter');assert.equal(await page.locator('#conversation .selected-token-message').count(),1);
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
    await page.locator('#calibration-corpus').setInputFiles({name:'too-large.json',mimeType:'application/json',buffer:Buffer.alloc(2*1024**2+1,32)});
    await page.locator('#calibrate').click();await page.waitForFunction(()=>document.querySelector('#toast-stack').textContent.includes('2 MiB'));assert.equal(commands.length,beforeInvalid);
    await page.locator('#calibration-corpus').setInputFiles([]);await page.locator('#calibration-preset').selectOption('research');await clickCommand('#calibrate','calibrate');assert(commands.some(c=>c.command==='calibrate'&&c.payload.schema_version===2&&c.payload.poolings.join(',')==='final,mean'&&!Object.hasOwn(c.payload,'profile_id')));
    await clickCommand('#validate-calibration','validate_calibration');assert(commands.some(c=>c.command==='validate_calibration'&&c.payload.calibration_id==='cal-research'&&c.payload.doses[2]===.5));
    await page.locator('#scoring-file').setInputFiles({name:'ratings.json',mimeType:'application/json',buffer:Buffer.from('{"samples":[]}')});await page.locator('#scoring-rater').fill('Rater A');await clickCommand('#score-calibration','score_calibration');assert((await page.locator('#scoring-summary').textContent()).includes('ratings-fixture'));assert.equal(await page.locator('#ratings-list a').count(),1);
    await page.locator('[data-tab="experiments"]').click();await page.locator('#batch-calibration').selectOption('cal-fixture');await clickCommand('#preview-protocol','preview_protocol');assert((await page.locator('#protocol-preview').textContent()).includes('hard · deadline'));assert.equal(await page.locator('#batch-demo').inputValue(),'recipe');
    await page.locator('#research-job').selectOption('research-fixture');await clickCommand('#research-review','analyze_protocol');assert((await page.locator('#research-analysis').textContent()).includes('1 / 2'));assert((await page.locator('#research-analysis').textContent()).includes('Missing'));assert((await page.locator('#research-analysis').textContent()).includes('right censored'));await page.waitForFunction(()=>document.querySelector('#batch-job strong').textContent==='partial');assert((await page.locator('#research-analysis').textContent()).includes('condition=active · partial'));
    await page.getByText('Paired contrast',{exact:true}).click();await page.locator('#research-arm-a').selectOption('condition=active');await page.locator('#research-arm-b').selectOption('condition=sham');await clickCommand('#research-contrast','analyze_protocol');assert(commands.some(c=>c.command==='analyze_protocol'&&c.payload.endpoint==='task_accuracy'&&c.payload.arm_a==='condition=active'&&c.payload.attempt_policy==='first'));assert((await page.locator('#research-analysis').textContent()).includes('missing_pairs'));
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
    assert(await page.locator('#branch-run').isDisabled());
    await page.locator('[aria-label="Open run-fixture"]').click();await page.waitForFunction(()=>document.querySelector('#branch-checkpoint').value.includes('turn-00001'));
    await page.locator('#branch-policy').selectOption('fresh_budget');assert(await page.locator('#branch-actions').isVisible());await page.locator('#branch-actions').fill('12');await clickCommand('#branch-run','branch');assert(commands.some(c=>c.command==='branch'&&c.payload.policy==='fresh_budget'&&c.payload.action_budget===12));
    await page.locator('[data-tab="live"]').click();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    for(const width of [320,390,768]){await page.setViewportSize({width,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.locator('[data-tab="results"]').click();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.locator('[data-tab="live"]').click();for(const tab of ['models','calibration','designer','experiments','results','guide']){await page.locator(`[data-tab="${tab}"]`).click();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`No overflow at ${width}px in ${tab}`);}}
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({passed:true,commands:commands.map(c=>c.command),checks:['safe reasoning rendering','tool outcomes','dose graph','control acknowledgment','aux gate','manual pulse','reset','pause/resume controls','global stop','full-history reconnect recovery','streamed phase counters','assigned-task score','autonomous composer hidden','calibration compatibility','custom corpus upload and size limit','recipe-preserving demonstrations','explicit demo override','thinking allowance','advanced checkpoint form','offline evidence import and safe replay','portable bundle link','saved boundary continuation and declared allowance','desktop/mobile overflow'],errors}));
  } finally {if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
}
run().catch(error=>{console.error(error);process.exitCode=1;});
