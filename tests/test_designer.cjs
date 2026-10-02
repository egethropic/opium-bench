/* Standalone browser integration test. Uses real CPU-only recipe validation.
 * node tests/test_designer.cjs
 * Optional: PLAYWRIGHT_MODULE, BROWSER_CHANNEL, LAB_PYTHON.
 * Windows defaults to wsl.exe -e python3 for the repository's Python resolver.
 * No model, external service, GPU inference or GitHub Actions is involved.
 */
'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const http=require('node:http');
const path=require('node:path');
const {spawnSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const repo=path.resolve(__dirname,'..');
const linuxPath=value=>value.replace(/^([A-Za-z]):[\\/]/,(_,drive)=>'/mnt/'+drive.toLowerCase()+'/').replace(/\\/g,'/');
const pythonCode=`import sys,json
sys.path.insert(0,sys.argv[1])
from lab.session_v2 import build_session
from lab.tool_definitions import tool_preview
from lab.recipes_v2 import model_cost_notice
request=json.load(sys.stdin)
try:
 p=request['payload']
 preview=build_session(p['config'],p.get('mode','experiment'),'json',include_task=True)
 result={'accepted':True,'preview':preview,'config_json':json.dumps(preview['config'],ensure_ascii=True,indent=2)}
 if request['command']=='preview_session':
  result['command_id']='fixture-exact'
  preview['rendered_prompt']='EXACT FIXTURE TEMPLATE\\n'+json.dumps(preview['messages'],ensure_ascii=True)
  preview['model_fingerprint_sha256']='fixture-fingerprint'
  preview['prompt_sha256']='fixture-sha256'
 print(json.dumps(result,ensure_ascii=True))
except Exception as e:
 print(json.dumps({'error':str(e)}))`;
const commands=[];
function command(body){
  assert.ok(['preview_recipe','preview_session'].includes(body.command));commands.push(body);
  const windows=process.platform==='win32'&&!process.env.LAB_PYTHON;
  const proc=spawnSync(windows?'wsl.exe':process.env.LAB_PYTHON||'python3',[
    ...(windows?['-e','python3']:[]),'-c',pythonCode,windows?linuxPath(repo):repo
  ],{input:JSON.stringify(body),encoding:'utf8',maxBuffer:8*1024*1024});
  if(proc.status!==0)throw new Error(proc.stderr||String(proc.error));
  return JSON.parse(proc.stdout);
}
const html=`<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Designer fixture</title><link rel="stylesheet" href="/designer.css"><style>body{margin:0;padding:16px;background:#f5f7f6}#mount{max-width:1200px;margin:auto}</style></head><body><main id="mount"></main><script src="/designer.js"></script><script>
window.saved={recipe:[],effect_presets:[],auxiliary_tools:[]};window.used=[];window.saveCalls=[];
window.editor=OpiumDesigner.mount(document.getElementById('mount'),{
 command:async(command,payload)=>{const response=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,payload})});const result=await response.json();if(result.error)throw new Error(result.error);window.lastResponse=result;return result;},
 onUse:(config,options)=>{used.push({config,options});},isModelLoaded:()=>true,
 onSave:async value=>{saveCalls.push(value);saved[value.kind]=saved[value.kind].filter(e=>e.id!==value.id);saved[value.kind].push({id:value.id,value:value.value});},
 onLoad:async(kind,id)=>saved[kind].find(value=>value.id===id).value,
 listSaved:async kind=>saved[kind].map(value=>({id:value.id,label:value.id}))
});
</script></body></html>`;
const server=http.createServer(async(req,res)=>{
 try{
  if(req.url==='/'){res.writeHead(200,{'Content-Type':'text/html'});return res.end(html);}
  if(req.url==='/designer.js'||req.url==='/designer.css'){res.writeHead(200,{'Content-Type':req.url.endsWith('js')?'text/javascript':'text/css'});return res.end(fs.readFileSync(path.join(repo,'lab/static',req.url.slice(1))));}
  if(req.url==='/command'){let raw='';for await(const chunk of req)raw+=chunk;res.writeHead(200,{'Content-Type':'application/json'});return res.end(JSON.stringify(command(JSON.parse(raw))));}
  if(req.url==='/favicon.ico'){res.writeHead(204);return res.end();}res.writeHead(404);res.end();
 }catch(e){res.writeHead(500,{'Content-Type':'application/json'});res.end(JSON.stringify({error:e.message}));}
});
async function run(){
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const origin=`http://127.0.0.1:${server.address().port}`;let browser;
 try{
  browser=await chromium.launch({headless:true,...(process.env.BROWSER_CHANNEL?{channel:process.env.BROWSER_CHANNEL}:{})});
  const page=await browser.newPage({viewport:{width:1280,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/*',route=>route.request().url().startsWith(origin+'/')?route.continue():route.abort());
  const tab=name=>page.getByRole('navigation',{name:'Recipe designer sections'}).getByRole('button',{name,exact:true}).click();
  const click=name=>page.getByRole('button',{name,exact:true}).click();
  const status=()=>page.locator('.od-status').innerText();
  async function validated(){await click('Validate recipe');await page.waitForFunction(()=>document.querySelector('.od-badge').textContent==='Validated · v2'||!document.querySelector('.od-errors').hidden);assert.equal(await page.locator('.od-errors').isVisible(),false,await page.locator('.od-errors').innerText());}
  await page.goto(origin);assert.equal(await page.getByRole('heading',{name:'Design the intervention'}).count(),1);
  await validated();let config=await page.evaluate(()=>editor.getRecipe());assert.equal(config.recipe_version,2);assert.equal(config.effect_presets.length,5);
  await click('Use recipe');await page.waitForFunction(()=>used.length===1);assert.equal(await page.evaluate(()=>used[0].config.rng_policy),'derived');
  // Form changes preserve the complete canonical object and affect actual resolver output.
  await tab('Effects');await page.getByLabel('Joy gain',{exact:true}).fill('-0.5');await page.getByLabel('Joy attenuation',{exact:true}).fill('0.25');
  await page.getByLabel('Decay clock',{exact:true}).selectOption('decisions');await page.getByLabel('Half-life',{exact:true}).fill('3');await page.getByLabel('Cutoff / duration').fill('9');
  await page.getByLabel('Stacking channel',{exact:true}).fill('independent');await page.getByLabel('Stacking policy',{exact:true}).selectOption('capped_additive');
  await page.getByLabel('Prefill',{exact:true}).check();await page.getByLabel('Prefill positions').selectOption('all');await validated();
  config=await page.evaluate(()=>editor.getRecipe());assert.equal(config.effect_presets[0].gains.joy,-.5);assert.equal(config.effect_presets[0].attenuation.joy,.25);assert.equal(config.effect_presets[0].decay.clock,'decisions');assert.deepEqual(config.effect_presets[0].phases,['prefill','reasoning','output']);
  // Input validity never converts a blank or fractional integer into an admissible value.
  await page.getByLabel('Cutoff / duration').fill('1.5');await click('Validate recipe');assert.equal(await page.locator('.od-errors').isVisible(),true);await page.getByLabel('Cutoff / duration').fill('9');await validated();
  // Unknown fields are retained and rejected server-side; roundtrip does not erase them.
  await tab('Recipe JSON');config=await page.evaluate(()=>editor.getRecipe());config.accidental_unknown='keep me';
  await page.getByLabel('Recipe JSON',{exact:true}).fill(JSON.stringify(config));await click('Apply JSON and validate');await page.waitForFunction(()=>document.querySelector('.od-errors').textContent.includes('recipe-v2'));
  assert.equal(await page.evaluate(()=>editor.getRecipe().accidental_unknown),'keep me');delete config.accidental_unknown;
  await page.getByLabel('Recipe JSON',{exact:true}).fill(JSON.stringify(config));await validated();
  // Native schema validation and custom required demonstration arguments.
  await tab('Tools');await page.getByLabel('Function name',{exact:true}).fill('read_order');await click('Validate recipe');await page.waitForFunction(()=>!document.querySelector('.od-errors').hidden);assert.match(await page.locator('.od-errors').innerText(),/reserved|collid|task/i);
  await page.getByLabel('Function name',{exact:true}).fill('aux_operation');await page.getByLabel('Model-visible description').fill('Optional <img src=x onerror=alert(1)> operation.');
  await page.getByLabel('Argument JSON Schema').fill(JSON.stringify({type:'object',properties:{level:{type:'integer',minimum:1,maximum:3}},required:['level'],additionalProperties:false}));
  await click('Validate recipe');await page.waitForFunction(()=>!document.querySelector('.od-errors').hidden);assert.match(await page.locator('.od-errors').innerText(),/required|arguments|schema/i);
  await tab('Protocol');await page.getByLabel('Demonstration calls').fill(JSON.stringify([{tool:'aux_operation',arguments:{level:2}}]));await validated();
  await tab('Preview');assert.equal(await page.locator('.od-content img').count(),0);assert.ok((await page.locator('.od-content').innerText()).includes('<img src=x onerror=alert(1)>'));
  await click('Preview loaded model template');await page.waitForFunction(()=>document.querySelector('.od-exact'));assert.match(await page.locator('.od-exact').innerText(),/EXACT FIXTURE TEMPLATE/);
  // An exact result cannot be accepted after a draft edit or without a request.
  assert.equal(await page.evaluate(()=>editor.setExactPreview(lastResponse.preview)),false);
  await tab('Protocol');await page.getByLabel('Master seed',{exact:true}).fill('42');await validated();
  const changed=await page.evaluate(()=>editor.getRecipe());assert.notDeepEqual(config.rng_seeds,changed.rng_seeds);
  // A generated switch becomes a faithful explicit schedule; probability errors stay visible.
  await tab('Mappings');await page.getByLabel('Current condition').selectOption('joy_to_pain');await click('Copy generated schedule to explicit');await page.waitForFunction(()=>document.querySelector('[data-path="mapping_policy"]').value==='explicit');
  config=await page.evaluate(()=>editor.getRecipe());assert.equal(config.mapping_schedule.length,2);assert.equal(config.mapping_schedule[1].after_decisions,10);
  config.mapping_schedule[0].mappings.aux_operation=[{preset_id:'joy',probability:.75},{preset_id:'pain',probability:.25}];
  await page.getByLabel('Schedule JSON').fill(JSON.stringify(config.mapping_schedule));await validated();
  await page.getByLabel('Probability',{exact:true}).first().fill('0.8');await click('Validate recipe');await page.waitForFunction(()=>document.querySelector('.od-errors').textContent.includes('sum to one'));
  await page.getByLabel('Probability',{exact:true}).first().fill('0.75');await validated();
  // Reusable library callbacks carry detached validated definitions.
  await page.locator('.od-library summary').click();await page.getByLabel('Save identifier').fill('fixture-recipe');await click('Save validated item');await page.waitForFunction(()=>saveCalls.length===1);
  await click('Refresh saved items');await page.waitForFunction(()=>document.querySelector('[aria-label="Saved items"]').options.length===1&&document.querySelector('[aria-label="Saved items"]').value==='fixture-recipe');
  await tab('Protocol');await page.getByLabel('Recipe label',{exact:true}).fill('Changed draft');await click('Load selected item');await page.waitForFunction(()=>document.querySelector('[data-path="label"]').value!=='Changed draft');
  assert.equal(await page.evaluate(()=>editor.getRecipe().label),await page.evaluate(()=>saveCalls[0].value.label));
  await page.getByLabel('Library kind').selectOption('effect_presets');await page.getByLabel('Save identifier').fill('fixture-effects');await click('Save validated item');await page.waitForFunction(()=>saveCalls.length===2);assert.equal(await page.evaluate(()=>Array.isArray(saveCalls[1].value)),true);
  await page.getByLabel('Library kind').selectOption('auxiliary_tools');await page.getByLabel('Save identifier').fill('fixture-tools');await click('Save validated item');await page.waitForFunction(()=>saveCalls.length===3);assert.equal(await page.evaluate(()=>saveCalls[2].value[0].name),'aux_operation');
  // File exports retain Python's full-width derived seeds; imports revalidate definitions.
  await page.getByLabel('Library kind').selectOption('recipe');
  const downloaded=page.waitForEvent('download');await click('Export validated item');const download=await downloaded;
  const stream=await download.createReadStream();let raw='';for await(const chunk of stream)raw+=chunk;
  assert.equal(raw,await page.evaluate(()=>lastResponse.config_json+'\n'));
  await page.getByLabel('Import definition JSON').setInputFiles({name:'recipe.json',mimeType:'application/json',buffer:Buffer.from(raw)});
  await click('Import and validate');await page.waitForFunction(()=>document.querySelector('.od-status').textContent.startsWith('Imported and validated'));
  const stableDraft=await page.evaluate(()=>editor.getRecipe());
  await page.getByLabel('Import definition JSON').setInputFiles({name:'bad.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({...stableDraft,unknown_field:true}))});
  await click('Import and validate');await page.waitForFunction(()=>!document.querySelector('.od-errors').hidden);assert.deepEqual(await page.evaluate(()=>editor.getRecipe()),stableDraft);
  // Hidden tool is still mapped but absent from the actual model context.
  await page.evaluate(()=>{const c=editor.getRecipe();c.demonstration='none';c.demonstration_calls=[];c.auxiliary_tools[0].visible=false;editor.setRecipe(c);});await validated();
  assert.equal(await page.evaluate(()=>lastResponse.preview.tools.some(t=>t.function.name==='aux_operation')),false);assert.equal(await page.evaluate(()=>editor.getRecipe().auxiliary_tools[0].visible),false);
  // Malformed JSON is never replaced on validation or navigation.
  await tab('Recipe JSON');await page.getByLabel('Recipe JSON',{exact:true}).fill('{bad');await click('Validate recipe');assert.match(await page.locator('.od-errors').innerText(),/Recipe JSON/);assert.equal(await page.getByLabel('Recipe JSON',{exact:true}).inputValue(),'{bad');
  await page.evaluate(()=>editor.setRecipe(OpiumDesigner.defaultRecipe()));await validated();
  // Unsafe explicit seeds fail before transmission instead of silently rounding.
  await page.evaluate(()=>{const c=editor.getRecipe();c.rng_policy='explicit';editor.setRecipe(c);});const before=commands.length;await click('Validate recipe');assert.match(await page.locator('.od-errors').innerText(),/safe integer/);assert.equal(commands.length,before);
  await page.evaluate(()=>editor.setRecipe(OpiumDesigner.defaultRecipe()));await validated();
  // All panels fit phone, tablet and desktop widths, including long JSON and schemas.
  for(const width of [320,390,768,1280]){await page.setViewportSize({width,height:1000});for(const name of ['Protocol','Effects','Tools','Mappings','Preview','Recipe JSON']){await tab(name);const dims=await page.evaluate(()=>({screen:innerWidth,doc:document.documentElement.scrollWidth}));assert.ok(dims.doc<=dims.screen+1,`${name} overflow at ${width}: ${dims.doc}`);}}
  const unlabeled=await page.locator('input:not([type=hidden]),select,textarea').evaluateAll(nodes=>nodes.filter(n=>!n.getAttribute('aria-label')&&!n.labels?.length).map(n=>n.outerHTML));assert.deepEqual(unlabeled,[]);
  if(process.env.DESIGNER_SCREENSHOT){await tab('Effects');await page.screenshot({path:process.env.DESIGNER_SCREENSHOT,fullPage:true});}
  // Worker events may precede the command acknowledgment. Accept that result once;
  // reject a late result for a changed draft and do not overwrite a previous edit.
  await page.evaluate(()=>{
    const host=document.createElement('div');host.id='race-mount';document.body.append(host);
    const seed=structuredClone(lastResponse.preview);window.queueMode='early';
    window.raceEditor=OpiumDesigner.mount(host,{command:async(name,payload)=>{
      if(name==='preview_recipe')return {accepted:true,preview:{...structuredClone(seed),config:payload.config}};
      const result={...structuredClone(seed),config:payload.config,rendered_prompt:'Early worker result'};
      if(queueMode==='early'){window.earlyAccepted=raceEditor.setExactPreview(result);return {accepted:true,command_id:'early'};}
      window.delayedResult=result;return {accepted:true,command_id:'late'};
    },initialRecipe:seed.config});
  });
  const race=page.locator('#race-mount');await race.getByRole('button',{name:'Preview',exact:true}).click();await race.getByRole('button',{name:'Preview loaded model template',exact:true}).click();await page.waitForFunction(()=>window.earlyAccepted===true);
  assert.equal(await race.locator('.od-exact').innerText(),'Early worker result');
  await page.evaluate(()=>{queueMode='late';});await race.getByRole('button',{name:'Preview loaded model template',exact:true}).click();await page.waitForFunction(()=>!!window.delayedResult);
  assert.equal(await page.evaluate(()=>{const c=raceEditor.getRecipe();c.label='Edited after request';raceEditor.setRecipe(c);return raceEditor.setExactPreview(delayedResult);}),false);
  await page.evaluate(()=>raceEditor.destroy());assert.equal(await race.locator('.opium-designer').count(),0);
  assert.deepEqual(errors,[]);assert.ok(commands.every(value=>['preview_recipe','preview_session'].includes(value.command)));console.log(JSON.stringify({passed:true,realResolverCalls:commands.length,widths:[320,390,768,1280],asyncPreviewRaces:true,browserErrors:errors}));
 }finally{if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
}
run().catch(error=>{console.error(error);process.exitCode=1;});
