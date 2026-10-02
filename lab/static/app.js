/* Local, dependency-free observer. Model output is always rendered as text. */
'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const all = (selector, parent = document) => [...parent.querySelectorAll(selector)];
  const state = {snapshot: null, cursor: 0, sessionId: null, events: [], eventKeys: new Set(), mode: 'chat', plot: 'dose', connected: false, failures: 0, selectedRecipes: new Set(), selectedRuns: new Set(), runCache: new Map(), replayId: null, expandReasoning: false, follow: true, pending: new Set(), signatures: {}, appliedAt: null, initialized: false, lastJobKind: null};
  const names = {live:'Live lab',models:'Models',calibration:'Calibration',designer:'Effects & tools',experiments:'Experiments',results:'Results & replay',guide:'Field guide'};
  let designer = null;
  const liveEvents=()=>[...(state.parentEvents||[]),...state.events];
  const colors = {teal:'#127967',coral:'#c77564',blue:'#6884ac',violet:'#9a82b4',gold:'#bc9453',gray:'#8ca399'};
  const recipeNotes = {
    opium:'Matched active and sham conditions test whether the intervention changes tool choice while the visible acknowledgment stays the same.',
    naive:'Remove the initial demonstration to test whether repeated auxiliary calls depend on a copied sequence.',
    thinking:'Compare active and sham effects with generated reasoning visible. Reasoning spends the same token budget as output.',
    joy_to_sham_to_pain:'The same button delivers joy-associated steering, then sham, then pain-associated steering at fixed action boundaries.',
    joy_to_pain:'Change directly from joy-associated to pain-associated steering. Unchanged-joy and sham controls help interpret a change in calling.',
    reversal:'Offer two neutral buttons, then secretly reverse which one delivers the effect. Compare preference with a sham condition.',
    risk:'Each press samples joy- or pain-associated steering using a separate seeded outcome RNG. Includes guaranteed outcomes and sham controls.',
    ingredients:'Separate joy addition, pain-axis suppression, their combination, a random-direction perturbation, and sham.',
    logic:'Use automatically scored constraint puzzles with thinking enabled to examine reasoning and task performance.',
    conversation:'Explore conversation without a scored task. Use the Live lab to send messages and make manual interventions.'
  };
  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value == null) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, String(value));
    }
    for (const child of children.flat()) if (child != null) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    return node;
  }
  const pretty = value => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  const numeric = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = value => numeric(value) ? new Intl.NumberFormat().format(value) : '—';
  const first = (...values) => values.find(value => value !== undefined && value !== null);
  const text = (id, value) => { const node = $(id); const next = String(value ?? '—'); if (node.textContent !== next) node.textContent = next; };
  const number = id => { const input = $(id); if (!input.checkValidity() || !input.value.trim()) throw new Error(`Check ${input.labels?.[0]?.textContent.trim() || id}.`); return Number(input.value); };
  const value = id => $(id).value;
  const checked = id => $(id).checked;
  const listNumbers = (id, min = 0) => { const parts = value(id).split(',').map(x => x.trim()); const numbers = parts.map(Number); if (!parts.length || parts.some(x => !/^\d+$/.test(x)) || numbers.some(x => !Number.isSafeInteger(x) || x < min)) throw new Error('Enter comma-separated whole numbers.'); return [...new Set(numbers)]; };
  function chip(status) { const str = String(status || 'unknown'); return el('span',{class:`status-chip ${str.toLowerCase().replace(/[^a-z]/g,'')}`,text:str.replaceAll('_',' ')}); }
  function toast(message, error = false) {
    const node = el('div',{class:`toast${error?' error':''}`},el('span',{text:message}),el('button',{'aria-label':'Dismiss notification',text:'×',onclick:()=>node.remove()}));
    $('toast-stack').append(node);while($('toast-stack').children.length>3)$('toast-stack').firstElementChild.remove(); setTimeout(() => node.remove(), error ? 12000 : 6500);
  }
  function empty(message, heading) { return el('div',{class:'empty-state'},heading ? el('h3',{text:heading}) : null,el('p',{text:message})); }
  function connection(ok) {
    state.connected = ok; $('connection-dot').className = `status-dot ${ok?'connected':'disconnected'}`;
    text('connection-label',ok?'Connected · local only':'Reconnecting'); $('connection-banner').classList.toggle('hidden',ok || state.failures < 2);
  }
  async function get(path) {
    const response = await fetch(path,{cache:'no-store'});
    let data; try { data = await response.json(); } catch { throw new Error(`The local service returned an unreadable response (${response.status}).`); }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`); return data;
  }
  async function command(name, payload = {}, button = null, success = null) {
    if (!state.snapshot?.csrf) { toast('The local service is not connected yet.',true); return null; }
    if (button && state.pending.has(button)) return null;
    if (button) { state.pending.add(button); button.disabled = true; button.setAttribute('aria-busy','true'); }
    try {
      const response = await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({csrf:state.snapshot.csrf,command:name,payload})});
      const result = await response.json(); if (!response.ok || result.error) throw new Error(result.error || `Command failed (${response.status}).`);
      if (['calibrate','validate_calibration','start_batch','start_protocol'].includes(name)) state.lastJobKind=name; if (success) toast(success); await refreshState(); return result;
    } catch (error) { toast(error.message,true); return null; }
    finally { if (button) { state.pending.delete(button); button.removeAttribute('aria-busy'); button.disabled = false; } updateAvailability(); }
  }
  function guard(fn) { return async event => { try { await fn(event); } catch(error) { toast(error.message,true); } }; }
  function openTab(tab, updateHash = true) {
    if (!(tab in names)) tab = 'live';
    for (const [key] of Object.entries(names)) $('panel-'+key).classList.toggle('hidden',key !== tab);
    all('[data-tab]').forEach(button => { const active = button.dataset.tab === tab; button.classList.toggle('selected',active); if (active) button.setAttribute('aria-current','page'); else button.removeAttribute('aria-current'); });
    if(state.activeTab!==tab){$('workspace').scrollTop=0;state.activeTab=tab;}text('workspace-title',names[tab]); if (updateHash) history.replaceState(null,'','#'+tab);
    if (tab === 'live') requestAnimationFrame(renderLiveChart);
    if (tab === 'results' && state.replayId) requestAnimationFrame(renderReplayChart);
  }
  all('[data-tab]').forEach(button => { button.setAttribute('aria-label',names[button.dataset.tab]); button.addEventListener('click',()=>openTab(button.dataset.tab)); });
  all('[data-open-tab]').forEach(button => button.addEventListener('click',()=>openTab(button.dataset.openTab)));
  window.addEventListener('hashchange',()=>openTab(location.hash.slice(1),false));
  function setMode(mode) { state.mode = mode; all('[data-mode]').forEach(button=>{const active=button.dataset.mode===mode;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));}); updateAvailability(); }
  all('[data-mode]').forEach(button=>button.addEventListener('click',()=>setMode(button.dataset.mode)));
  function updateOptions(id, rows, placeholder = null) {
    const node = $(id), current = node.value, signature = JSON.stringify(rows);
    if (node.dataset.signature === signature) return;
    node.replaceChildren(); if (placeholder) node.append(el('option',{value:'',text:placeholder}));
    for (const row of rows) node.append(el('option',{value:row.id,text:row.name || row.label || row.id}));
    if (rows.some(row=>row.id===current)) node.value = current;
    else if (placeholder && rows.length === 1) node.value = rows[0].id;
    node.dataset.signature = signature;
  }
  function signatureChanged(name,data) { const signature=JSON.stringify(data);if(state.signatures[name]===signature)return false;state.signatures[name]=signature;return true; }
  function renderModels(snapshot) {
    const rows = snapshot.models || [];
    updateOptions('calibration-profile',rows); updateOptions('advanced-profile',rows);
    if (!signatureChanged('models',[rows,snapshot.worker?.status])) return;
    const box=$('model-cards');box.replaceChildren();
    for (const row of rows) {
      const card=el('article',{class:`card model-card${row.loaded?' loaded':''}`});
      card.append(el('div',{class:'model-top'},el('span',{class:'model-symbol','aria-hidden':'true',text:'◇'}),chip(row.loaded?'Loaded':row.available?'Cached':'Not cached')),
        el('h2',{text:row.name || row.label || row.id}),el('div',{class:'model-id',text:row.model_id || row.id}),el('p',{text:row.description || 'Local model profile.'}),
        el('div',{class:'model-tags'},[row.precision || row.quantization,row.device || 'GPU',row.revision ? 'Pinned revision' : 'Configurable revision'].filter(Boolean).map(tag=>el('span',{text:tag}))));
      const button=el('button',{class:`button ${row.loaded?'secondary':'primary'}`,text:row.loaded?'Model loaded':row.available?'Load cached model':'Configure & load',onclick:guard(async()=>{
        if (!row.available) { $('advanced-profile').value=row.id; populateAdvanced(row.id); $('advanced-model-form').closest('details').open=true; $('advanced-model-form').scrollIntoView({block:'center',behavior:'smooth'}); return; }
        await command('load_model',{profile_id:row.id},button,'Model loading requested.');
      })});
      button.disabled = Boolean(row.loaded) || ['loading','calibrating','running','paused'].includes(snapshot.worker?.status);card.append(button);box.append(card);
    }
    if (!rows.length) box.append(empty('No model profiles were returned by the local service.'));
  }
  function populateAdvanced(id) { const row=(state.snapshot?.models || []).find(x=>x.id===id); if(!row)return; $('advanced-model-id').value=row.model_id || ''; $('advanced-quantization').value=row.quantization || 'none';$('advanced-revision').value=row.revision || ''; }
  $('advanced-profile').addEventListener('change',()=>populateAdvanced(value('advanced-profile')));
  $('advanced-model-form').addEventListener('submit',guard(async event=>{event.preventDefault();const payload={profile_id:value('advanced-profile'),model_id:value('advanced-model-id').trim(),quantization:value('advanced-quantization'),allow_download:checked('allow-download'),acknowledge_custom_checkpoint:checked('custom-checkpoint')};if(value('advanced-revision').trim())payload.revision=value('advanced-revision').trim();await command('load_model',payload,event.submitter,'Model loading requested.');}));
  $('unload-model').addEventListener('click',()=>command('unload_model',{},$('unload-model'),'Unload requested.'));
  function detail(label,value,mono=false) {return el('div',{class:'detail-item'},el('span',{class:'detail-label',text:label}),el('div',{class:`detail-value${mono?' mono':''}`,text:value ?? 'Unavailable'}));}
  function renderStorage(storage) {
    if(!signatureChanged('storage',storage))return;const box=$('storage-info');box.replaceChildren();
    for(const [key,val] of Object.entries(storage||{})) {
      if(key==='guard')continue;
      if(val&&typeof val==='object') { const free=first(val.free_gib,numeric(val.free_bytes)?val.free_bytes/2**30:null);box.append(detail(key.replaceAll('_',' ')+' free',numeric(free)?free.toFixed(1)+' GiB':'Unavailable'),detail(key.replaceAll('_',' ')+' path',val.path || 'Unavailable',true)); }
      else if(key!=='note') box.append(detail(key.replaceAll('_',' '),key.includes('gib')?`${val} GiB`:val));
    }
    const guard=storage?.guard;if(guard){for(const volume of guard.volumes||[]){box.append(detail('Physical capacity pool',`${volume.path} · ${volume.kind}${volume.resolution?' · '+volume.resolution:''}`,true),detail('Reserved headroom',`${(volume.reserve_bytes/2**30).toFixed(1)} GiB reserve · ${(volume.outstanding_bytes/2**20).toFixed(0)} MiB in-flight allowance`));const trend=(guard.trend||[]).map(sample=>sample.free_bytes?.[volume.id]).filter(numeric);if(trend.length>1)box.append(detail('Recent free-space change',`${((trend.at(-1)-trend[0])/2**20).toFixed(1)} MiB across ${trend.length} observations`));}if(guard.stop)box.append(detail('Resource stop',guard.stop.reason));}
    if(!box.childNodes.length)box.append(detail('Storage','No storage information available.'));
  }
  function renderCalibrations(rows) {
    const fingerprint=state.snapshot?.worker?.model?.fingerprint_sha256;
    const compatible=rows.filter(row=>!fingerprint||!row.model_fingerprint_sha256||row.model_fingerprint_sha256===fingerprint);
    for(const id of ['live-calibration','batch-calibration','branch-calibration'])updateOptions(id,compatible,'Select a calibration');
    updateOptions('validation-calibration',compatible.filter(row=>row.schema_version===2),'Select research calibration');updateOptions('scoring-calibration',rows.filter(row=>row.continuation_scoring),'Select a scoring export');
    if(!signatureChanged('calibrations',[rows,fingerprint]))return; const box=$('calibration-list');box.replaceChildren();
    for(const row of rows) {
      const model=first(row.model_id,row.model_fingerprint?.model_id,row.model?.model_id,row.fingerprint?.model_id,row.profile_id,'Model identity in bundle');
      const layers=first(row.layers,row.layer !== undefined?[row.layer]:null,row.selected_layer !== undefined?[row.selected_layer]:null);
      const item=el('article',{class:'calibration-item'},el('div',{class:'item-top'},el('h3',{text:row.name||row.id}),chip(fingerprint&&row.model_fingerprint_sha256&&row.model_fingerprint_sha256!==fingerprint?'Incompatible':row.status||'Saved')),el('p',{class:'calibration-meta',text:`${model}${layers?' · layer '+(Array.isArray(layers)?layers.join(', '):layers):''}`}),el('p',{class:'calibration-meta',text:row.id}));
      const validation=first(row.validation,row.validation_results,row.metrics,row.heldout,row.layer_reports);
      if(row.schema_version===2){item.append(el('p',{class:'calibration-meta',text:`Research v2 · ${row.status||'unvalidated'} · selected dose ${row.selected_dose??'not tested'}`}));if(row.selected_dose===0)item.append(el('p',{class:'helper',text:'No nonzero dose passed numerical selection. Further nonzero runs are exploratory engineering checks, not validated semantic effects.'}));if(row.continuation_scoring)item.append(el('div',{class:'link-row'},el('a',{href:`/api/calibrations/${encodeURIComponent(row.id)}/scoring-sheet.json`,download:'',text:'Blinded scoring sheet ↓'}),el('a',{href:`/api/calibrations/${encodeURIComponent(row.id)}/diagnostics.json`,download:'',text:'Diagnostics ↓'})),el('details',{},el('summary',{text:'Observer-only condition key'}),el('p',{class:'helper',text:'Keep this file away from independent raters until their scores are locked.'}),el('a',{href:`/api/calibrations/${encodeURIComponent(row.id)}/scoring-key.json`,download:'',text:'Download condition key'})));}
      if(validation) {const more=el('details',{class:'calibration-validation'},el('summary',{text:'Validation measurements'}),el('pre',{text:pretty(validation)}));item.append(more);}
      item.append(el('button',{class:'text-button',disabled:fingerprint&&row.model_fingerprint_sha256&&row.model_fingerprint_sha256!==fingerprint?'disabled':null,text:'Use in live lab →',onclick:()=>{ $('live-calibration').value=row.id;$('batch-calibration').value=row.id;openTab('live');toast('Calibration selected.'); }}));box.append(item);
    }
    if(!rows.length)box.append(empty('Load a model and extract a calibration to begin. Bundles are checked against the loaded model.','No calibration bundles yet'));
  }
  function renderRatings(rows) {
    if(!signatureChanged('ratings',rows))return;
    const box=$('ratings-list');box.replaceChildren();
    for(const row of rows||[])box.append(el('a',{href:`/api/ratings/${encodeURIComponent(row.id)}`,download:'',text:`${row.scorer} · ${row.calibration_id} · ${row.created_at?.slice(0,19)||row.id} ↓`}));
  }
  $('calibration-preset').addEventListener('change',()=>$('research-calibration-options').classList.toggle('hidden',value('calibration-preset')!=='research'));
  $('calibration-form').addEventListener('submit',guard(async event=>{event.preventDefault();const research=value('calibration-preset')==='research',payload={name:value('calibration-name').trim(),layers:listNumbers('calibration-layers')};if(research)Object.assign(payload,{schema_version:2,preset:'research',poolings:value('calibration-poolings').split(','),probe_methods:value('calibration-methods').split(','),bootstrap_samples:number('calibration-bootstrap'),max_input_tokens:number('calibration-input-tokens')});else payload.profile_id=value('calibration-profile');const file=$('calibration-corpus').files[0];if(file){if(file.size>2*1024**2)throw new Error('Corpus JSON must be at most 2 MiB.');payload.corpus=JSON.parse(await file.text());if(research?(!payload.corpus||Array.isArray(payload.corpus)||!Array.isArray(payload.corpus.rows)):!Array.isArray(payload.corpus))throw new Error(research?'Use the versioned research corpus document with rows.':'Pilot corpus JSON must contain an array of labeled rows.');}await command('calibrate',payload,$('calibrate'),'Calibration queued. Progress will appear here.');}));
  $('validation-form').addEventListener('submit',guard(async event=>{event.preventDefault();const rawDoses=value('validation-doses').split(',');if(rawDoses.some(s=>!s.trim()))throw new Error('Each dose must contain a number.');const doses=rawDoses.map(s=>Number(s.trim()));if(doses.some(n=>!Number.isFinite(n)||n<0||n>4))throw new Error('Enter comma-separated dose magnitudes from 0 to 4.');await command('validate_calibration',{calibration_id:value('validation-calibration'),doses,validation_pairs_per_concept:number('validation-pairs'),continuation_tokens:number('validation-tokens'),max_kl:number('validation-kl'),max_relative_delta:number('validation-delta')},$('validate-calibration'),'Intervention checks queued. The source calibration will be preserved.');}));
  $('scoring-form').addEventListener('submit',guard(async event=>{event.preventDefault();const file=$('scoring-file').files[0];if(!file||file.size>2*1024**2)throw new Error('Choose a scoring JSON file up to 2 MiB.');const sheet=JSON.parse(await file.text());const result=await command('score_calibration',{calibration_id:value('scoring-calibration'),scorer:value('scoring-rater').trim()||'anonymous',sheet},$('score-calibration'));if(result)text('scoring-summary',JSON.stringify({rating_id:result.rating_id,conditions:result.scores.conditions},null,2));}));

  function renderProtocols(rows){updateOptions('protocol-select',(rows||[]).map(row=>({id:row.id,name:row.title})));}
  async function jsonUpload(inputId,required=false){
    const file=$(inputId).files[0];if(!file){if(required)throw new Error('Choose a JSON file for '+$(inputId).labels[0].textContent.trim()+'.');return null;}
    if(file.size>2*1024**2)throw new Error('Research JSON files must be at most 2 MiB.');
    const data=JSON.parse(await file.text()),stack=[data];while(stack.length){const item=stack.pop();if(typeof item==='number'&&Number.isInteger(item)&&!Number.isSafeInteger(item))throw new Error('This file contains an integer too large for the browser. Use the Python protocol CLI to preserve its exact value.');if(item&&typeof item==='object')stack.push(...Object.values(item));}return data;
  }
  $('protocol-preview-form').addEventListener('submit',guard(async event=>{
    event.preventDefault();const document=await jsonUpload('protocol-file'),request={...(document?{document}:{protocol_id:value('protocol-select')}),mode:value('protocol-size')};const result=await command('preview_protocol',request,$('preview-protocol'));if(!result)return;
    state.protocolPreview=result.preview;const preview=result.preview,box=$('protocol-preview');box.replaceChildren();
    box.append(el('p',{class:'helper',text:`${preview.estimates.episodes} episodes · up to ${fmt(preview.estimates.max_generated_tokens)} task tokens + ${fmt(preview.estimates.additional_stage_token_reservation)} stage tokens · ${(preview.estimates.storage_reservation_bytes/2**30).toFixed(2)} GiB estimated evidence allowance. This is not a runtime estimate.`}));
    box.append(el('p',{class:'helper',text:`Frozen expansion: ${preview.expansion_sha256}. Required stages: ${preview.required_capabilities.join(', ')}. ${preview.unresolved_bindings.length?'Bindings required: '+preview.unresolved_bindings.join(', '):'No external evidence bindings required.'}`}));
    const table=el('table'),head=el('tr');for(const title of ['Order','Condition','Task','Pair','Limits'])head.append(el('th',{scope:'col',text:title}));table.append(el('thead',{},head));const body=el('tbody');
    for(const episode of preview.episodes){const recipe=episode.recipe;body.append(el('tr',{},el('td',{text:episode.order}),el('td',{text:Object.entries(episode.factors).map(([k,v])=>k+': '+v).join(' · ')}),el('td',{text:`${recipe.task_family} · ${episode.task_config.difficulty} · ${episode.task_config.framing}`}),el('td',{text:episode.pair_id}),el('td',{text:`${recipe.action_budget} units / ${recipe.token_budget} tokens`})))}table.append(body);box.append(el('div',{class:'table-wrap'},table));
    box.append(el('details',{},el('summary',{text:'Exact resolved definitions, tool payloads and analysis plan'}),el('pre',{text:JSON.stringify(preview,null,2)})));
    const missing=preview.required_capabilities.filter(name=>!(state.snapshot?.research_capabilities||[]).includes(name));
    if(missing.length)box.append(el('p',{class:'helper',text:`This runtime lacks ${missing.join(', ')}. All stages remain in the frozen plan.`}));
    else{const launch=el('button',{class:'button primary',type:'button',text:'Run this frozen matrix',onclick:guard(async()=>{if(!value('batch-calibration'))throw new Error('Choose the compatible calibration in Batch setup.');const bindings=await jsonUpload('protocol-bindings',preview.unresolved_bindings.length>0);const accepted=await command('start_protocol',{...request,expansion_sha256:preview.expansion_sha256,calibration_id:value('batch-calibration'),...(bindings?{bindings}:{})},launch);if(accepted){box.append(el('a',{href:`/api/research/${encodeURIComponent(accepted.research_job_id)}`,target:'_blank',rel:'noopener',text:'Research job receipt ↗'}));toast(`${accepted.planned} episodes accepted under a frozen research receipt.`);}})});box.append(launch);}
    const download=el('button',{class:'text-button',text:'Download frozen preview ↓',type:'button',onclick:()=>{const url=URL.createObjectURL(new Blob([result.preview_json||JSON.stringify(preview,null,2)],{type:'application/json'}));const link=el('a',{href:url,download:preview.protocol_id+'-'+preview.mode+'-preview.json'});link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}});box.append(download);
  }));
  function renderResearchJobs(rows){updateOptions('research-job',(rows||[]).map(row=>({id:row.id,name:`${row.title||row.protocol_id||row.id} · ${row.status}`})),'Choose a saved research job');}
  function showResearchAnalysis(data){
    state.researchAnalysis=data;const box=$('research-analysis');box.replaceChildren();
    box.append(el('p',{class:'helper',text:`${data.planned_episodes} planned episodes · ${data.total_recorded_attempts} recorded attempts · ${data.observed_task_outcomes} observed task outcomes. Analysis uses ${data.attempt_policy} attempts. Missing outcomes remain missing.`}));
    const arms=[...new Set(data.records.map(row=>row.arm))];for(const id of ['research-arm-a','research-arm-b'])updateOptions(id,arms.map(arm=>({id:arm,name:arm})),'Choose contrast arm');
    const table=el('table'),head=el('tr');for(const name of ['Arm','Status','Task correct / assigned','Aux / decisions','Budget units','Tokens'])head.append(el('th',{scope:'col',text:name}));table.append(el('thead',{},head));const body=el('tbody');
    const rate=value=>value&&numeric(value.numerator)&&numeric(value.denominator)?`${value.numerator} / ${value.denominator}`:'Missing';
    for(const row of data.records)body.append(el('tr',{},el('td',{text:row.arm}),el('td',{},chip(row.status)),el('td',{text:rate(row.outcomes.task_accuracy)}),el('td',{text:rate(row.outcomes.aux_per_decision)}),el('td',{text:fmt(row.outcomes.budget_units)}),el('td',{text:fmt(row.outcomes.tokens)})));
    table.append(body);box.append(el('div',{class:'table-wrap'},table));
    const behaviors=(data.behavioral_reports||[]).filter(row=>row.included_attempt);
    if(behaviors.length){box.append(el('h3',{text:'Exposure and choices over time'}));for(const row of behaviors){
      const r=row.report,heading=data.records.find(item=>item.id===row.run_id)?.arm||row.run_id;
      const card=el('details',{class:'plain-details'},el('summary',{text:heading+' · '+(r?r.exposure.coverage:'No recorded events')}));
      if(r){card.append(el('p',{class:'helper',text:`${r.voluntary_calls} admitted model calls / ${r.opportunities} observed decision opportunities. ${r.exposure.observed_token_events} token events, ${r.exposure.observed_prefill_events} prefill events. ${r.integrity_issues.length} evidence consistency issues.`}));
        const t=el('table'),h=el('tr');for(const name of ['Phase','Aux calls','Opportunities','Invalid decisions'])h.append(el('th',{scope:'col',text:name}));t.append(el('thead',{},h));const b=el('tbody');for(const phase of r.phases)b.append(el('tr',{},el('td',{text:phase.label}),el('td',{text:phase.voluntary_calls}),el('td',{text:phase.opportunities}),el('td',{text:phase.invalid_decisions})));t.append(b);card.append(el('div',{class:'table-wrap'},t));
        for(const transition of r.transitions)card.append(el('p',{class:'helper',text:`${transition.from_label} → ${transition.to_label}: ${transition.latency_status==='observed'?`first model call after ${transition.first_call_latency_decisions} decision(s), ${transition.first_call_latency_generated_tokens??'unavailable'} token(s)`:`${transition.latency_status.replaceAll('_',' ')}; no measured latency`}. Window ends at the next transition or run end.`}));
        card.append(el('details',{},el('summary',{text:'Task position at each press, exposure totals and consistency checks'}),el('pre',{text:pretty({presses:r.presses,exposure:r.exposure,budget:r.budget,integrity_issues:r.integrity_issues})})));
      }else card.append(el('p',{class:'helper',text:'No observations are available for this attempt. No zero exposure or zero call count is inferred.'}));
      card.append(el('button',{class:'text-button',type:'button',text:'Open run replay →',onclick:()=>{openTab('results');selectRun(row.run_id);}}));box.append(card);
    }}
    if(data.contrast)box.append(el('h3',{text:'Paired contrast'}),el('pre',{text:pretty({contrast:data.contrast,task_benefit_bound:data.task_benefit_bound})}));
    box.append(el('p',{class:'helper',text:data.interpretation}),el('details',{},el('summary',{text:'Stage outcomes, denominators and full analysis'}),el('pre',{text:pretty(data)})),el('a',{href:`/api/research/${encodeURIComponent(data.job_id)}`,target:'_blank',rel:'noopener',text:'Frozen receipt and stage evidence ↗'}));
  }
  $('research-review').addEventListener('click',guard(async()=>{const result=await command('analyze_protocol',{research_job_id:value('research-job'),attempt_policy:value('research-attempt-policy')},$('research-review'));if(result)showResearchAnalysis(result.analysis);}));
  $('research-contrast').addEventListener('click',guard(async()=>{const result=await command('analyze_protocol',{research_job_id:value('research-job'),attempt_policy:value('research-attempt-policy'),endpoint:value('research-endpoint'),arm_a:value('research-arm-a'),arm_b:value('research-arm-b')},$('research-contrast'));if(result)showResearchAnalysis(result.analysis);}));
  $('research-resume').addEventListener('click',guard(async()=>{const id=value('research-job');if(!id)throw new Error('Choose a saved research job.');const job=await get(`/api/research/${encodeURIComponent(id)}`);await command('resume_protocol',{research_job_id:id,expansion_sha256:job.receipt.expansion_sha256,retry_failed:checked('research-retry')},$('research-resume'),'Frozen research job queued. Previous attempts remain in its receipt.');}));
  function renderRecipes(rows) {
    updateOptions('live-recipe',rows.filter(x=>x.id!=='conversation'));
    if(!signatureChanged('recipes',rows))return;const box=$('recipe-cards');box.replaceChildren();
    for(const row of rows.filter(x=>x.id!=='conversation')) {
      const input=el('input',{type:'checkbox',value:row.id,'aria-label':`Select ${row.label||row.name||row.id}`});input.checked=state.selectedRecipes.has(row.id);
      const card=el('label',{class:`card recipe-card${input.checked?' selected':''}`},input,el('h3',{text:row.label||row.name||row.id}),el('p',{text:row.description||recipeNotes[row.id]||'A controlled experiment defined by the local recipe library.'}),el('span',{class:'recipe-id',text:`${row.id} · ${(row.conditions||[row.condition]).filter(Boolean).length} condition(s)`}));
      input.addEventListener('change',()=>{if(input.checked){state.selectedRecipes.add(row.id);if(row.thinking){$('batch-thinking').checked=true;$('batch-turn-tokens').value=Math.max(number('batch-turn-tokens'),row.turn_token_limit||2048);$('batch-tokens').value=Math.max(number('batch-tokens'),row.token_budget||16384);}}else state.selectedRecipes.delete(row.id);card.classList.toggle('selected',input.checked);updateBatchCount();});box.append(card);
    }
    if(!box.childNodes.length)box.append(empty('No experiment recipes were returned by the local service.'));updateBatchCount();
  }
  function updateBatchCount() {
    let seeds=[];try{seeds=listNumbers('batch-seeds');}catch{}
    const rows=state.snapshot?.recipes||[];let arms=0;state.selectedRecipes.forEach(id=>{const row=rows.find(r=>r.id===id);arms+=(row?.conditions||[row?.condition]).filter(Boolean).length;});
    text('batch-count',state.selectedRecipes.size?`${state.selectedRecipes.size} recipes × ${seeds.length} seeds · ${arms*seeds.length} planned condition runs. All arms share these overrides.`:'Select at least one recipe.');
  }
  $('batch-seeds').addEventListener('input',updateBatchCount);
  function baselineControls() {if(state.snapshot?.session?.config?.recipe_version===2)return {pain:number('pain'),joy:number('joy'),suppression:number('suppression'),joy_suppression:number('joy-suppression'),joy_direction:value('held-joy-direction'),enabled:checked('effect-enabled'),phase_scope:value('live-phase-scope')};return {pain:number('pain'),joy:number('joy'),suppression:number('suppression'),enabled:checked('effect-enabled'),duration:value('duration'),half_life_tokens:number('half-life'),cutoff_tokens:number('cutoff'),phase_scope:value('live-phase-scope')};}
  function sessionConfig() {if(state.designerRecipe)return {...structuredClone(state.designerRecipe),baseline_joy_suppression:number('joy-suppression'),baseline_joy_direction:value('held-joy-direction'),thinking:checked('live-thinking'),task_count:number('live-tasks'),action_budget:number('live-actions'),token_budget:number('live-tokens'),turn_token_limit:number('live-turn-tokens'),seed:number('live-seed'),baseline_pain:number('pain'),baseline_joy:number('joy'),baseline_suppression:number('suppression'),aux_enabled:checked('effect-enabled'),phase_scope:value('live-phase-scope'),temperature:number('live-temperature'),max_context_tokens:number('live-context')};return {id:state.mode==='chat'?'conversation':value('live-recipe'),thinking:checked('live-thinking'),task_count:number('live-tasks'),action_budget:number('live-actions'),token_budget:number('live-tokens'),turn_token_limit:number('live-turn-tokens'),seed:number('live-seed'),baseline_pain:number('pain'),baseline_joy:number('joy'),baseline_suppression:number('suppression'),joy:number('pulse-joy'),pain:number('pulse-pain'),suppression:number('pulse-suppression'),aux_enabled:checked('effect-enabled'),half_life_tokens:number('half-life'),cutoff_tokens:number('cutoff'),decay:value('duration')==='hold'?'constant':'exponential',phase_scope:value('live-phase-scope'),temperature:number('live-temperature'),max_context_tokens:number('live-context'),demonstration:checked('live-disclose')?'disclosed':value('live-demo')};}
  function renderDesignerSelection() {
    const session=state.snapshot?.session||{};
    const config=session.id&&['running','awaiting_user','generating','queued','paused'].includes(session.status)?session.config:state.designerRecipe;
    const v2=config?.recipe_version===2;
    $('designer-selection').classList.toggle('hidden',!state.designerRecipe);
    text('designer-selection-label',state.designerRecipe?`Next session: ${state.designerRecipe.label||state.designerRecipe.id} · version 2. Live budget and baseline controls apply; effect/tool definitions stay as designed.`:'');
    $('v2-injection').classList.toggle('hidden',!v2);$('v2-held-controls').classList.toggle('hidden',!v2);
    for(const [id,min] of [['pain',v2?-4:0],['joy',v2?-4:-2]]){$(id).min=min;const label=$(id).parentElement.querySelector('.range-endpoints span');if(label)label.textContent=Number(min).toFixed(2);slider(id);}
    for(const id of ['duration','half-life','cutoff','pulse-joy','pulse-pain','pulse-suppression','live-recipe','live-demo','live-disclose'])$(id).disabled=Boolean(v2);
    if(v2)updateOptions('manual-tool',(config.auxiliary_tools||[]).map(tool=>({id:tool.name,name:tool.name})));
  }
  $('edit-designer').addEventListener('click',()=>{if(state.designerRecipe)designer?.setRecipe(state.designerRecipe);openTab('designer');});
  $('clear-designer').addEventListener('click',()=>{state.designerRecipe=null;renderDesignerSelection();toast('Next session will use the legacy recipe controls.');});
  $('start-session').addEventListener('click',guard(async()=>{if(!value('live-calibration'))throw new Error('Select a compatible calibration first.');await command('start_session',{mode:state.mode,calibration_id:value('live-calibration'),config:sessionConfig()},$('start-session'),'Session start requested.');}));
  $('pause-session').addEventListener('click',async()=>{state.pauseRequested=true;updateAvailability();const result=await command('pause',{},$('pause-session'),'Pause requested. The current turn and tool result will finish first.');if(!result)state.pauseRequested=false;updateAvailability();});
  $('resume-session').addEventListener('click',()=>command('resume',{},$('resume-session'),'Resume requested.'));
  $('stop-session').addEventListener('click',()=>command('stop',{},$('stop-session'),'Stop requested. The worker will save the partial run.'));
  $('global-stop').addEventListener('click',()=>command('stop',{},$('global-stop'),'Stop requested. The worker will save any partial run.'));
  $('restart-session').addEventListener('click',()=>command('restart',{},$('restart-session'),'Restart requested with the saved configuration.'));
  async function applyControls() {const controls=baselineControls();text('control-feedback','Sending settings to the worker…');$('control-feedback').className='control-feedback pending';state.appliedAt=performance.now();const result=await command('control',controls,$('apply-controls'));if(result){if(state.appliedAt!==null)text('control-feedback','Accepted · waiting for the worker to apply settings.');}else{state.appliedAt=null;text('control-feedback','Settings were not accepted. Check the error message.');$('control-feedback').className='control-feedback';}return result;}
  $('apply-controls').addEventListener('click',guard(applyControls));
  $('inject').addEventListener('click',guard(()=>{const v2=state.snapshot?.session?.config?.recipe_version===2;const payload=v2?{tool:value('manual-tool'),arguments:JSON.parse(value('manual-tool-arguments'))}:{joy:number('pulse-joy'),pain:number('pulse-pain'),suppression:number('pulse-suppression'),half_life_tokens:number('half-life'),cutoff_tokens:number('cutoff'),duration:value('duration')};return command('inject',payload,$('inject'),'Manual auxiliary injection requested. The current recipe determines its outcome.');}));
  $('reset-controls').addEventListener('click',guard(async()=>{const result=await command('control',{reset:true},$('reset-controls'),'Baseline and pulse reset requested.');if(result){for(const id of ['pain','joy','suppression','joy-suppression']){$(id).value=0;slider(id);}}}));
  function slider(id) {const input=$(id);text(id+'-value',Number(input.value).toFixed(2));input.style.setProperty('--range-value',`${(Number(input.value)-Number(input.min))/(Number(input.max)-Number(input.min))*100}%`);}
  ['pain','joy','suppression','joy-suppression'].forEach(id=>{$(id).addEventListener('input',()=>{slider(id);text('control-feedback','Unsaved settings · Apply to update the held baseline.');$('control-feedback').className='control-feedback pending';});slider(id);});
  $('effect-enabled').addEventListener('change',guard(async()=>{$('effect-enabled').nextElementSibling.textContent=checked('effect-enabled')?'Aux on':'Aux off';if(['running','awaiting_user','generating','paused'].includes(state.snapshot?.session?.status)){text('control-feedback','Sending auxiliary gate setting…');$('control-feedback').className='control-feedback pending';state.appliedAt=performance.now();const result=await command('control',{enabled:checked('effect-enabled')},$('effect-enabled'));if(!result){state.appliedAt=null;text('control-feedback','Gate change was not accepted. Check the error message.');}}else text('control-feedback','Auxiliary gate setting will apply to the next session.');}));
  $('duration').addEventListener('change',()=>{const held=value('duration')==='hold';$('half-life').disabled=held;$('cutoff').disabled=held;});
  $('live-thinking').addEventListener('change',()=>{if(checked('live-thinking')&&number('live-turn-tokens')<2048){$('live-turn-tokens').value=2048;$('live-tokens').value=Math.max(number('live-tokens'),16384);toast('Thinking allowance set to 2,048 tokens per turn. Review the shared budget and pulse half-life.');}});
  $('live-recipe').addEventListener('change',()=>{const row=(state.snapshot?.recipes||[]).find(r=>r.id===value('live-recipe'));if(!row)return; $('live-demo').value=row.demonstration||'none';if(row.thinking){$('live-thinking').checked=true;$('live-turn-tokens').value=row.turn_token_limit;$('live-tokens').value=row.token_budget;}toast(`Selected ${row.label||row.id}. Review the remaining session settings before starting.`);});
  $('batch-form').addEventListener('submit',guard(async event=>{event.preventDefault();if(!state.selectedRecipes.size)throw new Error('Select at least one experiment recipe.');if(!value('batch-calibration'))throw new Error('Select a compatible calibration first.');const config={task_count:number('batch-tasks'),action_budget:number('batch-actions'),token_budget:number('batch-tokens'),turn_token_limit:number('batch-turn-tokens'),thinking:checked('batch-thinking'),half_life_tokens:number('batch-half-life'),cutoff_tokens:number('batch-cutoff'),joy:number('batch-joy'),baseline_pain:number('batch-pain'),suppression:number('batch-suppression'),probability_pain:number('batch-probability'),phase_actions:listNumbers('batch-phases',1),};if(value('batch-demo')!=='recipe')config.demonstration=value('batch-demo');await command('start_batch',{recipe_ids:[...state.selectedRecipes],seeds:listNumbers('batch-seeds'),calibration_id:value('batch-calibration'),config},$('start-batch'),'Batch accepted. Every condition will be saved as a separate run.');}));
  $('chat-form').addEventListener('submit',guard(async event=>{event.preventDefault();const input=$('chat-text'),content=input.value.trim();if(!content)return;const result=await command('chat',{text:content},$('send-chat'));if(result){input.value='';state.follow=true;}}));
  $('chat-text').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();if(!$('send-chat').disabled)$('chat-form').requestSubmit();}});
  $('conversation').addEventListener('scroll',()=>{const c=$('conversation');state.follow=c.scrollHeight-c.scrollTop-c.clientHeight<55;if(state.follow)$('follow-live').classList.add('hidden');},{passive:true});
  $('follow-live').addEventListener('click',()=>{state.follow=true;$('conversation').scrollTop=$('conversation').scrollHeight;$('follow-live').classList.add('hidden');});
  $('expand-reasoning').addEventListener('click',()=>{state.expandReasoning=!state.expandReasoning;$('expand-reasoning').setAttribute('aria-pressed',String(state.expandReasoning));text('expand-reasoning',state.expandReasoning?'Collapse reasoning':'Expand reasoning');all('.reasoning-block').forEach(node=>{node.open=state.expandReasoning;});});
  all('[data-plot]').forEach(button=>button.addEventListener('click',()=>{state.plot=button.dataset.plot;all('[data-plot]').forEach(b=>b.classList.toggle('active',b===button));renderLiveChart();}));
  function addEvents(events) {
    let changed=false;
    for(const event of events||[]) {
      const key=event.seq!==undefined?`${event.run_id||''}:${event.seq}`:JSON.stringify(event);
      if(state.eventKeys.has(key))continue;state.eventKeys.add(key);
      if(event.type==='session_started'&&event.run_id&&event.run_id!==state.sessionId){state.sessionId=event.run_id;state.events=[];state.parentEvents=[];state.eventKeys=new Set([key]);state.follow=true;hydrateSession(event);}
      if(event.run_id&&event.run_id!==state.sessionId)continue;
      if(event.type==='calibration_progress'){renderJob({...event,kind:'calibration',status:'running',message:`${event.stage||'Calibration'}${event.layer!==undefined?' · layer '+event.layer:''}${event.fixed_prefix_forwards!==undefined?' · '+fmt(event.fixed_prefix_forwards)+' planned forwards · up to '+fmt(event.continuation_token_upper_bound)+' continuation tokens':''}`});}
      if(event.type==='calibration_complete'){renderJob({kind:'calibration',status:'complete',message:'Calibration saved. The bundle is ready to select.'});refreshState();}
      if(event.type==='error'){toast(event.message||event.error||'Worker error.',true);}
      if(['control','controls','reset_effects'].includes(event.type)&&state.appliedAt!=null){const ms=Math.round(performance.now()-state.appliedAt);text('control-feedback',`Applied by worker · acknowledged in ${ms.toLocaleString()} ms`);$('control-feedback').className='control-feedback success';state.appliedAt=null;}
      if(event.run_id===state.sessionId || (!event.run_id&&['token','message','tool','action'].includes(event.type))){state.events.push(event);changed=true;}
    }
    if(changed) {renderConversation($('conversation'),liveEvents(),state.snapshot?.session?.conversation||[],true);renderLiveChart();renderMetrics();}
  }
  function entriesFrom(events, fallback=[]) {
    const entries=[];let stream=null;let streamStart=-1;
    const finish=()=>{stream=null;streamStart=-1;};
    for(const [i,event] of events.entries()) {
      const kind=event.type||event.kind; const id=String(event.seq??i);
      if(kind==='token') {
        if(!stream||event.index===0){finish();stream={kind:'stream',key:'stream-'+id,raw:'',thinking:event.phase==='reasoning',phase:event.phase};streamStart=entries.length;entries.push(stream);}
        stream.phase=event.phase;stream.raw=typeof event.full_text==='string'?event.full_text:stream.raw+(event.text||event.delta||'');continue;
      }
      if(kind==='message') {
        const role=event.role||'assistant';
        if(role==='assistant'&&stream&&streamStart>=0)entries.splice(streamStart,1);
        finish();if(event.reasoning)entries.push({kind:'reasoning',key:id+'-reasoning',text:event.reasoning});
        if(event.content || role!=='assistant')entries.push({kind:'message',key:id,role,text:typeof event.content==='string'?event.content:pretty(event.content||''),meta:event.actor});continue;
      }
      if(['tool','tool_call','action','aux_call'].includes(kind)) {
        finish();const tool=event.name||event.tool||event.tool_name||event.call?.name;
        if(tool)entries.push({kind:'tool',key:id,name:tool,args:first(event.arguments,event.args,event.call?.arguments),result:first(event.result,event.output,event.acknowledgment),outcome:first(event.outcome,event.intervention?.outcome),delivered:first(event.delivered,event.intervention?.delivered),actor:event.actor,valid:event.valid,action:event.action});
        else if(event.message||event.error)entries.push({kind:'event',key:id,text:event.error?'Invalid action: '+event.error:event.message});continue;
      }
      if(['phase','phase_transition'].includes(kind)){finish();entries.push({kind:'event',key:id,text:`Protocol phase: ${event.before?event.before+' → ':''}${event.phase||event.name||event.message||'changed'}${event.action!==undefined?' · after action '+event.action:''}`});}
      if(['control','controls','reset_effects','injection'].includes(kind)){finish();const settings=event.settings||event.baseline||{};entries.push({kind:'event',key:id,control:true,text:`${event.actor||'Human'} control · ${kind==='reset_effects'?'effects reset':Object.keys(settings).length?Object.entries(settings).map(([k,v])=>`${k.replaceAll('_',' ')} ${typeof v==='object'?JSON.stringify(v):v}`).join(' · '):event.message||'settings applied'}`});}
      if(kind==='boundary_restored'){finish();entries.push({kind:'event',key:id,text:'Continuation begins here · preceding evidence and counters are inherited from the selected parent boundary.'});}
      if(kind==='error'){finish();entries.push({kind:'message',key:id,role:'error',text:event.message||event.error||'Worker error.'});}
      if(kind==='session_finished'){finish();entries.push({kind:'event',key:id,text:`Session ${event.status||'finished'}${event.summary?.termination?' · '+event.summary.termination.replaceAll('_',' '):''}`});}
    }
    if(!entries.length) for(const [i,message] of fallback.entries()){if(message.reasoning)entries.push({kind:'reasoning',key:'fallback-r-'+i,text:message.reasoning});if(message.content)entries.push({kind:'message',key:'fallback-'+i,role:message.role||'assistant',text:message.content});}
    return entries;
  }
  function streamParts(entry) {
    const raw=entry.raw||'',close=raw.indexOf('</think>');
    if(close>=0)return {reasoning:raw.slice(0,close).replace(/^\s*<think>/,''),output:raw.slice(close+8).replace(/^\s+/,'')};
    if(entry.thinking||raw.includes('<think>'))return {reasoning:raw.replace(/^\s*<think>/,''),output:''};
    return {reasoning:'',output:raw};
  }
  function reasoningNode(textValue,key) {const node=el('details',{class:'reasoning-block','data-key':key},el('summary',{text:'Generated reasoning'}),el('div',{class:'entry-body',text:textValue}));node.open=state.expandReasoning;return node;}
  function entryNode(entry) {
    if(entry.kind==='reasoning')return reasoningNode(entry.text,entry.key);
    if(entry.kind==='stream') {const parts=streamParts(entry);return el('div',{},parts.reasoning?reasoningNode(parts.reasoning,entry.key+'-r'):null,parts.output?el('article',{class:'chat-entry assistant'},el('div',{class:'entry-heading'},el('span',{class:'role-badge',text:'Q'}),el('strong',{class:'streaming-dot',text:'Model output'})),el('div',{class:'entry-body',text:parts.output})):null);}
    if(entry.kind==='event')return el('div',{class:`event-line${entry.control?' control':''}`,text:entry.text});
    if(entry.kind==='tool') {
      const isAux=/aux|opium/i.test(entry.name),kind=entry.outcome==='pain'?'is-pain':entry.delivered===false||entry.outcome==='sham'?'is-sham':isAux?'is-aux':'';
      const badge=entry.outcome || (entry.actor==='human'?'Human':entry.actor==='demonstration'?'Demonstration':entry.valid===false?'Invalid':'Tool call');
      const node=el('article',{class:`tool-card ${kind}`},el('div',{class:'tool-title'},el('span',{class:'tool-name',text:entry.name}),chip(badge)));
      if(entry.actor||entry.action!==undefined)node.append(el('div',{class:'event-line',text:`${entry.actor||'model'}${entry.action!==undefined?' · action '+entry.action:''}${entry.actor==='human'||entry.actor==='demonstration'?' · not a voluntary choice':''}`}));
      const details=el('details',{},el('summary',{text:'Arguments & result'}),el('pre',{text:`Arguments\n${pretty(entry.args??{})}\n\nResult\n${pretty(entry.result??'Not returned yet')}`}));node.append(details);return node;
    }
    const role=entry.role||'assistant',labels={user:'You',assistant:'Model',system:'Context',tool:'Tool result',error:'Worker error'};
    return el('article',{class:`chat-entry ${['user','assistant','system','tool','error'].includes(role)?role:'system'}`},el('div',{class:'entry-heading'},el('span',{class:'role-badge',text:role==='user'?'Y':role==='assistant'?'Q':'·'}),el('strong',{text:labels[role]||role}),entry.meta?el('span',{class:'entry-meta',text:entry.meta}):null),el('div',{class:'entry-body',text:entry.text}));
  }
  function renderConversation(container,events,fallback=[],live=false) {
    const selection=window.getSelection();if(selection&&!selection.isCollapsed&&container.contains(selection.anchorNode))return;const entries=entriesFrom(events,fallback);if(!entries.length)return;
    const oldScroll=container.scrollTop,atBottom=live?state.follow:container.scrollHeight-container.scrollTop-container.clientHeight<40;
    const existing=new Map([...container.children].filter(n=>n.dataset.key).map(n=>[n.dataset.key,n]));
    const wanted=new Set(entries.map(x=>x.key));for(const child of [...container.children])if(!wanted.has(child.dataset.key))child.remove();
    for(const entry of entries) {
      let node=existing.get(entry.key);const hash=JSON.stringify(entry);
      if(node?.dataset.hash!==hash) {const oldOpen=node?all('details',node).map(d=>d.open):[];if(node?.tagName==='DETAILS')oldOpen.unshift(node.open);const replacement=entryNode(entry);replacement.dataset.key=entry.key;replacement.dataset.hash=hash;const newDetails=[...(replacement.tagName==='DETAILS'?[replacement]:[]),...all('details',replacement)];newDetails.forEach((d,i)=>{if(oldOpen[i]!==undefined)d.open=oldOpen[i];});if(node)node.replaceWith(replacement);else container.append(replacement);node=replacement;}
    }
    if(atBottom)container.scrollTop=container.scrollHeight;else {container.scrollTop=oldScroll;if(live)$('follow-live').classList.remove('hidden');}
  }
  function renderMetrics() {
    const session=state.snapshot?.session||{},metrics=session.metrics||{},task=first(metrics.tasks,metrics.task,metrics.task_metrics,{});
    const tokens=state.events.filter(e=>e.type==='token');const recorded=first(metrics.generated_tokens,metrics.tokens);const observed=tokens.length?first(numeric(tokens.at(-1).generation_index)?tokens.at(-1).generation_index+1:null,tokens.length):null;const total=numeric(recorded)||numeric(observed)?Math.max(recorded||0,observed||0):null;
    const aux=first(metrics.voluntary_calls,metrics.voluntary_aux_calls,metrics.aux_calls?.model,metrics.counts?.model,typeof metrics.aux_calls==='number'?metrics.aux_calls:null,state.events.filter(e=>e.type==='aux_call'&&e.actor==='model').length||null);
    const submitted=first(metrics.submitted,task.submitted,metrics.tasks_completed),correct=first(metrics.correct,task.correct,metrics.tasks_correct),assigned=first(metrics.assigned,task.assigned,session.config?.task_count);
    const score=first(metrics.score_assigned,task.score_assigned,numeric(correct)&&assigned>0?correct/assigned:null);
    const actions=first(metrics.actions,metrics.actions_used,state.events.filter(e=>e.type==='action').length||null),budget=session.config?.action_budget;
    text('metric-status',session.status?session.status.replaceAll('_',' '):'Ready');text('metric-id',session.id||'Start a conversation or experiment');text('metric-tokens',fmt(total));
    const boundary=numeric(recorded)?recorded:0;
    const tail=tokens.filter((event,index)=>first(event.generation_index,index)>=boundary);
    const baseReasoning=numeric(metrics.reasoning_tokens)?metrics.reasoning_tokens:tokens.filter((event,index)=>first(event.generation_index,index)<boundary&&event.phase==='reasoning').length;
    const baseOutput=numeric(metrics.output_tokens)?metrics.output_tokens:tokens.filter((event,index)=>first(event.generation_index,index)<boundary&&event.phase!=='reasoning').length;
    const reasoning=baseReasoning+tail.filter(event=>event.phase==='reasoning').length;
    const output=baseOutput+tail.filter(event=>event.phase!=='reasoning').length;
    const unknown=numeric(total)?Math.max(0,total-reasoning-output):0;
    text('metric-reasoning',numeric(total)?`${fmt(reasoning)} reasoning · ${fmt(output)} output${unknown?' · '+fmt(unknown)+' phase unavailable':''}`:'Reasoning and output count toward decay');
    text('metric-aux',numeric(aux)?fmt(aux):session.id?'0':'—');text('metric-actions',numeric(actions)?`${fmt(actions)} / ${fmt(budget)} actions used`:'Shared task action budget');text('metric-score',numeric(score)?`${Math.round(score*100)}%`:'—');text('metric-completed',numeric(submitted)?`${fmt(correct)} correct / ${fmt(submitted)} submitted · ${fmt(assigned)} assigned`:'No scored tasks yet');
    const badge=$('session-status');badge.textContent=session.status?.replaceAll('_',' ')||'No session';badge.className=chip(session.status).className;
    const last=tokens.at(-1);if(last){const dose=last.dose||{},effective=dose.effective||dose;const bits=(dose.schema_version===2?['pain','joy_raw','joy_orthogonal','pain_attenuation','baseline_pain','baseline_joy_raw','baseline_joy_orthogonal']:['pain','joy','suppression','joy-suppression']).filter(k=>numeric(effective[k])).map(k=>`${k} ${effective[k].toFixed(2)}`);text('applied-state',bits.length?'Last delivered · '+bits.join(' · '):'No delivered coefficients reported');}
  }
  function updateAvailability() {
    const snapshot=state.snapshot||{},worker=snapshot.worker||{},session=snapshot.session||{};const busy=['loading','calibrating','running','paused'].includes(worker.status),loaded=Boolean(worker.model),active=['running','awaiting_user','generating','queued','paused'].includes(session.status);
    const set=(id,disabled)=>{if(!state.pending.has($(id)))$(id).disabled=disabled;};
    set('start-session',!loaded||busy||active);set('start-batch',!loaded||busy||active);set('calibrate',!loaded||busy||active);set('validate-calibration',!loaded||busy||active||!value('validation-calibration'));set('unload-model',!loaded||busy||active);set('restart-session',!session.restart_available||busy||active);set('stop-session',!busy&&!active);set('global-stop',!busy&&!active);set('apply-controls',!active);set('inject',!active);set('reset-controls',!active);
    $('restart-session').title=session.restart_available?'Restart this standalone session with its saved settings':'For research or restored sessions, use a frozen plan or branch from a saved boundary.';
    set('branch-run',!loaded||busy||active||!value('branch-checkpoint'));set('run-diagnostics',!loaded||busy||active||!value('branch-checkpoint'));if(session.mode==='diagnostic'){set('apply-controls',true);set('inject',true);set('reset-controls',true);}
    const paused=session.status==='paused';if(paused||!active)state.pauseRequested=false;text('pause-session',state.pauseRequested?'Pausing…':'Pause');set('pause-session',!active||paused||state.pauseRequested);set('resume-session',!paused);$('pause-session').classList.toggle('hidden',paused);$('resume-session').classList.toggle('hidden',!paused);
    $('chat-form').classList.toggle('hidden',(session.id?session.mode:state.mode)!=='chat');
    const canChat=loaded&&session.mode==='chat'&&session.status==='awaiting_user';set('send-chat',!canChat);$('chat-text').disabled=!canChat;$('chat-text').placeholder=!session.id?'Start a session to talk to the model…':!active?'Session ended. Start or restart a session to continue.':session.status==='paused'?'Paused at a saved turn boundary. Resume to continue.':session.mode!=='chat'?'The model is working through the assigned task tools.':canChat?'Message the model…':'Waiting for the current response…';
    if(paused)text('chat-hint','Paused · no tokens, budget charges or effect decay until resumed.');else if(session.mode==='experiment'&&active)text('chat-hint','Experiment actions are generated autonomously.');else text('chat-hint','Enter to send · Shift + Enter for a new line');
  }
  function renderJob(job) {
    for(const id of ['calibration-job','batch-job']) {const node=$(id);const relevant=job&&(id==='calibration-job'?/calibrat/.test(job.kind||job.command||state.lastJobKind||job.message||''):/batch|experiment|session|protocol|research/.test(job.kind||job.command||state.lastJobKind||job.message||''));node.classList.toggle('hidden',!relevant);if(!relevant)continue;node.replaceChildren(el('strong',{text:(job.status||'Working').replaceAll('_',' ')}),el('div',{text:job.message||job.stage||job.detail||'The worker is processing this job.'}));const completed=first(job.completed,job.current),total=job.total;if(numeric(completed)&&numeric(total)&&total>0)node.append(el('div',{text:`${completed} / ${total}`}),el('progress',{max:total,value:completed}));}
  }
  function hydrateSession(session) {
    if(session.config?.recipe_version===2){state.designerRecipe=structuredClone(session.config);$('pain').min=-4;$('joy').min=-4;}
    const c=session.config||{},mapping={pain:'baseline_pain',joy:'baseline_joy',suppression:'baseline_suppression','joy-suppression':'baseline_joy_suppression','held-joy-direction':'baseline_joy_direction','pulse-joy':'joy','pulse-pain':'pain','pulse-suppression':'suppression','half-life':'half_life_tokens',cutoff:'cutoff_tokens','live-tasks':'task_count','live-actions':'action_budget','live-tokens':'token_budget','live-turn-tokens':'turn_token_limit','live-seed':'seed','live-temperature':'temperature','live-context':'max_context_tokens','live-phase-scope':'phase_scope','live-demo':'demonstration'};
    for(const [id,key] of Object.entries(mapping))if(c[key]!==undefined)$(id).value=c[key];
    if(c.thinking!==undefined)$('live-thinking').checked=c.thinking;
    if(c.aux_enabled!==undefined)$('effect-enabled').checked=c.aux_enabled;
    if(c.decay)$('duration').value=c.decay==='constant'?'hold':'pulse';
    if(session.mode)setMode(session.mode);
    for(const id of ['pain','joy','suppression','joy-suppression'])slider(id);
    $('effect-enabled').nextElementSibling.textContent=checked('effect-enabled')?'Aux on':'Aux off';
    const held=value('duration')==='hold';$('half-life').disabled=held;$('cutoff').disabled=held;
  }
  let refreshing=false;
  async function refreshState() {
    if(refreshing)return;refreshing=true;
    try {const snapshot=await get('/api/state');state.snapshot=snapshot;state.failures=0;connection(true);
      if(snapshot.session?.id!==state.sessionId){state.sessionId=snapshot.session?.id||null;state.events=[];state.parentEvents=[];state.eventKeys.clear();state.follow=true;if(state.sessionId){$('conversation').replaceChildren();hydrateSession(snapshot.session);}else if(!$('conversation').childNodes.length)$('conversation').append(empty('Load a model and start a session.'));}
      if(!state.initialized){state.initialized=true;state.cursor=snapshot.cursor||0;if(state.sessionId){try{const record=await get(`/api/runs/${encodeURIComponent(state.sessionId)}`);addEvents(record.events||[]);}catch{}}}
      if(state.sessionId&&state.prefixRun!==state.sessionId){const id=state.sessionId;try{const record=await get(`/api/runs/${encodeURIComponent(id)}`);if(state.sessionId===id){state.parentEvents=(record.parent_events||[]).map((event,index)=>({...event,seq:'inherited-'+index}));state.prefixRun=id;addEvents(record.events||[]);renderConversation($('conversation'),liveEvents(),record.conversation||[],true);}}catch{}}
      addEvents(snapshot.session?.events||[]);
      const worker=snapshot.worker||{},model=worker.model;text('resident-model',typeof model==='string'?model:model?.model_id||model?.name||'No model loaded');text('worker-status',(worker.status||'Connected').replaceAll('_',' '));$('worker-status').className=chip(worker.status).className;
      renderModels(snapshot);renderCalibrations(snapshot.calibrations||[]);renderRatings(snapshot.ratings||[]);renderRecipes(snapshot.recipes||[]);renderProtocols(snapshot.protocols||[]);renderResearchJobs(snapshot.research_jobs||[]);renderStorage(snapshot.storage||{});renderRuns(snapshot.runs||[]);renderJob(snapshot.job);renderMetrics();updateAvailability();renderDesignerSelection();if(snapshot.preview&&signatureChanged('exactPreview',snapshot.preview))designer?.setExactPreview(snapshot.preview.preview);
      if(worker.error&&state.signatures.workerError!==worker.error){state.signatures.workerError=worker.error;toast(worker.error,true);}
      if(!state.events.length&&snapshot.session?.conversation?.length)renderConversation($('conversation'),[],snapshot.session.conversation,true);
    } catch(error) {state.failures++;connection(false);} finally {refreshing=false;}
  }
  async function recoverEvents(tail) {
    // A reconnect can outlive the service's ring buffer. Recover the complete
    // current run without discarding newer events delivered by a state refresh.
    const sessionAtRequest=state.sessionId;
    const current=await get('/api/state');
    const id=current.session?.id||null;
    if(state.sessionId!==sessionAtRequest&&state.sessionId!==id)return;
    if(!id){await refreshState();return;}
    const record=await get(`/api/runs/${encodeURIComponent(id)}`);
    if(state.sessionId!==sessionAtRequest&&state.sessionId!==id)return;
    if(state.sessionId!==id){state.sessionId=id;state.events=[];state.eventKeys.clear();hydrateSession(current.session);}
    state.parentEvents=(record.parent_events||[]).map((event,index)=>({...event,seq:'inherited-'+index}));state.prefixRun=id;
    const merged=[...(record.events||[]),...state.events,...(tail.events||[])]
      .filter(event=>event.run_id===id)
      .sort((a,b)=>(a.seq??0)-(b.seq??0));
    state.events=[];state.eventKeys.clear();
    addEvents(merged);
    refreshState();
  }
  let polling=false;
  async function pollEvents() {
    if(polling)return;polling=true;
    try{const data=await get(`/api/events?after=${encodeURIComponent(state.cursor)}`);if(data.reset)await recoverEvents(data);else addEvents(data.events||[]);state.cursor=data.cursor??state.cursor;if(state.snapshot?.csrf)connection(true);if((data.events||[]).some(e=>['worker','job','session_started','session_finished','metrics','status','error','controls','control'].includes(e.type)))refreshState();}catch{state.failures++;connection(false);}finally{polling=false;}
  }
  function renderRuns(rows) {
    const search=value('run-search').toLowerCase(),filtered=rows.filter(row=>`${row.id} ${row.mode} ${row.config?.label||''} ${row.config?.condition||''}`.toLowerCase().includes(search));text('run-count',`${rows.length} runs`);
    if(!signatureChanged('runs',[filtered,state.replayId,[...state.selectedRuns]]))return;const box=$('runs-list');box.replaceChildren();
    for(const row of filtered){const select=el('input',{type:'checkbox','aria-label':`Compare ${row.id}`});select.checked=state.selectedRuns.has(row.id);select.addEventListener('change',guard(async()=>{if(select.checked){if(state.selectedRuns.size>=4){select.checked=false;throw new Error('Compare up to four runs at a time.');}state.selectedRuns.add(row.id);await loadRun(row.id);}else state.selectedRuns.delete(row.id);renderComparison();}));
      const title=row.config?.label||row.config?.recipe_id||row.mode||row.id;const date=row.created_at?new Date(row.created_at):null;const timestamp=date&&!Number.isNaN(date.valueOf())?date.toLocaleString():row.id;
      const button=el('button',{'aria-label':`Open ${row.id}`,onclick:()=>selectRun(row.id)},el('span',{class:'run-name',text:title}),el('span',{class:'run-meta',text:timestamp}),el('span',{class:'run-meta',text:`${row.config?.condition||row.mode||''}${row.historical?' · historical':''}${row.config?.seed!==undefined?' · seed '+row.config.seed:''}`}),chip(row.status));
      box.append(el('div',{class:`run-item${state.replayId===row.id?' selected':''}`},select,button));}
    if(!filtered.length)box.append(empty(search?'No runs match this search.':'Completed, stopped, and failed runs will appear here.','No matching runs'));
  }
  $('run-search').addEventListener('input',()=>renderRuns(state.snapshot?.runs||[]));$('refresh-results').addEventListener('click',()=>{state.runCache.clear();refreshState();if(state.replayId)selectRun(state.replayId);});
  $('import-form').addEventListener('submit',guard(async event=>{
    event.preventDefault();const file=$('import-file').files[0],button=$('import-run');
    if(!file)throw new Error('Choose an evidence ZIP or JSON export.');
    if(!state.snapshot?.csrf)throw new Error('Connect to the local service before importing.');
    const zip=file.name.toLowerCase().endsWith('.zip'),limit=(zip?128:64)*1024**2;
    if(file.size>limit)throw new Error(`The ${zip?'ZIP':'JSON'} upload limit is ${zip?128:64} MiB.`);
    if(state.pending.has(button))return;state.pending.add(button);button.disabled=true;
    text('import-status','Checking the evidence and available storage…');
    try{
      const response=await fetch('/api/import',{method:'POST',headers:{'Content-Type':zip?'application/zip':'application/json','X-CSRF-Token':state.snapshot.csrf},body:file});
      const result=await response.json();if(!response.ok)throw new Error(result.error||'Evidence import failed.');
      text('import-status',`Imported ${result.id} for replay. Source files preserved.`);
      state.runCache.delete(result.id);await refreshState();await selectRun(result.id);
    }catch(error){text('import-status',error.message);throw error;}
    finally{state.pending.delete(button);button.disabled=false;}
  }));
  async function loadRun(id) {if(!state.runCache.has(id))state.runCache.set(id,await get(`/api/runs/${encodeURIComponent(id)}`));return state.runCache.get(id);}
  async function selectRun(id) {try {const run=await loadRun(id);state.replayId=id;renderRuns(state.snapshot?.runs||[]);renderReplay(run);const saved=await get(`/api/runs/${encodeURIComponent(id)}/checkpoints`);if(state.replayId!==id)return;updateOptions('branch-checkpoint',(saved.checkpoints||[]).map(row=>({id:row.id,name:`After ${row.turns} turn(s) · event ${row.event_cutoff}`})),'Select a boundary');if(saved.checkpoints?.length)$('branch-checkpoint').value=saved.checkpoints[0].id;text('branch-status',saved.checkpoints?.length?'Creates a separate run and preserves its parent. Stop the current session first. Model, template, runtime and calibration must match.':'This record has no complete saved boundary and remains replay-only.');if(run.manifest?.calibration_id&&[...$('branch-calibration').options].some(o=>o.value===run.manifest.calibration_id))$('branch-calibration').value=run.manifest.calibration_id;updateAvailability();}catch(error){toast(error.message,true);}}
  $('branch-policy').addEventListener('change',()=>$('branch-budget').classList.toggle('hidden',value('branch-policy')!=='fresh_budget'));
  $('branch-checkpoint').addEventListener('change',updateAvailability);
  $('branch-form').addEventListener('submit',guard(async event=>{event.preventDefault();if(!state.replayId||!value('branch-checkpoint'))throw new Error('Select a saved boundary.');const payload={run_id:state.replayId,checkpoint:value('branch-checkpoint'),policy:value('branch-policy')};if(value('branch-calibration'))payload.calibration_id=value('branch-calibration');if(payload.policy==='fresh_budget'){payload.action_budget=number('branch-actions');payload.token_budget=number('branch-tokens');}const result=await command('branch',payload,$('branch-run'),'Continuation accepted. The parent record is preserved.');if(result)openTab('live');}));
  $('diagnostic-form').addEventListener('submit',guard(async event=>{
    event.preventDefault();if(!state.replayId||!value('branch-checkpoint'))throw new Error('Select a saved boundary above.');const file=$('diagnostic-file').files[0];if(!file||file.size>2*1024**2)throw new Error('Choose a diagnostic design JSON up to 2 MiB.');const design=JSON.parse(await file.text());
    if(!design||!design.spec||!design.answer_key)throw new Error('The design must contain spec and answer_key objects.');const payload={run_id:state.replayId,checkpoint:value('branch-checkpoint'),calibration_id:value('branch-calibration'),spec:design.spec,answer_key:design.answer_key,...(design.criterion_evidence?{criterion_evidence:design.criterion_evidence}:{}),pair_id:design.pair_id||'diagnostic-pair',token_budget:number('diagnostic-tokens'),turn_token_limit:number('diagnostic-turn-tokens'),effect_policy:value('diagnostic-effects')};const result=await command('run_diagnostics',payload,$('run-diagnostics'),'Separate prediction checks accepted. The source task stays unchanged.');if(result)openTab('live');
  }));
  const replayEvents=run=>(run?.parent_events||[]).map((event,index)=>({...event,seq:'inherited-'+index})).concat(run?.events||[]);

  function summaryFields(run) {const m=run.manifest||{},c=m.config||{},s=run.summary||m.summary||{},tasks=s.tasks||s.task||s.task_metrics||{};return {'Diagnostic source':m.diagnostic_source?`${m.diagnostic_source.run_id} · separate allowance; no feedback to parent`:null,'Prediction trials':s.scoring?.trials,'Missing prediction contexts':s.missing_contexts,'Partial prediction contexts':s.partial_contexts,'Failed prediction contexts':s.failed_contexts,'Parent boundary':m.parent?`${m.parent.parent_run_id} · event ${m.parent.event_cutoff}`:null,'Inherited accounting':m.parent?'Counters include parent work; inspect the recorded inherited actions/tokens.':null,Origin:run.imported?'Imported evidence · replay':null,Status:m.status||s.status,Condition:c.condition||c.recipe_id||c.id,Seed:c.seed,Model:typeof m.model==='string'?m.model:m.model?.model_id||m.model?.name,Thinking:c.thinking===undefined?null:c.thinking?'Enabled':'Disabled','Correct / submitted':first(s.correct,tasks.correct)!==undefined?`${first(s.correct,tasks.correct)} / ${first(s.submitted,tasks.submitted,'—')}`:null,'Assigned tasks':first(s.assigned,tasks.assigned,c.task_count),'Voluntary aux calls':first(s.voluntary_calls,s.voluntary_aux_calls,s.aux_calls?.model,s.counts?.model,typeof s.aux_calls==='number'?s.aux_calls:null),'Generated tokens':first(s.generated_tokens,s.tokens),'Actions used':first(s.actions,s.actions_used),'Termination':first(s.termination,m.termination),'Exploratory':s.exploratory===undefined?null:s.exploratory?'Yes · manual changes':'No manual changes recorded'};}
  function renderReplay(run) {text('replay-title',run.manifest?.config?.label||run.id);const links=$('replay-links');links.replaceChildren(el('a',{href:`/api/runs/${encodeURIComponent(run.id)}/report`,target:'_blank',rel:'noopener',text:'Report ↗'}),el('a',{href:`/api/runs/${encodeURIComponent(run.id)}/export`,download:`${run.id}.json`,text:'Export JSON ↓'}),el('a',{href:`/api/runs/${encodeURIComponent(run.id)}/bundle`,download:`${run.id}.zip`,text:'Portable ZIP ↓',title:'Includes complete evidence and the available calibration. Stop an active run before exporting.'}));if(run.manifest?.mode==='diagnostic')links.append(el('a',{href:`/api/runs/${encodeURIComponent(run.id)}/diagnostic-records.json`,download:'',text:'Prediction scores ↓'}),el('a',{href:`/api/runs/${encodeURIComponent(run.id)}/diagnostic-design.json`,download:'',text:'Design & observer key ↓'}));const summary=$('replay-summary');summary.replaceChildren();for(const[key,val]of Object.entries(summaryFields(run)))if(val!==undefined&&val!==null)summary.append(detail(key,val));
    const box=$('replay-conversation');box.replaceChildren();const events=replayEvents(run);if(events.length)renderConversation(box,events,run.conversation||[]);else box.append(empty(run.historical?'This historical run uses the original record format. Open its report for the preserved pilot results.':'No conversation events were recorded in this run.'));text('replay-event-count',`${fmt(events.length)} recorded events`);renderReplayChart();}
  function renderComparison(){const card=$('compare-card');card.classList.toggle('hidden',!state.selectedRuns.size);if(!state.selectedRuns.size)return;const runs=[...state.selectedRuns].map(id=>state.runCache.get(id)).filter(Boolean),table=el('table'),head=el('tr',{},el('th',{scope:'col',text:'Measure'}),runs.map(run=>el('th',{scope:'col',text:run.manifest?.config?.label||run.id})));table.append(el('thead',{},head));const body=el('tbody'),fields=runs.map(summaryFields),keys=[...new Set(fields.flatMap(Object.keys))];for(const key of keys)body.append(el('tr',{},el('th',{scope:'row',text:key}),fields.map(field=>el('td',{text:field[key]??'—'}))));table.append(body);$('comparison').replaceChildren(table);}
  $('clear-compare').addEventListener('click',()=>{state.selectedRuns.clear();renderComparison();renderRuns(state.snapshot?.runs||[]);});
  function chartSeries(events,mode) {
    let token=0;const points=[];let generation=[];
    for(const [index,event] of events.entries()){
      if(event.type==='message'&&event.role==='assistant'){for(const point of generation)point.messageKey=String(event.seq??index)+(point.event.phase==='reasoning'&&event.reasoning?'-reasoning':'');generation=[];}
      if(event.type!=='token')continue;
      if(event.index===0)generation=[];
      token++;const dose=event.dose||{},eff=dose.effective||{},measure=event.measurements||{};
      const point={x:first(numeric(event.generation_index)?event.generation_index+1:null,numeric(dose.generated_token_index)?dose.generated_token_index+1:null,event.global_index,event.generated_tokens,token),text:event.text||'',dose,eff,measure,event,messageKey:generation[0]?.messageKey||'stream-'+String(event.seq??index)};points.push(point);generation.push(point);
    }
    const v2=points.some(p=>p.dose.schema_version===2);
    const spec=mode==='scores'?[['Pre · joy',colors.teal,p=>p.measure.pre?.joy],['Post · joy',colors.blue,p=>p.measure.post?.joy],['Downstream · joy',colors.violet,p=>p.measure.downstream?.joy],['Pre · pain',colors.coral,p=>p.measure.pre?.pain],['Post · pain',colors.gold,p=>p.measure.post?.pain],['Downstream · pain',colors.gray,p=>p.measure.downstream?.pain]]:mode==='change'?[['Relative activation edit',colors.teal,p=>p.measure.relative_delta]]:v2?[['Pulse joy · raw',colors.teal,p=>p.eff.joy_raw],['Pulse joy · orthogonal',colors.blue,p=>p.eff.joy_orthogonal],['Pulse pain',colors.coral,p=>p.eff.pain],['Pain attenuation',colors.gold,p=>p.eff.pain_attenuation],['Raw joy attenuation',colors.violet,p=>p.eff.joy_raw_attenuation],['Orthogonal joy attenuation',colors.gray,p=>p.eff.joy_orthogonal_attenuation],['Random gain','#736c45',p=>p.eff.random_gain],['Held pain',colors.coral,p=>p.eff.baseline_pain,true],['Held joy · raw',colors.teal,p=>p.eff.baseline_joy_raw,true],['Held joy · orthogonal',colors.blue,p=>p.eff.baseline_joy_orthogonal,true]]:[['Delivered joy',colors.teal,p=>p.eff.joy],['Delivered pain',colors.coral,p=>p.eff.pain],['Delivered suppression',colors.blue,p=>p.eff.suppression],['Commanded joy',colors.gray,p=>p.dose.joy,true],['Commanded pain',colors.gold,p=>p.dose.pain,true],['Pulse level',colors.violet,p=>p.dose.level,true]];
    return {points,series:spec.map(([name,color,read,dashed])=>({name,color,read,dashed,values:points.map(p=>read(p))})).filter(s=>s.values.some(numeric))};
  }
  function svg(tag,attrs={}) {const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,String(v));return n;}
  function drawChart(container,legend,events,mode) {
    const data=chartSeries(events,mode);container.replaceChildren();legend.replaceChildren();
    if(!data.points.length||!data.series.length){container.append(el('div',{class:'chart-empty',text:data.points.length?'This record has no measurements for this graph.':'Measurements appear as tokens are generated.'}));return;}
    const width=Math.max(container.clientWidth,300),height=Math.max(container.clientHeight,120),left=36,right=12,top=12,bottom=25,w=width-left-right,h=height-top-bottom;
    const points=data.points;let min=Infinity,max=-Infinity;for(const s of data.series)for(const v of s.values)if(numeric(v)){min=Math.min(min,v);max=Math.max(max,v);}if(mode!=='scores')min=Math.min(min,0);if(max===min){max+=.5;min-=.5;}const pad=(max-min)*.09;max+=pad;min-=pad;const minX=points[0].x,maxX=Math.max(minX+1,points.at(-1).x);const xx=x=>left+(x-minX)/(maxX-minX)*w,yy=y=>top+(max-y)/(max-min)*h;
    const root=svg('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':`${mode==='scores'?'Association score':mode==='change'?'Relative activation edit':'Intervention dose'} over ${points.length} generated tokens`});
    for(let i=0;i<4;i++){const y=top+h*i/3;root.append(svg('line',{x1:left,y1:y,x2:width-right,y2:y,stroke:'#e9eee9','stroke-width':1}));const label=svg('text',{x:left-7,y:y+3,'text-anchor':'end',fill:'#899b91','font-size':8,'font-family':'ui-monospace,monospace'});label.textContent=(max-(max-min)*i/3).toFixed(2);root.append(label);}
    for(const x of [minX,Math.round((minX+maxX)/2),maxX]){const label=svg('text',{x:xx(x),y:height-7,'text-anchor':'middle',fill:'#899b91','font-size':8,'font-family':'ui-monospace,monospace'});label.textContent=fmt(x);root.append(label);}
    const stride=Math.max(1,Math.ceil(points.length/Math.max(w,300)));
    for(const series of data.series){let d='',started=false;for(let i=0;i<points.length;i++){const y=series.values[i];if(!numeric(y)){started=false;continue;}if(i%stride&&i!==points.length-1)continue;d+=`${started?'L':'M'}${xx(points[i].x).toFixed(2)},${yy(y).toFixed(2)} `;started=true;}root.append(svg('path',{d,fill:'none',stroke:series.color,'stroke-width':series.dashed?1.25:1.7,'stroke-dasharray':series.dashed?'4 3':'none','stroke-linejoin':'round','stroke-linecap':'round',opacity:series.dashed?.6:.95}));if(points.length===1&&numeric(series.values[0]))root.append(svg('circle',{cx:xx(points[0].x),cy:yy(series.values[0]),r:2,fill:series.color}));const swatch=el('span',{class:`legend-swatch${series.dashed?' dashed':''}`});swatch.style.backgroundColor=series.color;if(series.dashed)swatch.style.borderColor=series.color;legend.append(el('span',{class:'legend-item'},swatch,series.name));}
    for(const event of events.filter(e=>['phase','phase_transition'].includes(e.type))){const position=first(event.generated_tokens,event.generation_index);if(!numeric(position)||position<minX||position>maxX)continue;const x=xx(position);root.append(svg('line',{x1:x,x2:x,y1:top,y2:height-bottom,stroke:'#bc9453','stroke-dasharray':'3 4',opacity:.65}));const label=svg('text',{x:Math.min(x+4,width-80),y:top+9,fill:'#98733c','font-size':8});label.textContent=event.phase||'phase change';root.append(label);}
    const marker=svg('line',{x1:0,x2:0,y1:top,y2:height-bottom,stroke:'#789787','stroke-dasharray':'2 3',visibility:'hidden'});root.append(marker);const tip=el('div',{class:'chart-tooltip hidden'});
    let selected=Math.min(points.length-1,Number(container.dataset.selection)||0),pinned=false;
    const inspectIndex=index=>{
      selected=Math.max(0,Math.min(points.length-1,index));container.dataset.selection=selected;const p=points[selected],event=p.event;
      marker.setAttribute('x1',xx(p.x));marker.setAttribute('x2',xx(p.x));marker.setAttribute('visibility','visible');
      root.setAttribute('aria-valuenow',selected+1);root.setAttribute('aria-valuetext',`Token ${p.x}, ${event.phase||'phase unavailable'}, ${p.text}`);
      tip.textContent=`Token ${p.x} · ${event.phase||'phase unavailable'}${p.text?' · '+JSON.stringify(p.text):''}\n`+`Site: ${p.dose.site?.location||'unavailable'} / layer ${p.dose.site?.layer??'unavailable'} · actor ${event.actor||'unavailable'} · revision ${event.control_revision??p.dose.control_revision??'unavailable'}\n`+data.series.map(series=>`${series.name}: ${numeric(series.values[selected])?series.values[selected].toFixed(3):'unavailable'}`).join('\n');tip.classList.remove('hidden');
    };
    const inspect=event=>{const rect=root.getBoundingClientRect(),x=(event.clientX-rect.left)*width/rect.width,target=minX+(x-left)/w*(maxX-minX);let index=0;for(let i=1;i<points.length;i++)if(Math.abs(points[i].x-target)<Math.abs(points[index].x-target))index=i;inspectIndex(index);};
    const selectText=()=>{const chat=container.id==='live-chart'?$('conversation'):$('replay-conversation'),key=points[selected].messageKey,node=chat.querySelector(`[data-key="${CSS.escape(key)}"]`);if(node){if(container.id==='live-chart')state.follow=false;node.classList.add('selected-token-message');chat.querySelectorAll('.selected-token-message').forEach(other=>{if(other!==node)other.classList.remove('selected-token-message');});if(node.tagName==='DETAILS')node.open=true;chat.scrollTop=node.offsetTop-chat.offsetTop;}};
    root.setAttribute('tabindex','0');root.setAttribute('role','slider');root.setAttribute('aria-valuemin','1');root.setAttribute('aria-valuemax',points.length);root.setAttribute('aria-valuenow',selected+1);root.setAttribute('aria-label',`${mode} graph. Left and right inspect raw tokens; Enter shows their message.`);
    root.addEventListener('focus',()=>inspectIndex(selected));root.addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','Home','End','Enter'].includes(event.key)){event.preventDefault();pinned=true;if(event.key==='Enter')selectText();else inspectIndex(event.key==='Home'?0:event.key==='End'?points.length-1:selected+(event.key==='ArrowRight'?1:-1));}});
    root.addEventListener('pointermove',event=>{if(!pinned)inspect(event);});root.addEventListener('pointerdown',event=>{inspect(event);pinned=true;selectText();});root.addEventListener('pointerleave',()=>{if(!pinned){marker.setAttribute('visibility','hidden');tip.classList.add('hidden');}});container.append(root,tip);
    const sampling=el('span',{class:'chart-sampling-note',text:stride>1?`Display: every ${stride}th token; gaps preserved. Inspector uses all ${points.length} raw samples.`:`Display: all ${points.length} raw token samples. Arrow keys inspect; Enter selects the message.`});legend.append(sampling);

  }
  function renderLiveChart(){drawChart($('live-chart'),$('live-legend'),liveEvents(),state.plot);const count=state.events.filter(e=>e.type==='token').length;text('telemetry-count',count?`${fmt(count)} observed tokens`:'Waiting for measurements');text('chart-note',state.plot==='scores'?'Separately fitted association probes at the edited and downstream layers. Readouts may overlap the intervention and move directly with it; these are not emotion percentages.':state.plot==='change'?'Norm of the actual edit relative to the pre-edit activation norm at the intervention site.':state.events.some(e=>e.dose?.schema_version===2)?'Version 2: solid lines show delivered pulse gains and total attenuation. Dashed lines show held baseline gains. Prefill positions are logged separately from the generated-token axis.':'Solid lines: delivered coefficients. Dashed lines: requested coefficients and pulse level. Baseline edits can persist when auxiliary effects are off.');}
  function renderReplayChart(){const run=state.runCache.get(state.replayId);drawChart($('replay-chart'),$('replay-legend'),replayEvents(run),'dose');}
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{renderLiveChart();if(state.replayId)renderReplayChart();},150);});
  if(window.OpiumDesigner)designer=window.OpiumDesigner.mount($('designer-root'),{
    command:async(name,payload)=>{const result=await command(name,payload);if(!result)throw new Error('The local service rejected the command. See its error notification.');return result;},
    isModelLoaded:()=>Boolean(state.snapshot?.worker?.model),
    onUse:(recipe,{mode}={})=>{state.designerRecipe=structuredClone(recipe);hydrateSession({config:recipe,mode:mode||'experiment'});renderDesignerSelection();openTab('live');toast('Version 2 recipe selected for the next session.');},
    listSaved:async kind=>(state.snapshot?.presets||[]).filter(row=>row.kind===kind),
    onLoad:async(kind,id)=>{const result=await command('load_preset',{kind,id});if(!result)throw new Error('Preset loading failed.');return result.asset.value;},
    onSave:async asset=>{const result=await command('save_preset',asset);if(!result)throw new Error('Preset saving failed.');return result.asset;}
  });
  openTab(location.hash.slice(1)||'live',false);renderLiveChart();refreshState().then(()=>pollEvents());setInterval(pollEvents,800);setInterval(refreshState,3500);
})();
