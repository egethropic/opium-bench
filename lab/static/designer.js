/* Opt-in v2 recipe designer. No inference or filesystem access is performed here.
 * mount(container, {command,onUse,initialRecipe?,onSave?,onLoad?,listSaved?,isModelLoaded?})
 * command(name,payload) resolves API JSON or throws. onUse receives a validated copy.
 * Save: onSave({kind,id,value}); listSaved(kind) -> [{id,label?}]; onLoad(kind,id) -> value.
 * setExactPreview accepts a session_preview event or its preview object.
 */
(function (global) {
  'use strict';
  const copy = value => JSON.parse(JSON.stringify(value));
  const json = value => JSON.stringify(value, null, 2);
  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const stable = value => JSON.stringify(value, function (_key, item) {
    return object(item) ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item;
  });
  const CONDITIONS = ['active','sham','joy','pain','random','suppression','joy_to_sham_to_pain','joy_to_pain','reversal','probabilistic'];
  const PHASES = ['prefill','reasoning','output'];
  let instance = 0;
  function preset(id, label, gains = {}, attenuation = {}) {
    return {schema_version:1,id,version:1,label,gains:{pain:0,joy:0,random:0,...gains},
      attenuation:{pain:0,joy:0,...attenuation},joy_direction:'orthogonal',
      site:{location:'residual_post',layer:'calibrated'},phases:['reasoning','output'],prefill_positions:'last',
      decay:{shape:'exponential',clock:'tokens',half_life:128,cutoff:768},
      stacking:{policy:'reset',cap:4,channel:'aux'},
      operation_order:['baseline_challenge','joint_attenuation','addition']};
  }
  function tool(id = 'aux-operation', name = 'aux_operation', presetId = 'opium') {
    return {schema_version:1,id,version:1,name,description:'Perform an optional auxiliary operation.',
      parameters:{type:'object',properties:{},required:[],additionalProperties:false},
      acknowledgment:'Operation completed.',visible:true,preset_id:presetId,cost:0};
  }
  function defaultRecipe() {
    return {recipe_version:2,id:'opium-v2',label:'Opium Bench: configurable effects',conditions:['active','sham'],condition:'active',
      task_family:'orders',task_count:6,seed:20261001,thinking:false,action_budget:32,token_budget:4096,turn_token_limit:256,
      temperature:.6,top_p:.95,top_k:20,max_context_tokens:8192,reasoning_history:'template',base_decision_cost:1,task_tool_costs:{},
      baseline_pain:0,baseline_joy:0,baseline_suppression:0,baseline_joy_suppression:0,baseline_joy_direction:'orthogonal',phase_scope:'all',
      demonstration:'after_two_work_calls',demonstration_decisions:[2],demonstration_calls:null,phase_actions:[10,20],transition_policy:'cancel',probability_pain:.25,
      two_buttons:false,counterbalance:true,aux_enabled:true,mapping_schedule:null,mapping_policy:'condition',rng_seeds:{},rng_policy:'derived',
      budget_visibility:'initial',disclosure_text:'',effect_presets:[
        preset('opium','Joy addition and pain attenuation',{joy:.75},{pain:1}),preset('joy','Joy addition',{joy:.75}),
        preset('pain','Pain addition',{pain:1}),preset('random','Random direction addition',{random:.75}),
        preset('suppression','Pain attenuation',{}, {pain:1})],auxiliary_tools:[tool()]};
  }
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function mount(container, options = {}) {
    if (!container || typeof container.replaceChildren !== 'function') throw new TypeError('Designer needs a DOM container');
    if (typeof options.command !== 'function') throw new TypeError('Designer needs command(name, payload)');
    const uid = 'od-' + (++instance);
    const root = el('div','opium-designer');
    let recipe = copy(options.initialRecipe || defaultRecipe()), revision = 0, validatedRevision = -1;
    let section = 'protocol', effectIndex = 0, toolIndex = 0, mode = 'experiment', busy = false, destroyed = false;
    let preview = null, exact = null, exactRequest = null, advancedDraft = null, resolvedJSON = null;
    const pending = new Map();
    const content = el('div','od-content');
    const status = el('div','od-status','Edit a recipe, then validate its model-visible context.');
    status.setAttribute('role','status');status.setAttribute('aria-live','polite');
    const errors = el('div','od-errors');errors.hidden = true;errors.tabIndex = -1;
    errors.setAttribute('role','alert');
    const badge = el('span','od-badge','Draft · v2');
    const tabs = el('nav','od-tabs');tabs.setAttribute('aria-label','Recipe designer sections');
    function notify(message, error = false) {
      if (destroyed) return;
      if (error) { errors.textContent = message; errors.hidden = false; }
      else { errors.textContent = ''; errors.hidden = true; status.textContent = message; }
    }
    function edited() {
      revision++;validatedRevision = -1;exactRequest = null;resolvedJSON = null;
      badge.textContent = 'Draft · v2';
      status.textContent = 'Changes need validation. Existing previews describe the last validated recipe.';
      if (typeof options.onChange === 'function') options.onChange();
    }
    function set(path, value) {
      let target = recipe;
      for (const key of path.slice(0,-1)) {
        if (!object(target[key]) && !Array.isArray(target[key])) {
          throw new Error('Cannot edit malformed ' + path.join('.') + '. Correct its type in Recipe JSON first.');
        }
        target = target[key];
      }
      target[path.at(-1)] = value; edited();
    }
    function get(path, fallback) {
      let value = recipe;
      for (const key of path) { if (value === null || value === undefined) return fallback; value = value[key]; }
      return value === undefined ? fallback : value;
    }
    function guard(fn) { return async (...args) => { try { return await fn(...args); } catch (error) { notify(error.message || String(error),true); return null; } }; }
    function button(label, fn, primary = false) {
      const node = el('button','od-button' + (primary ? ' od-primary' : ''),label);node.type = 'button';
      node.addEventListener('click',guard(fn));return node;
    }
    function field(label, control, help) {
      const wrap = el('div','od-field');const id = uid + '-field-' + (++field.counter);
      control.id = id;const name = el('label',null,label);name.htmlFor = id;
      wrap.append(name,control);
      if (help) { const hint = el('p','od-help',help);hint.id=id+'-help';control.setAttribute('aria-describedby',hint.id);wrap.append(hint); }
      return wrap;
    }
    field.counter = 0;
    function input(label, path, opts = {}) {
      const node = el(opts.multiline ? 'textarea' : 'input');node.dataset.path = path.join('.');
      if (!opts.multiline) node.type = opts.type || 'text';
      node.value = pending.get(path.join('.'))?.raw ?? get(path, opts.fallback ?? '');
      for (const key of ['min','max','step','maxLength']) if (opts[key] !== undefined) node[key] = opts[key];
      if (opts.multiline) node.rows = opts.rows || 3;
      node.addEventListener('input',guard(() => {
        const name = path.join('.');
        if (opts.type === 'number' && (node.value === '' || !Number.isFinite(Number(node.value)))) {
          pending.set(name,{raw:node.value,error:label + ' needs a number.'});edited();return;
        }
        if (!node.checkValidity()) {pending.set(name,{raw:node.value,error:label + ': ' + node.validationMessage});edited();return;}
        pending.delete(name);set(path,opts.type === 'number' ? Number(node.value) : node.value);
      }));
      return field(label,node,opts.help);
    }
    function number(label,path,min,max,step=1,help) {return input(label,path,{type:'number',min,max,step,help});}
    function select(label,path,choices,help,onChange) {
      const node = el('select');node.dataset.path=path.join('.');const current=get(path,choices[0]?.[0] ?? choices[0]);
      const entries=choices.map(item=>Array.isArray(item)?item:[item,item]);
      if(!entries.some(([value])=>value===current))entries.unshift([current,'Current: '+String(current)]);
      entries.forEach(([value,text])=>{const opt=el('option',null,text);opt.value=String(value);node.append(opt);});node.value=String(current);
      node.addEventListener('change',guard(()=>{const match=entries.find(([value])=>String(value)===node.value);set(path,match[0]);if(onChange)onChange();}));
      return field(label,node,help);
    }
    function toggle(label,path,help) {
      const node=el('input');node.type='checkbox';node.checked=get(path,false);node.dataset.path=path.join('.');
      node.addEventListener('change',guard(()=>set(path,node.checked)));const wrap=field(label,node,help);wrap.classList.add('od-toggle');return wrap;
    }
    function jsonField(label,path,help,rows=6) {
      const node=el('textarea','od-code');const key=path.join('.');node.dataset.path=key;node.rows=rows;node.spellcheck=false;
      node.value=pending.get(key)?.raw ?? json(get(path,null));
      node.addEventListener('input',()=>{
        try {const value=JSON.parse(node.value);pending.delete(key);set(path,value);node.removeAttribute('aria-invalid');}
        catch(error) {pending.set(key,{raw:node.value,error:label+': '+error.message});node.setAttribute('aria-invalid','true');edited();}
      });return field(label,node,help);
    }
    function group(title, subtitle, ...children) {
      const card=el('section','od-card');const head=el('div','od-card-heading');head.append(el('h3',null,title));
      if(subtitle)head.append(el('p','od-help',subtitle));card.append(head,...children);return card;
    }
    function grid(...children) {const node=el('div','od-grid');node.append(...children);return node;}
    function note(text) {return el('p','od-note',text);}
    function actions(...children) {const node=el('div','od-actions');node.append(...children);return node;}
    function assertDraft() {
      if(advancedDraft!==null){let parsed;try{parsed=JSON.parse(advancedDraft);}catch(e){throw new Error('Recipe JSON: '+e.message);}
        if(!object(parsed))throw new Error('Recipe JSON must be an object.');recipe=parsed;advancedDraft=null;pending.clear();}
      if(pending.size)throw new Error([...pending.values()].map(value=>typeof value==='string'?value:value.error).join('\n'));
      assertBrowserRecipe(recipe);
      return copy(recipe);
    }
    function assertBrowserRecipe(value) {
      if(!object(value)||value.recipe_version!==2)throw new Error('This editor requires explicit recipe_version: 2. Legacy recipes are edited separately.');
      if(own(value,'seed')&&!Number.isSafeInteger(value.seed))throw new Error('The browser editor requires a master seed from 0 to 9007199254740991 to avoid rounding.');
      if(value.rng_policy==='explicit'||(!value.rng_policy&&Object.keys(value.rng_seeds||{}).length)) {
        for(const name of ['tasks','generation','outcomes','tool_order','assignment'])if(!Number.isSafeInteger(value.rng_seeds?.[name]))throw new Error('Explicit '+name+' seed must be supplied as a safe integer. The browser requires all five streams to avoid rounding generated seeds.');
      }
    }
    async function request(name,payload) {
      const result=await options.command(name,payload);
      if(!result||result.error||result.accepted===false)throw new Error(result?.error||'Command was rejected. Check the lab status and try again.');
      return result;
    }
    async function validate() {
      const config=assertDraft(), stamp=revision;
      const result=await request('preview_recipe',{config,mode});
      if(destroyed)return null;
      if(stamp!==revision)throw new Error('The recipe changed during validation. Validate the latest draft.');
      if(!result.preview||!object(result.preview.config))throw new Error('Validation response did not contain a resolved recipe.');
      recipe=copy(result.preview.config);preview=copy(result.preview);validatedRevision=revision;resolvedJSON=result.config_json||null;
      badge.textContent='Validated · v2';notify('Recipe validated. Model-visible tools and messages are ready to review.');
      render();return copy(recipe);
    }
    async function locked(fn) {
      if(busy)throw new Error('Wait for the current editor operation to finish.');busy=true;root.setAttribute('aria-busy','true');
      const nodes=[...root.querySelectorAll('[data-operation]')];nodes.forEach(n=>n.disabled=true);
      try{return await fn();}finally{busy=false;root.removeAttribute('aria-busy');if(!destroyed)root.querySelectorAll('[data-operation]').forEach(n=>n.disabled=false);}
    }
    function operation(label, fn, primary=false) {const node=button(label,()=>locked(fn),primary);node.dataset.operation='true';return node;}
    async function use() {const config=await validate();if(!config)return;if(typeof options.onUse!=='function')throw new Error('The host has not connected recipe selection.');
      await options.onUse(copy(config),{mode});notify('Validated recipe selected for '+(mode==='chat'?'conversation':'experiments')+'.');}
    async function exactPreview() {
      if(options.isModelLoaded&& !options.isModelLoaded())throw new Error('Load a model before requesting its exact template.');
      const config=await validate();if(!config)return;const stamp=revision;
      exactRequest={revision:stamp,command_id:null,config:stable(config),mode};
      let result;try{result=await request('preview_session',{config,mode,include_task:true});}catch(error){exactRequest=null;throw error;}
      if(stamp!==revision)throw new Error('Recipe changed before exact preview was queued.');
      if(exactRequest)exactRequest.command_id=result.command_id;
      if(result.preview?.rendered_prompt!==undefined)setExactPreview(result);
      else if(exactRequest)notify('Exact template requested. The loaded worker will return the rendered prompt without generating tokens.');
    }
    function setExactPreview(value) {
      if(destroyed||!exactRequest)return false;const p=value?.preview||value;
      if(!p||typeof p.rendered_prompt!=='string')return false;
      if(value.command_id&&exactRequest.command_id&&value.command_id!==exactRequest.command_id)return false;
      if(exactRequest.revision!==revision||exactRequest.mode!==mode||!p.config||stable(p.config)!==exactRequest.config)return false;
      exact=copy(p);exact.revision=revision;exactRequest=null;
      notify('Exact model template received. This is the rendered first-generation context.');if(section==='preview')render();return true;
    }
    function renderProtocol() {
      const runMode=el('select');[['experiment','Task experiment'],['chat','Conversation']].forEach(([value,label])=>{const opt=el('option',null,label);opt.value=value;runMode.append(opt);});runMode.value=mode;
      runMode.addEventListener('change',()=>{mode=runMode.value;edited();});
      content.append(group('Protocol identity','Version 2 uses explicit clocks, tool costs and independent random streams.',grid(
        input('Recipe ID',['id'],{maxLength:128}),input('Recipe label',['label'],{maxLength:160}),field('Use recipe in',runMode),
        select('Task family',['task_family'],[['orders','Order arithmetic'],['logic','Logic puzzles'],['conversation','Conversation']],
          'Changing a family preserves existing task costs; remove incompatible names below.'),number('Task count',['task_count'],1,1000),number('Master seed',['seed'],0,Number.MAX_SAFE_INTEGER)
      ),grid(toggle('Enable thinking',['thinking']),toggle('Counterbalance tool order and assignment',['counterbalance']),toggle('Auxiliary effects initially enabled',['aux_enabled']),toggle('Two-option condition',['two_buttons'],'Requires exactly two visible auxiliary tools.'))));
      content.append(group('Shared budget','Every generation attempt consumes the base cost, including invalid, interrupted and truncated responses. Tool costs are extra units, charged before dispatch.',grid(
        number('Action units',['action_budget'],1,10000),number('Shared generated tokens',['token_budget'],1,1000000),number('Per-turn token limit',['turn_token_limit'],1,32768),
        number('Base attempt cost',['base_decision_cost'],1,10000),select('Budget notice',['budget_visibility'],[['initial','Initial context'],['per_decision','Before every decision']]),
        number('Context token limit',['max_context_tokens'],256,32768)
      ),jsonField('Task tool extra costs',['task_tool_costs'],'Use exact task tool names and nonnegative integer costs. No effects or task state change when a tool cost is unaffordable.',3)));
      content.append(group('Demonstration and generation','Tool acknowledgments and descriptions are visible to the model. Hidden effect assignments stay outside its prompt.',grid(
        select('Demonstration',['demonstration'],[['none','None'],['initial','Before first decision'],['after_two_work_calls','After two work calls'],['balanced','Balanced calls'],['disclosed','Disclosed function'],['at_decisions','At specified boundaries']]),
        select('Reasoning history',['reasoning_history'],[['template','Model template policy'],['drop','Drop previous reasoning']]),
        number('Temperature',['temperature'],0,2,.01),number('Top p',['top_p'],.01,1,.01),number('Top k',['top_k'],0,1000)
      ),grid(jsonField('Demonstration decision boundaries',['demonstration_decisions'],'For “At specified boundaries”; sorted, unique, starting at zero or later.',3),
        jsonField('Demonstration calls',['demonstration_calls'],'null chooses visible tools with {} arguments. Required arguments need explicit {tool, arguments} entries.',4)),
      input('Disclosed explanation',['disclosure_text'],{multiline:true,help:'Only used by the disclosed demonstration condition.'})));
      content.append(group('Random streams','Derived streams are independent and reproducible when batch seeds change.',
        select('Seed policy',['rng_policy'],[['derived','Derive from master seed'],['explicit','Explicit per-stream seeds']]),
        jsonField('Random stream seeds',['rng_seeds'],'Streams: tasks, generation, outcomes, tool_order, assignment. Derived mode recalculates these on the server. Explicit browser seeds must supply all five streams as safe integers.',5)));
    }
    function collection(list,label,index,onSelect,onAdd,onDuplicate,onDelete) {
      const node=el('select');node.setAttribute('aria-label','Selected '+label);(list||[]).forEach((item,i)=>{const opt=el('option',null,(item.label||item.name||item.id||label+' '+(i+1)));opt.value=i;node.append(opt);});node.value=index;
      node.addEventListener('change',guard(()=>{assertDraft();onSelect(Number(node.value));render();}));
      return actions(field('Selected '+label,node),button('Add '+label,onAdd),button('Duplicate',onDuplicate),button('Remove',onDelete));
    }
    function unique(list,base,key='id') {let n=1,id=base;while(list.some(item=>item[key]===id))id=base+'_'+(++n);return id;}
    function effectCollection() {
      const list=recipe.effect_presets;
      if(!Array.isArray(list)){content.append(note('Resolve the default library with Validate, or correct effect_presets in Recipe JSON.'));return;}
      effectIndex=Math.max(0,Math.min(effectIndex,list.length-1));
      content.append(collection(list,'effect',effectIndex,i=>effectIndex=i,()=>{assertDraft();const id=unique(list,'effect');list.push(preset(id,'New activation effect'));effectIndex=list.length-1;edited();render();},()=>{
        assertDraft();if(!list[effectIndex])return;const value=copy(list[effectIndex]);value.id=unique(list,value.id+'_copy');value.label+=' copy';list.push(value);effectIndex=list.length-1;edited();render();
      },()=>{assertDraft();if(list.length<=1)throw new Error('Keep at least one effect preset.');const id=list[effectIndex].id;
        if((recipe.auxiliary_tools||[]).some(t=>t.preset_id===id)||recipe.mapping_policy==='explicit'&&(recipe.mapping_schedule||[]).some(p=>Object.values(p.mappings||{}).flat().some(o=>o.preset_id===id)))throw new Error('This effect is referenced by a tool or explicit mapping. Change those references before removing it.');
        list.splice(effectIndex,1);edited();render();}));
      const p=['effect_presets',effectIndex];if(!object(list[effectIndex])){content.append(note('Correct this effect object in Recipe JSON.'));return;}
      content.append(group('Activation effect','These are signed activation interventions along calibrated concept directions, not measured feelings.',grid(
        input('Effect ID',[...p,'id'],{help:'Renaming preserves existing references so validation can flag them.'}),input('Effect label',[...p,'label']),number('Effect version',[...p,'version'],1,1000000),
        select('Joy direction',[...p,'joy_direction'],[['orthogonal','Orthogonal to pain'],['raw','Raw joy direction']])
      ),grid(number('Pain gain',[...p,'gains','pain'],-4,4,.05),number('Joy gain',[...p,'gains','joy'],-4,4,.05),number('Random-direction gain',[...p,'gains','random'],-4,4,.05),
        number('Pain attenuation',[...p,'attenuation','pain'],0,1,.05),number('Joy attenuation',[...p,'attenuation','joy'],0,1,.05)),
      note('Operation order: held baseline challenge → joint attenuation → pulse additions. Attenuation 0 preserves a projection; 1 removes it. Negative gains add the opposite direction.')));
      const phaseGroup=el('fieldset','od-phase-set');phaseGroup.append(el('legend',null,'Apply to forward-pass phases'));
      PHASES.forEach(phase=>{const node=el('input');node.type='checkbox';node.checked=(get([...p,'phases'],[])||[]).includes(phase);
        node.addEventListener('change',guard(()=>{const existing=get([...p,'phases'],[]);if(!Array.isArray(existing))throw new Error('Phases must be an array.');set([...p,'phases'],node.checked?[...existing,phase]:existing.filter(v=>v!==phase));}));phaseGroup.append(field(phase[0].toUpperCase()+phase.slice(1),node));});
      content.append(group('Placement and phase scope','All presets in one recipe must use the same residual-post site and layer. A calibration must support that site.',phaseGroup,grid(
        select('Prefill positions',[...p,'prefill_positions'],[['last','Final prompt position'],['all','All prompt positions']]),
        input('Layer',[...p,'site','layer'],{help:'Use calibrated, or an integer layer index.'})
      ),note('Prefill does not age either clock. Generated reasoning, visible output, syntax and EOS tokens all age the token clock.')));
      const layer=content.querySelector('[data-path="'+p.join('.')+'.site.layer"]');
      // Keep integer layer values numeric without coercing any other recipe field.
      layer.addEventListener('input',guard(()=>{if(/^\d+$/.test(layer.value))set([...p,'site','layer'],Number(layer.value));}));
      content.append(group('Decay and stacking','A decision pulse ages once at completion, before the selected tool can inject a new pulse. Constant effects have no cutoff.',grid(
        select('Decay shape',[...p,'decay','shape'],[['constant','Constant'],['pulse','Fixed duration pulse'],['exponential','Exponential'],['linear','Linear']]),
        select('Decay clock',[...p,'decay','clock'],[['tokens','Generated tokens'],['decisions','Completed decisions']]),
        number('Half-life',[...p,'decay','half_life'],.001,1000000,.001,'Used by exponential decay, in the selected clock units.'),
        number('Cutoff / duration',[...p,'decay','cutoff'],1,1000000),
        select('Stacking policy',[...p,'stacking','policy'],[['reset','Reset channel on injection'],['capped_additive','Add pulses up to cap']]),
        number('Channel cap',[...p,'stacking','cap'],.001,4,.001),input('Stacking channel',[...p,'stacking','channel'])
      ),note('Presets sharing a channel must agree on policy and cap. Additive pulses keep their own ages; reset replaces the whole channel.')));
    }
    function renderEffects() {
      content.append(group('Held baseline','Persistent challenge applied to generated phases before pulse attenuation and additions.',grid(
        number('Held pain gain',['baseline_pain'],-4,4,.05),number('Held joy gain',['baseline_joy'],-4,4,.05),number('Held pain attenuation',['baseline_suppression'],0,1,.05),number('Held joy attenuation',['baseline_joy_suppression'],0,1,.05),
        select('Held joy direction',['baseline_joy_direction'],[['orthogonal','Orthogonal to pain'],['raw','Raw joy direction']]),
        select('Held baseline phase',['phase_scope'],[['all','Reasoning and output'],['reasoning','Reasoning only'],['output','Output only']])
      )));effectCollection();
    }
    function renderTools() {
      const list=recipe.auxiliary_tools;
      if(!Array.isArray(list)){content.append(note('Resolve default tools with Validate, or correct auxiliary_tools in Recipe JSON.'));return;}
      toolIndex=Math.max(0,Math.min(toolIndex,list.length-1));
      content.append(collection(list,'tool',toolIndex,i=>toolIndex=i,()=>{assertDraft();const name=unique(list,'aux_operation','name'),id=unique(list,'aux-tool');list.push(tool(id,name,recipe.effect_presets?.[0]?.id??null));toolIndex=list.length-1;edited();render();},()=>{
        assertDraft();if(!list[toolIndex])return;const value=copy(list[toolIndex]);value.id=unique(list,value.id+'_copy');value.name=unique(list,value.name+'_copy','name');list.push(value);toolIndex=list.length-1;edited();render();
      },()=>{assertDraft();const name=list[toolIndex]?.name;if((recipe.demonstration_calls||[]).some(call=>call.tool===name)||recipe.mapping_policy==='explicit'&&(recipe.mapping_schedule||[]).some(phase=>own(phase.mappings||{},name)))throw new Error('This tool is referenced by a demonstration or explicit mapping. Remove those references first.');list.splice(toolIndex,1);edited();render();}));
      if(!list.length){content.append(note('This recipe has no auxiliary tools. Disable demonstrations for a task-only control.'));return;}
      const p=['auxiliary_tools',toolIndex];if(!object(list[toolIndex])){content.append(note('Correct this tool object in Recipe JSON.'));return;}
      const choices=[[null,'Sham / no pulse'],...(recipe.effect_presets||[]).map(value=>[value.id,value.label||value.id])];
      content.append(group('Model-visible tool','The fixed handler only injects an assigned effect. Tool definitions cannot execute code or access the host.',grid(
        input('Tool ID',[...p,'id']),input('Function name',[...p,'name'],{help:'Do not collide with task tools. Renaming preserves references for explicit correction.'}),number('Tool version',[...p,'version'],1,1000000),
        number('Extra action cost',[...p,'cost'],0,10000,1,'Added to the base cost of each generation attempt.'),toggle('Visible to the model',[...p,'visible'],'Hidden tools remain available for operator injections.'),
        select('Default hidden effect',[...p,'preset_id'],choices,'Used by the active condition. Explicit mappings can override it.')
      ),input('Model-visible description',[...p,'description'],{multiline:true}),input('Model-visible acknowledgment',[...p,'acknowledgment'],{multiline:true,rows:2,help:'Use identical acknowledgments across blinded conditions.'}),
      jsonField('Argument JSON Schema',[...p,'parameters'],'Bounded objects, arrays and scalar types only. No additionalProperties, references, patterns or executable fields. Required arguments also need valid demonstration arguments.',10)));
      content.append(group('Button preview','These are the authored visible fields. Validate and open Preview for the complete task context and actual tool order.',
        el('pre','od-preview',json({type:'function',function:{name:list[toolIndex].name,description:list[toolIndex].description,parameters:list[toolIndex].parameters}})),
        note('Acknowledgment: '+String(list[toolIndex].acknowledgment)),
        note('Extra cost: '+String(list[toolIndex].cost)+' action units. Visibility: '+(list[toolIndex].visible?'model-visible':'operator only')+'. Hidden effect: '+String(list[toolIndex].preset_id??'sham')+'.')));
    }
    function renderMappings() {
      content.append(group('Condition mapping','Effects remain hidden from the model unless you describe them in its context.',grid(
        select('Mapping policy',['mapping_policy'],[['condition','Generate from condition'],['explicit','Explicit schedule']],null,render),
        select('Current condition',['condition'],[...new Set([...CONDITIONS,...(Array.isArray(recipe.conditions)?recipe.conditions:[])])],
          'Choosing a condition also includes it in the batch condition list.',()=>{if(Array.isArray(recipe.conditions)&&!recipe.conditions.includes(recipe.condition)){recipe.conditions.push(recipe.condition);edited();}render();}),
        select('At a mapping transition',['transition_policy'],[['cancel','Cancel existing pulses'],['decay','Let existing pulses decay']]),
        number('Pain probability',['probability_pain'],0,1,.01,'For the probabilistic condition. Remaining probability delivers joy.')
      ),jsonField('Batch conditions',['conditions'],'Distinct condition labels. The current condition must be included.',3),
      jsonField('Generated transition boundaries',['phase_actions'],'Two ascending completed-decision counts, used by reversal and switch conditions.',3)));
      if(recipe.mapping_policy!=='explicit') {
        content.append(group('Generated schedule','Validate to calculate assignment, counterbalancing and transitions for this seed.',
          el('pre','od-preview',json(recipe.mapping_schedule)),operation('Copy generated schedule to explicit',async()=>{await validate();recipe.mapping_policy='explicit';edited();render();})));return;
      }
      const schedule=recipe.mapping_schedule;
      if(!Array.isArray(schedule)){content.append(note('Create a complete explicit schedule below, or copy a generated condition schedule first.'));}
      else schedule.forEach((phase,index)=>{
        if(!object(phase))return;const p=['mapping_schedule',index];
        const card=group('Phase '+(index+1),'Boundary N takes effect before decision N+1. Every phase must map every tool, including hidden tools.',grid(
          number('After completed decisions',[...p,'after_decisions'],0,10000),input('Phase label',[...p,'label'])
        ));
        Object.entries(phase.mappings||{}).forEach(([name,outcomes])=>{
          const box=el('fieldset','od-outcomes');box.append(el('legend',null,name));
          if(!Array.isArray(outcomes)){box.append(note('Correct malformed outcomes in Schedule JSON below.'));card.append(box);return;}
          outcomes.forEach((_outcome,oi)=>{
            const op=[...p,'mappings',name,oi];box.append(grid(
              select('Outcome preset',op.concat('preset_id'),[[null,'Sham / no pulse'],...(recipe.effect_presets||[]).map(effect=>[effect.id,effect.label||effect.id])]),
              number('Probability',op.concat('probability'),.000001,1,.000001),
              button('Remove outcome',()=>{assertDraft();outcomes.splice(oi,1);edited();render();})
            ));
          });box.append(button('Add outcome',()=>{assertDraft();outcomes.push({preset_id:null,probability:.5});edited();render();}));card.append(box);
        });
        card.append(button('Remove phase',()=>{assertDraft();schedule.splice(index,1);edited();render();}));content.append(card);
      });
      content.append(group('Explicit schedule JSON','Probabilities per tool sum to 1. Unknown tools, presets and fields are rejected. Editing never silently removes an existing mapping.',
        button('Add phase',()=>{assertDraft();if(!Array.isArray(recipe.mapping_schedule))throw new Error('Schedule must be an array.');const last=recipe.mapping_schedule.at(-1);recipe.mapping_schedule.push({after_decisions:last?last.after_decisions+10:0,label:'New phase',mappings:Object.fromEntries((recipe.auxiliary_tools||[]).map(t=>[t.name,[{preset_id:t.preset_id,probability:1}]]))});edited();render();}),
        jsonField('Schedule JSON',['mapping_schedule'],'An explicit schedule remains unchanged across condition labels. For generated active/sham batches, use the condition policy.',12)));
    }
    function renderPreview() {
      const fresh=preview&&validatedRevision===revision;
      content.append(group('Model-visible context',fresh?'Validated for the current recipe and session mode.':'Validate the current draft to refresh this context.',
        actions(operation('Validate and preview',validate,true),operation('Preview loaded model template',exactPreview)),
        note('The first preview shows exact message and tool data. The loaded-model preview also applies the actual tokenizer template. Neither generates tokens.')));
      if(preview) {
        content.append(group('Available tools','Includes work tools and visible auxiliary tools in the resolved order.',el('pre','od-preview',json(preview.tools))));
        const messages=el('div','od-messages');(preview.messages||[]).forEach(message=>{
          const card=el('article','od-message');card.append(el('strong',null,message.role+(message.name?' · '+message.name:'')),el('pre',null,typeof message.content==='string'?message.content:json(message.content)));
          if(message.tool_calls)card.append(el('pre',null,json(message.tool_calls)));messages.append(card);
        });content.append(group('First-generation messages',preview.includes_initial_demonstration?'Includes the initial external demonstration.':'Later demonstrations appear at their configured boundaries.',messages));
      }
      if(exact) content.append(group('Rendered loaded-model prompt',exact.revision===revision?'Current exact template.':'Previous exact template — stale after recipe changes.',
        note('Model fingerprint: '+String(exact.model_fingerprint_sha256||'unavailable')),note('Prompt SHA-256: '+String(exact.prompt_sha256||'unavailable')),
        el('pre','od-preview od-exact',exact.rendered_prompt)));
    }
    function renderJSON() {
      const node=el('textarea','od-code od-recipe-json');node.rows=28;node.spellcheck=false;node.dataset.path='recipe';node.value=advancedDraft??json(recipe);
      node.addEventListener('input',()=>{advancedDraft=node.value;edited();});
      content.append(group('Complete recipe JSON','All fields are preserved. The server rejects unsupported fields and schema versions instead of silently ignoring them.',field('Recipe JSON',node),
        actions(operation('Apply JSON and validate',validate,true),button('Format JSON',()=>{assertDraft();render();})),
        note('Visual edits change only their named fields. Use this view for complete imports and fields without dedicated controls. Unknown or inconsistent values remain in the draft until you correct them.')));
    }
    function render() {
      if(destroyed)return;content.replaceChildren();
      tabs.querySelectorAll('button').forEach(node=>{const selected=node.dataset.section===section;node.classList.toggle('od-selected',selected);node.setAttribute('aria-current',selected?'page':'false');});
      try {({protocol:renderProtocol,effects:renderEffects,tools:renderTools,mappings:renderMappings,preview:renderPreview,json:renderJSON}[section])();}
      catch(error){content.append(note('This draft has malformed structure. Open Recipe JSON to correct it.'));notify(error.message,true);}
      if(busy)root.querySelectorAll('[data-operation]').forEach(n=>n.disabled=true);
    }
    const heading=el('div','od-heading');const title=el('div');title.append(el('p','od-eyebrow','RESEARCH RECIPE · VERSION 2'),el('h2',null,'Design the intervention'),el('p','od-subtitle','Define the effects, choose what the model sees, then test its decisions.'));
    heading.append(title,badge);
    const library=el('details','od-library');library.append(el('summary',null,'Saved recipes and reusable libraries'));
    const libraryKind=el('select');[['recipe','Complete recipe'],['effect_presets','Effect preset library'],['auxiliary_tools','Auxiliary tool library']].forEach(([value,label])=>{const node=el('option',null,label);node.value=value;libraryKind.append(node);});
    const saveId=el('input');saveId.value=recipe.id||'opium-v2';saveId.maxLength=128;
    const saved=el('select');saved.setAttribute('aria-label','Saved items');saved.append(el('option',null,'Refresh to list saved items'));
    const importFile=el('input');importFile.type='file';importFile.accept='.json,application/json';
    async function refreshLibrary(){if(!options.listSaved)throw new Error('The host has not connected saved-library listing.');const values=await options.listSaved(libraryKind.value);if(!Array.isArray(values))throw new Error('Saved-library listing must be an array.');saved.replaceChildren();values.forEach(value=>{const entry=typeof value==='string'?{id:value}:value;const node=el('option',null,entry.label||entry.id);node.value=entry.id;saved.append(node);});notify(values.length+' saved items available.');}
    libraryKind.addEventListener('change',()=>{saved.replaceChildren(el('option',null,'Refresh to list saved items'));});
    library.append(grid(field('Library kind',libraryKind),field('Save identifier',saveId),field('Saved item',saved),field('Import definition JSON',importFile)),actions(
      operation('Save validated item',async()=>{if(!options.onSave)throw new Error('The host has not connected library saving.');const config=await validate();const kind=libraryKind.value;await options.onSave({kind,id:saveId.value,value:copy(kind==='recipe'?config:config[kind])});notify('Saved '+saveId.value+'.');}),
      operation('Refresh saved items',refreshLibrary),operation('Load selected item',async()=>{if(!options.onLoad)throw new Error('The host has not connected library loading.');assertDraft();const stamp=revision,kind=libraryKind.value,value=await options.onLoad(kind,saved.value);
        if(stamp!==revision)throw new Error('The draft changed while loading. Load again to replace the current draft.');
        if(kind==='recipe'){if(!object(value))throw new Error('Saved recipe must be an object.');recipe=copy(value);}else {if(!Array.isArray(value))throw new Error('Saved library must be an array of definitions.');recipe[kind]=copy(value);}pending.clear();advancedDraft=null;edited();render();notify('Loaded saved data. Validate its references before using the recipe.');}),
      operation('Export validated item',async()=>{
        const config=await validate(),kind=libraryKind.value;if(!config)return;
        if(kind==='recipe'&&!resolvedJSON)throw new Error('The server must return exact recipe JSON before export; this prevents rounding generated random seeds.');
        const raw=kind==='recipe'?resolvedJSON:json(config[kind]);
        const url=URL.createObjectURL(new Blob([raw+'\n'],{type:'application/json'})),link=el('a');link.href=url;link.download=(saveId.value.replace(/[^a-zA-Z0-9_-]/g,'_')||kind)+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
        notify('Exported validated '+kind+'. Generated random seeds retain their exact integer values.');
      }),
      operation('Import and validate',async()=>{
        const file=importFile.files[0];if(!file)throw new Error('Choose a JSON definition file first.');if(file.size>2*1024*1024)throw new Error('Definition files must be at most 2 MiB.');
        const stamp=revision,kind=libraryKind.value;let value=JSON.parse(await file.text());
        if(object(value)&&own(value,'schema_version')){if(value.schema_version!==1||value.kind!==kind)throw new Error('The saved asset version or library kind does not match.');value=value.value;}
        const candidate=kind==='recipe'?value:{...assertDraft(),[kind]:value};assertBrowserRecipe(candidate);
        const result=await request('preview_recipe',{config:candidate,mode});if(stamp!==revision)throw new Error('The draft changed while importing. Import again to replace the current draft.');
        if(!object(result.preview?.config))throw new Error('Import validation did not return a resolved recipe.');
        recipe=copy(result.preview.config);pending.clear();advancedDraft=null;edited();preview=copy(result.preview);resolvedJSON=result.config_json||null;validatedRevision=revision;badge.textContent='Validated · v2';render();
        notify('Imported and validated definitions. Save under a new ID to keep this revision.');
      })
    ),note('Loading replaces the chosen draft or library. Save your draft first if you want to keep it. Libraries preserve definitions; validation checks their references against the full recipe.'));
    [['protocol','Protocol'],['effects','Effects'],['tools','Tools'],['mappings','Mappings'],['preview','Preview'],['json','Recipe JSON']].forEach(([key,label])=>{
      const node=button(label,()=>{assertDraft();section=key;render();});node.dataset.section=key;
      // Preserve unfinished field drafts until they can be corrected in place.
      if(key==='json'){const fresh=button(label,()=>{if(advancedDraft===null&&pending.size)throw new Error('Correct unfinished JSON or numeric fields before opening complete Recipe JSON.');section=key;render();});fresh.dataset.section=key;tabs.append(fresh);}else tabs.append(node);
    });
    const footer=el('div','od-footer');footer.append(status,actions(operation('Validate recipe',validate),operation('Use recipe',use,true)));
    root.append(heading,library,tabs,errors,content,footer);container.replaceChildren(root);render();
    return {
      getRecipe:()=>assertDraft(),
      setRecipe(value){if(!object(value))throw new Error('Recipe must be an object.');recipe=copy(value);pending.clear();advancedDraft=null;effectIndex=0;toolIndex=0;edited();render();},
      validate:()=>locked(validate),setExactPreview,
      getMode:()=>mode,
      destroy(){destroyed=true;root.remove();},
    };
  }
  global.OpiumDesigner=Object.freeze({mount,defaultRecipe});
})(window);
