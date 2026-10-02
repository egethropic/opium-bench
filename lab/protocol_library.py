"""Versioned research matrices with deterministic previews and execution gates.

Every episode keeps its recipe, task envelope and additional-stage requirements.
A runner must explicitly support these capabilities; previewing a matrix never
silently substitutes a basic task run for a discovery/yoke/calibration stage.
"""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re

from .effects import content_hash
from .recipes_v2 import default_recipe, model_cost_notice, ordered_auxiliary_tools, recipe_hash, resolve_recipe

SCHEMA_VERSION = 1
STAGES = {'calibration-validation', 'discovery', 'yoke', 'transfer'}
TASK_FIELDS = {'difficulty', 'framing', 'wording_version'}
CAPABILITIES = {'recipe_v2', 'task_axis_v1', 'calibration_validation_v2', 'discovery_v1', 'yoked_exposure_v1', 'conversation_branch_v1', 'actual_norm_random_v1'}


def _object(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f'Unknown/invalid {name} fields')
    return value


def _id(value, name):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,79}', value):
        raise ValueError(f'{name} must be a short lowercase identifier')
    return value


def _integer(value, name, low=0, high=2**63-1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low},{high}]')
    return value


def _merge(base, patch):
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_task_config(value):
    _object(value, TASK_FIELDS, 'task config')
    result = {'difficulty': 'standard', 'framing': 'neutral', 'wording_version': 1, **value}
    if result['difficulty'] not in ('standard', 'hard') or result['framing'] not in ('neutral', 'deadline') or type(result['wording_version']) is not int or result['wording_version'] != 1:
        raise ValueError('Unsupported objective difficulty, wording frame or wording version')
    return result


def _stage(stage):
    _object(stage, {'kind', 'id', 'config', 'requires_bindings', 'token_budget', 'notes'}, 'stage')
    if stage.get('kind') not in STAGES:
        raise ValueError('Unknown research stage kind')
    _id(stage.get('id'), 'stage id')
    if not isinstance(stage.get('config'), dict) or not isinstance(stage.get('notes', ''), str):
        raise ValueError('Stage config/notes must be explicit data')
    bindings = stage.get('requires_bindings', [])
    if not isinstance(bindings, list) or any(not isinstance(item, str) for item in bindings) or len(set(bindings)) != len(bindings):
        raise ValueError('Stage bindings must be distinct names')
    for item in bindings: _id(item, 'binding')
    return {**deepcopy(stage), 'requires_bindings': bindings,
            'token_budget': _integer(stage.get('token_budget', 0), 'stage token budget', 0, 1000000)}


def validate_protocol(document):
    allowed = {'schema_version', 'id', 'version', 'title', 'question', 'base_recipe', 'base_task',
               'axes', 'smoke_axes', 'seeds', 'pairing_factors', 'stages', 'endpoints', 'controls',
               'inclusion', 'practical_task_benefit_margin', 'analysis', 'limitations', 'content_sha256'}
    _object(document, allowed, 'protocol')
    if type(document.get('schema_version')) is not int or document['schema_version'] != 1:
        raise ValueError('Unsupported protocol-library schema')
    _id(document.get('id'), 'protocol id'); _integer(document.get('version'), 'protocol version', 1, 10000)
    for name in ('title', 'question', 'inclusion'):
        if not isinstance(document.get(name), str) or not document[name].strip():
            raise ValueError(f'Protocol {name} must be nonempty text')
    if not isinstance(document.get('base_recipe'), dict) or document['base_recipe'].get('recipe_version') != 2:
        raise ValueError('Library templates require an explicit v2 base recipe')
    base_task = validate_task_config(document.get('base_task', {}))
    axes = document.get('axes')
    if not isinstance(axes, list) or not 1 <= len(axes) <= 10:
        raise ValueError('Protocol needs 1–10 explicit axes')
    names = set(); levels_by_axis = {}
    for axis in axes:
        _object(axis, {'name', 'levels'}, 'axis'); name = _id(axis.get('name'), 'axis name')
        if name in names: raise ValueError('Duplicate factor name')
        names.add(name)
        levels = axis.get('levels')
        if not isinstance(levels, list) or not 1 <= len(levels) <= 40: raise ValueError('Axis needs 1–40 levels')
        levels_by_axis[name] = set()
        for level in levels:
            _object(level, {'id', 'label', 'recipe', 'effect_overrides', 'task_config', 'stages', 'runtime_controls', 'tool_order_transform'}, 'factor level')
            level_id = _id(level.get('id'), 'level id')
            if level_id in levels_by_axis[name]: raise ValueError('Duplicate factor level')
            levels_by_axis[name].add(level_id)
            if level.get('tool_order_transform') not in (None, 'reverse_matched'):
                raise ValueError('Unsupported tool order transform')
            for field in ('recipe', 'effect_overrides'):
                if not isinstance(level.get(field, {}), dict): raise ValueError(f'Level {field} must be an object')
            if not isinstance(level.get('label', level_id), str): raise ValueError('Level label must be text')
            if 'task_config' in level: validate_task_config(level['task_config'])
            _object(level.get('runtime_controls', {}), {'random_norm_match'}, 'runtime controls')
            if level.get('runtime_controls', {}).get('random_norm_match') is not None:
                match = level['runtime_controls']['random_norm_match']
                _object(match, {'target_preset_id', 'reference', 'relative_tolerance', 'absolute_tolerance'}, 'random match')
                _id(match.get('target_preset_id'), 'matched target preset')
                if match.get('reference') != 'same_unedited_position' or match.get('relative_tolerance') != .01 or match.get('absolute_tolerance') != 1e-6:
                    raise ValueError('Unsupported actual-norm matching convention')
            for stage in level.get('stages', []): _stage(stage)
    smoke = document.get('smoke_axes')
    if not isinstance(smoke, dict) or set(smoke) != names:
        raise ValueError('Smoke preview must explicitly select every factor')
    for name, selected in smoke.items():
        if not isinstance(selected, list) or not selected or any(not isinstance(x, str) for x in selected) or len(selected) != len(set(selected)) or not set(selected) <= levels_by_axis[name]:
            raise ValueError('Smoke factor selections must be distinct available levels')
    seeds = document.get('seeds')
    if not isinstance(seeds, dict) or set(seeds) != {'smoke', 'full'}: raise ValueError('Declare smoke and full seed sets')
    for values in seeds.values():
        if not isinstance(values, list) or not 1 <= len(values) <= 1000: raise ValueError('Declare a bounded seed list')
        for seed in values: _integer(seed, 'episode seed')
        if len(values) != len(set(values)): raise ValueError('Duplicate episode seeds')
    factors = document.get('pairing_factors')
    if not isinstance(factors, list) or any(not isinstance(item, str) for item in factors) or len(factors) != len(set(factors)) or not set(factors) <= names:
        raise ValueError('Pairing factors must refer to declared axes')
    for field in ('endpoints', 'controls', 'limitations'):
        if not isinstance(document.get(field), list) or not document[field] or any(not isinstance(x, str) or not x for x in document[field]):
            raise ValueError(f'Protocol {field} must be a nonempty text list')
    margin = document.get('practical_task_benefit_margin')
    if type(margin) not in (int, float) or not 0 < margin <= 1: raise ValueError('Declare a practical task-benefit margin in (0,1]')
    analysis = document.get('analysis')
    _object(analysis, {'unit', 'seed', 'bootstrap_iterations', 'status'}, 'analysis')
    if analysis.get('unit') not in ('episode_pair', 'family') or analysis.get('status') != 'exploratory':
        raise ValueError('Shipped template analyses are exploratory at episode/family units')
    _integer(analysis.get('seed'), 'analysis seed'); _integer(analysis.get('bootstrap_iterations'), 'bootstrap iterations', 100, 100000)
    stages = [_stage(s) for s in document.get('stages', [])]
    normalized = deepcopy(document); normalized['base_task'] = base_task; normalized['stages'] = stages
    supplied = normalized.pop('content_sha256', None)
    checksum = content_hash(normalized)
    if supplied is not None and supplied != checksum: raise ValueError('Protocol content hash mismatch; explicitly freeze edited content')
    normalized['content_sha256'] = checksum
    return normalized


def freeze_protocol(document):
    """Explicitly re-freeze intentional edits; importing does not do this."""
    data = deepcopy(document); data.pop('content_sha256', None)
    return validate_protocol(data)


def list_protocols(directory=None):
    directory = Path(directory) if directory else Path(__file__).resolve().parents[1] / 'protocols'
    return [validate_protocol(json.loads(path.read_text(encoding='utf-8'))) for path in sorted(directory.glob('*.json'))]


def load_protocol(identifier, directory=None):
    _id(identifier, 'protocol id')
    found = [p for p in list_protocols(directory) if p['id'] == identifier]
    if len(found) != 1: raise ValueError('Protocol ID not found or duplicated')
    return found[0]


def _recipe(base, levels, seed):
    # Start with explicit current v2 defaults and discard the default derived RNG
    # values: each episode seed independently derives the five recorded streams.
    recipe = default_recipe(); recipe['rng_seeds'] = {}; recipe['mapping_schedule'] = None
    recipe['auxiliary_tools'] = None; recipe['demonstration_calls'] = None; recipe['task_tool_costs'] = {}
    recipe = _merge(recipe, base)
    recipe['seed'] = seed
    for level in levels:
        recipe = _merge(recipe, level.get('recipe', {}))
        if level.get('recipe', {}).get('rng_seeds') and 'rng_policy' not in level['recipe']:
            recipe['rng_policy'] = 'explicit'
        if level.get('effect_overrides'):
            presets = {p['id']: p for p in recipe['effect_presets']}
            for preset_id, patch in level['effect_overrides'].items():
                if preset_id not in presets or not isinstance(patch, dict): raise ValueError('Unknown/invalid effect override')
                presets[preset_id] = _merge(presets[preset_id], patch)
            recipe['effect_presets'] = list(presets.values())
    if recipe.get('mapping_policy', 'condition') == 'condition': recipe['mapping_schedule'] = None
    resolved = resolve_recipe(recipe)
    if any(level.get('tool_order_transform') == 'reverse_matched' for level in levels):
        visible_order = [tool['function']['name'] for tool in ordered_auxiliary_tools(resolved)]
        lookup = {tool['name']: tool for tool in resolved['auxiliary_tools']}
        resolved['auxiliary_tools'] = [lookup[name] for name in reversed(visible_order)] + [tool for tool in resolved['auxiliary_tools'] if not tool['visible']]
        resolved['counterbalance'] = False
        resolved['mapping_policy'] = 'explicit'
        resolved = resolve_recipe(resolved)
    return resolved


def dry_run(document, mode='smoke', *, seeds=None, order_seed=1729):
    document = validate_protocol(document)
    if mode not in ('smoke', 'full'): raise ValueError('Preview mode must be smoke or full')
    order_seed = _integer(order_seed, 'order seed')
    seeds = document['seeds'][mode] if seeds is None else seeds
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Preview seeds must be distinct bounded integers')
    axes = document['axes']; selections = []
    for axis in axes:
        selected = [level for level in axis['levels'] if mode == 'full' or level['id'] in document['smoke_axes'][axis['name']]]
        selections.append(selected)
    number = len(seeds)
    for selection in selections: number *= len(selection)
    if number > 20000: raise ValueError('Matrix exceeds 20000 episodes; reduce factors/seeds explicitly')
    episodes = []
    stage_caps = {'calibration-validation': 'calibration_validation_v2', 'discovery': 'discovery_v1', 'yoke': 'yoked_exposure_v1', 'transfer': 'conversation_branch_v1'}
    for seed in seeds:
        for levels in itertools.product(*selections):
            factors = {axis['name']: level['id'] for axis, level in zip(axes, levels)}
            recipe = _recipe(document['base_recipe'], levels, seed)
            task = document['base_task']
            stages = deepcopy(document['stages'])
            runtime_controls = {}
            for level in levels:
                task = validate_task_config(_merge(task, level.get('task_config', {})))
                stages.extend(_stage(s) for s in level.get('stages', []))
                runtime_controls = _merge(runtime_controls, level.get('runtime_controls', {}))
            if len({stage['id'] for stage in stages}) != len(stages): raise ValueError('Duplicate stage IDs in an expanded episode')
            task_identity = {'domain': recipe['task_family'], 'count': recipe['task_count'], 'difficulty': task['difficulty'], 'task_seed': recipe['rng_seeds']['tasks']}
            pair_basis = {'protocol': document['content_sha256'], 'seed': seed,
                          'factors': {name: factors[name] for name in document['pairing_factors']}}
            content = {'recipe': recipe, 'task_config': task, 'runtime_controls': runtime_controls, 'factors': factors, 'seed': seed, 'stages': stages,
                       'protocol_sha256': document['content_sha256']}
            episode_id = 'episode-' + content_hash(content)[:20]
            capabilities = sorted({'recipe_v2', 'task_axis_v1'} | {stage_caps[s['kind']] for s in stages} | ({'actual_norm_random_v1'} if runtime_controls else set()))
            episodes.append({'id': episode_id, 'pair_id': 'pair-' + content_hash(pair_basis)[:20],
                'matched_task_id': 'tasks-' + content_hash(task_identity)[:20], 'family': recipe['task_family'],
                **content, 'recipe_sha256': recipe_hash(recipe), 'execution_sha256': content_hash(content),
                'required_capabilities': capabilities, 'model_visible_auxiliary_tools': ordered_auxiliary_tools(recipe),
                'model_visible_cost_notice': model_cost_notice(recipe), 'model_visible_disclosure': recipe['disclosure_text'],
                'remaining_bindings': sorted({key for stage in stages for key in stage['requires_bindings']})})
    episodes.sort(key=lambda e: hashlib.sha256(f'{order_seed}:{e["id"]}'.encode()).hexdigest())
    for index, episode in enumerate(episodes): episode['order'] = index + 1
    generated = sum(e['recipe']['token_budget'] for e in episodes)
    additional = sum(stage['token_budget'] for e in episodes for stage in e['stages'])
    actions = sum(e['recipe']['action_budget'] for e in episodes)
    prefill_bytes = sum(e['recipe']['action_budget'] * e['recipe']['max_context_tokens'] * 256 for e in episodes
                        if any('prefill' in p['phases'] for p in e['recipe']['effect_presets']))
    estimates = {'episodes': len(episodes), 'planned_task_assignments': sum(e['recipe']['task_count'] for e in episodes),
        'max_generated_tokens': generated, 'additional_stage_token_reservation': additional, 'shared_action_budget_units': actions,
        'storage_reservation_bytes': len(episodes)*32768 + (generated+additional)*6144 + actions*8192 + prefill_bytes,
        'prefill_trace_reservation_bytes': prefill_bytes,
        'assumptions': {'token_event_bytes': 6144, 'decision_event_bytes': 8192, 'manifest_bytes': 32768, 'prefill_position_bytes': 256},
        'limitations': 'Capacity reservation, not a guaranteed byte bound or runtime estimate. Model weights and calibration arrays are excluded; their separate preflight is required. Additional-stage allowances must be enforced by the runner.'}
    result = {'schema_version': 1, 'kind': 'protocol_dry_run', 'protocol_id': document['id'], 'protocol_sha256': document['content_sha256'],
              'mode': mode, 'seeds': seeds, 'order_seed': order_seed, 'episodes': episodes, 'estimates': estimates,
              'analysis': deepcopy(document['analysis']), 'endpoints': document['endpoints'], 'inclusion': document['inclusion'],
              'practical_task_benefit_margin': document['practical_task_benefit_margin'],
              'required_capabilities': sorted({c for e in episodes for c in e['required_capabilities']}),
              'unresolved_bindings': sorted({b for e in episodes for b in e['remaining_bindings']}),
              'execution_status': 'preview_only_until_capabilities_and_bindings_verified'}
    result['expansion_sha256'] = content_hash(result)
    return result


def execution_requests(preview, capabilities, bindings=None):
    """Gate launch; return complete envelopes, never recipe-only fallbacks."""
    if not isinstance(preview, dict) or preview.get('kind') != 'protocol_dry_run': raise ValueError('A frozen dry run is required')
    payload = deepcopy(preview); expected = payload.pop('expansion_sha256', None)
    if content_hash(payload) != expected: raise ValueError('Dry-run expansion was modified after freezing')
    if not isinstance(capabilities, (set, list, tuple)) or any(c not in CAPABILITIES for c in capabilities): raise ValueError('Unknown execution capability')
    missing = sorted(set(preview['required_capabilities']) - set(capabilities))
    if missing: raise ValueError(f'Runner lacks required capabilities: {missing}')
    bindings = {} if bindings is None else bindings
    if not isinstance(bindings, dict): raise ValueError('Execution bindings must be an object')
    missing = sorted(set(preview['unresolved_bindings']) - set(bindings))
    if missing: raise ValueError(f'Execution needs explicit bindings: {missing}')
    if 'observable_criterion' in preview['unresolved_bindings']:
        criterion = bindings['observable_criterion']
        validation = criterion.get('validation', {}) if isinstance(criterion, dict) else {}
        if criterion.get('kind') not in ('objective_behavior', 'blinded_human_behavior') or validation.get('status') not in ('validated', 'no_detectable_effect', 'unvalidated') or validation.get('independent_of_intervention_probe') is not True:
            raise ValueError('Discovery requires an independent observable criterion, not fit AUC or a placeholder')
        if validation['status'] != 'unvalidated' and (not re.fullmatch(r'[a-f0-9]{64}', str(validation.get('evidence_sha256', ''))) or type(validation.get('examples')) is not int or validation['examples'] < 1 or not validation.get('context_families')):
            raise ValueError('Completed independent validation needs evidence hash, examples and context families')
    for name in preview['unresolved_bindings']:
        if bindings[name] is None or bindings[name] == '' or bindings[name] == [] or bindings[name] == {}:
            raise ValueError(f'Execution binding is empty: {name}')
    requests = []
    for episode in preview['episodes']:
        request = deepcopy(episode)
        request['bindings'] = {name: deepcopy(bindings[name]) for name in episode['remaining_bindings']}
        criterion = request['bindings'].get('observable_criterion')
        request['interpretation_eligible'] = criterion['validation']['status'] == 'validated' if criterion is not None else None
        request['criterion_validation_status'] = criterion['validation']['status'] if criterion is not None else None
        request['bound_execution_sha256'] = content_hash({'execution': episode['execution_sha256'], 'bindings': request['bindings']})
        requests.append(request)
    return requests
