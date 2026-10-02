"""Isolated diagnostic generations from an immutable visible task boundary.

The prediction questions, answers and private mapping never feed back into the
parent session. Each context restores the same effect state and uses a separate
prespecified generation seed and diagnostic allowance.
"""
from copy import deepcopy
from pathlib import Path

from .checkpoints import restore, validate
from .discovery import prepare_branches, score_response, summarize_scores
from .storage import atomic_json, utc_now
from .runtime_controls import validate_runtime_controls


def validate_diagnostic_policy(checkpoint, policy):
    """Return controls to carry; reject unsupported state before allocating output.

    Yoke schedules need their own before-forward cursor semantics. Until a
    diagnostic adapter restores that driver, only explicitly sham diagnostics
    may branch from a yoked parent. Sham resets effects and discards controls.
    """
    checkpoint = validate(checkpoint)
    if policy not in {"carry_boundary_state", "sham"}:
        raise ValueError("Choose carry_boundary_state or sham diagnostic effects")
    if policy == "sham":
        return {}
    if "yoke_state" in checkpoint["session"]:
        raise ValueError("Yoked boundary state cannot be carried into diagnostics; choose explicit sham effects")
    return validate_runtime_controls(checkpoint["session"].get("runtime_controls"), recipe=checkpoint["session"]["config"])


def run_diagnostics(runtime, checkpoint, spec, answer_key, *, pair_id, calibration_dir,
                    out_dir, token_budget, turn_token_limit=256, effect_policy='carry_boundary_state', criterion_evidence=None,
                    emit=lambda event: None, should_stop=lambda: False):
    if type(token_budget) is not int or not 1 <= token_budget <= 100000:
        raise ValueError('Diagnostic token budget must be an integer from 1 to 100000')
    if type(turn_token_limit) is not int or not 1 <= turn_token_limit <= 16384:
        raise ValueError('Diagnostic per-context token limit must be an integer from 1 to 16384')
    checkpoint = validate(checkpoint)
    runtime_controls = validate_diagnostic_policy(checkpoint, effect_policy)
    branches = prepare_branches(checkpoint, spec, answer_key, pair_id=pair_id, criterion_evidence=criterion_evidence)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(target / 'diagnostic-design.json', dict(branches=branches, token_budget=token_budget,
        turn_token_limit=turn_token_limit, effect_policy=effect_policy,
        accounting='Separate diagnostic allowance; parent task budget and history unchanged', created_at=utc_now()))
    atomic_json(target / 'source-checkpoint.json', checkpoint)
    if criterion_evidence is not None:
        atomic_json(target / 'criterion-evidence-source.json', {'raw_utf8': criterion_evidence.decode('utf-8') if isinstance(criterion_evidence, bytes) else criterion_evidence})
    tokens, records, failure = 0, [], None
    for index, branch in enumerate(branches['branches']):
        if should_stop() or tokens >= token_budget:
            break
        session, effect, _, _ = restore(checkpoint)
        context = branch['observer_only']['context_id']
        if effect_policy == 'sham':
            effect.reset()
            effect.set_controls(enabled=False)
        # Only this subtree is model input. Observer metadata determines the
        # sampling stream and later scoring; no answer key reaches generation.
        visible = deepcopy(branch['model_input'])
        config = dict(session['config'], max_new_tokens=min(turn_token_limit, token_budget-tokens),
                      seed=branch['observer_only']['generation_seed'])
        local_tokens = 0
        emit(dict(type='diagnostic_start', actor='diagnostic', context_id=context, context_index=index,
                  parent_checkpoint_sha256=checkpoint['sha256'], effect_policy=effect_policy))
        def event(item):
            nonlocal tokens, local_tokens
            item = deepcopy(item)
            if item.get('type') == 'token':
                tokens += 1; local_tokens += 1; effect.advance(1)
                item.update(generation_index=(item.get('dose') or {}).get('generated_token_index', tokens-1),
                            diagnostic_context_token_index=local_tokens-1)
            emit(dict(item, actor='diagnostic', context_id=context, diagnostic_generated_tokens=tokens))
        try:
            result = runtime.generate(visible['messages'], visible['tools'], config, Path(calibration_dir),
                effect.snapshot, event, lambda: should_stop() or tokens >= token_budget,
                **({'runtime_controls': runtime_controls} if runtime_controls else {}))
            response = result.get('content', result.get('raw_text', ''))
            score = score_response(branch, response)
            interrupted = bool(should_stop()) or result.get('finish_reason') == 'stopped'
            partial = interrupted or bool(result.get('truncated')) or result.get('finish_reason') == 'length'
            record = dict(score=score, context_id=context, status='partial' if partial else 'complete',
                         generated_tokens=local_tokens, interrupted=interrupted,
                         generation_metadata={key:value for key,value in result.items() if key not in {'raw_text','reasoning','content','token_ids'}})
            records.append(record)
            emit(dict(type='message', role='assistant', actor='diagnostic', context_id=context,
                      content=response, reasoning=result.get('reasoning','')))
            emit(dict(type='diagnostic_score', actor='observer', context_id=context, score=score))
        except Exception as exc:
            records.append(dict(context_id=context, status='failed', error=str(exc), generated_tokens=local_tokens))
            atomic_json(target / 'diagnostic-records.json', records)
            failure = exc
            break
        atomic_json(target / 'diagnostic-records.json', records)
        if record['interrupted']:
            break
    scores = [record['score'] for record in records if record.get('status') == 'complete']
    summary = dict(kind='diagnostic', completed_contexts=sum(r.get('status') == 'complete' for r in records),
        partial_contexts=sum(r.get('status') == 'partial' for r in records),
        failed_contexts=sum(r.get('status') == 'failed' for r in records),
        observed_contexts=len(records), planned_contexts=len(branches['branches']),
        missing_contexts=len(branches['branches'])-len(records), tokens=tokens, token_budget=token_budget,
        effect_policy=effect_policy, parent_checkpoint_sha256=checkpoint['sha256'],
        termination='diagnostic_failed' if failure else 'stopped_by_user' if should_stop() or any(r.get('interrupted') for r in records) else 'diagnostic_token_budget' if len(records)<len(branches['branches']) else 'diagnostic_complete',
        scoring=summarize_scores(scores), primary_inclusion='complete contexts only; partial raw scores retained separately',
        partial_scoring=summarize_scores([r['score'] for r in records if r.get('status') == 'partial']),
        runtime_controls=runtime_controls, answer_feedback_to_parent=False)
    atomic_json(target / 'summary.json', summary)
    if failure is not None:
        raise failure
    return summary
