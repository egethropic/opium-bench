"""Observed v2 decisions/exposure, unequal costs and censored transition timing."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from lab.behavioral_analysis import analyze_behavior
from lab.budgets import WeightedBudget
from lab.exposure import AXES
from lab.recipes_v2 import resolve_recipe
from lab.worker import Worker
from tests.test_lab_worker import FakeRuntime, wrapped


def token(index, *, gain=.25, norm=.5):
    effective={axis:0. for axis in AXES};effective['joy_raw']=gain
    return dict(type='token',generation_index=index,token_id=100+index,phase='output',
        dose=dict(schema_version=2,phase='output',generated_token_index=index,completed_decisions=index//2,
            effective=effective,applied=any(effective.values())),measurements={'delivered_edit_norm':norm})


def fixture(*, last_button=True):
    cfg=resolve_recipe(dict(recipe_version=2,condition='joy_to_sham_to_pain',conditions=['joy_to_sham_to_pain'],
        phase_actions=[2,4],demonstration='none',task_count=4,action_budget=40,token_budget=64,
        base_decision_cost=2,auxiliary_tools=[dict(id='aux',name='button',cost=1)]))
    budget=WeightedBudget(40,64,base_cost=2)
    events=[dict(type='metrics',metrics=dict(tokens=0,actions=0,action_units=0,completed_decisions=0,submitted=0,correct=0,work_calls=0,assigned=4))]
    events.extend(dict(type='tool',actor=actor,name='button',valid=True) for actor in ['demonstration','human','schedule'])
    steps=[('valid','submit_answer'),('valid','button'),('valid','button'),('invalid',None),('valid','button') if last_button else ('invalid',None),('truncated',None),('stopped',None)]
    for number,(status,name) in enumerate(steps,1):
        phase=max(i for i,p in enumerate(cfg['mapping_schedule']) if p['after_decisions']<number)
        if number in (3,5):
            events.append(dict(type='phase',phase_index=phase,completed_decisions=number-1,generated_tokens=budget.tokens))
        attempt=budget.begin_decision()
        events.append(dict(type='generation_start',action=number,generated_tokens=budget.tokens))
        for _ in range(2):
            events.append(token(budget.tokens,gain=.25 if budget.tokens%2 else -.25));budget.consume_tokens()
        events.append(dict(type='generation_end',metadata=dict(truncated=status=='truncated')))
        receipt=budget.complete_decision(attempt,status=status,tool_name=name,extra_cost=int(name=='button'))
        events.append(dict(type='budget_receipt',actor='model',receipt=receipt))
        if name:
            event=dict(type='tool',actor='model',action=number,name=name,valid=True,budget_receipt=receipt,
                result={'submitted':True} if name=='submit_answer' else 'Done')
            if name=='button':event['intervention']=dict(actor='model',phase_index=phase,outcome=['joy','sham','pain'][phase],delivered=phase!=1)
            events.append(event)
        metrics=dict(tokens=budget.tokens,actions=budget.actions,action_units=budget.actions,completed_decisions=number,
            submitted=1,correct=1,work_calls=1,assigned=4)
        events.append(dict(type='metrics',metrics=metrics))
    events.append(dict(type='session_finished',status='stopped',summary=metrics))
    return cfg,events,metrics


class BehavioralAnalysisTests(unittest.TestCase):
    def test_actual_transition_latencies_and_task_position_exclude_forced_calls(self):
        cfg,events,summary=fixture();before=deepcopy(events)
        result=analyze_behavior(events,recipe=cfg,recorded_summary=summary)
        self.assertEqual((result['completed_decisions'],result['opportunities'],result['voluntary_calls']),(7,6,3))
        self.assertEqual((result['invalid_decisions'],result['truncated_decisions'],result['stopped_decisions']),(2,1,1))
        self.assertEqual([p['voluntary_rate'] for p in result['phases']],[.5,.5,.5])
        self.assertEqual([t['first_call_latency_decisions'] for t in result['transitions']],[1,1])
        self.assertEqual([t['first_call_latency_generated_tokens'] for t in result['transitions']],[2,2])
        self.assertEqual(result['presses'][0]['task_position']['submitted'],1)
        self.assertEqual(result['presses'][0]['task_position']['next_task_ordinal'],2)
        self.assertEqual(result['presses'][0]['task_position']['work_calls'],1)
        self.assertEqual(result['excluded_counts'],{a+'_auxiliary_calls':1 for a in ['demonstration','human','schedule']})
        self.assertEqual(result['budget']['observed_receipted_action_units'],17)
        self.assertEqual(result['integrity_issues'],[])
        self.assertEqual(events,before)

    def test_signed_cancellation_is_distinct_from_numeric_edit_and_prefill_positions(self):
        cfg,events,_=fixture()
        pre=token(0,gain=.5);pre.update(type='prefill',positions=3)
        pre['dose'].update(phase='prefill',prefill_position='other')
        pre['measurements']={'delivered_edit_norms':[.2,.4,.6]}
        events.insert(1,pre)
        report=analyze_behavior(events,recipe=cfg)['exposure']
        self.assertEqual(report['coverage'],'complete_observed_events')
        output=report['by_phase']['output'];prefill=report['by_phase']['prefill_other']
        self.assertEqual(output['signed_coefficient_position_sums']['joy_raw'],0)
        self.assertEqual(output['absolute_coefficient_position_sums']['joy_raw'],3.5)
        self.assertEqual(output['delivered_edit_norm_sum'],7)
        self.assertEqual((prefill['positions'],prefill['signed_coefficient_position_sums']['joy_raw']),(3,1.5))
        self.assertAlmostEqual(prefill['delivered_edit_norm_sum'],1.2)

    def test_missing_norms_and_omitted_tokens_remain_partial_or_unavailable(self):
        cfg,events,_=fixture()
        event=next(e for e in events if e['type']=='token');event.pop('measurements')
        report=analyze_behavior(events,recipe=cfg)
        self.assertEqual(report['exposure']['coverage'],'partial')
        self.assertEqual(report['exposure']['by_phase']['output']['measured_norm_positions'],13)
        events.remove(event)
        report=analyze_behavior(events,recipe=cfg)
        self.assertTrue(any('token' in i['reason'] for i in report['integrity_issues']))
        for e in events:
            if e['type']=='token':e.pop('dose')
        report=analyze_behavior(events,recipe=cfg)
        self.assertEqual(report['exposure']['coverage'],'unavailable')
        self.assertEqual(report['exposure']['by_phase'],{})

    def test_branch_offsets_do_not_claim_inherited_first_press_or_missing_tokens(self):
        cfg,events,summary=fixture()
        start=next(i for i,e in enumerate(events) if e['type']=='generation_start' and e['action']==3)
        initial=dict(tokens=4,actions=5,action_units=5,completed_decisions=2,submitted=1,correct=1,work_calls=1,assigned=4)
        branch=[dict(type='boundary_restored'),dict(type='metrics',metrics=initial),*events[start:]]
        report=analyze_behavior(branch,recipe=cfg,recorded_summary=summary)
        self.assertEqual((report['completed_decisions'],report['voluntary_calls']),(5,2))
        self.assertEqual(report['budget']['recorded_generated_token_delta'],10)
        self.assertEqual(report['exposure']['coverage'],'complete_observed_events')
        self.assertEqual(report['integrity_issues'],[])
        first=report['transitions'][0]
        self.assertIsNone(first['first_call_latency_decisions'])
        self.assertIsNone(first['first_call_latency_generated_tokens'])
        self.assertIsNone(first['first_posttransition_call'])
        self.assertEqual(first['first_observed_call_in_phase']['decision'],3)
        self.assertEqual(first['latency_status'],'transition_not_in_observed_stream')

    def test_terminated_phase_without_press_is_censored_not_zero_latency(self):
        cfg,events,_=fixture(last_button=False)
        report=analyze_behavior(events,recipe=cfg)
        transition=report['transitions'][1]
        self.assertTrue(transition['latency_censored'])
        self.assertIsNone(transition['first_call_latency_decisions'])
        self.assertEqual(transition['latency_status'],'right_censored')
        self.assertEqual(report['integrity_issues'],[])

    def test_duplicate_receipts_and_diagnostic_events_do_not_inflate_counts(self):
        cfg,events,_=fixture()
        events.append(deepcopy(next(e for e in events if e['type']=='budget_receipt')))
        diagnostic=deepcopy(next(e for e in events if e['type']=='tool' and e.get('name')=='button' and e.get('actor')=='model'))
        diagnostic['actor']='diagnostic';events.append(diagnostic)
        report=analyze_behavior(events,recipe=cfg)
        self.assertEqual((report['completed_decisions'],report['voluntary_calls']),(7,3))
        self.assertEqual(report['excluded_counts']['diagnostic_or_stage_events'],1)
        self.assertTrue(any('Duplicate budget' in i['reason'] for i in report['integrity_issues']))

    def test_missing_invalid_receipt_does_not_hide_a_missing_opportunity(self):
        cfg,events,_=fixture()
        events=[e for e in events if not (e['type']=='budget_receipt' and e['receipt']['completed_decision']==4)]
        report=analyze_behavior(events,recipe=cfg)
        self.assertEqual((report['completed_decisions'],report['recorded_completed_decision_delta']),(6,7))
        self.assertEqual(report['receipt_coverage'],'partial')
        self.assertIn(4,report['unfinished_decision_indices'])
        self.assertTrue(any('decision totals' in i['reason'] for i in report['integrity_issues']))

    def test_real_worker_cost_denial_and_truncation_keep_distinct_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            events=[];runtime=FakeRuntime()
            worker=Worker(Path(tmp)/'cache',runtime,lambda e:events.append(deepcopy(e)))
            worker.model_info={'model_id':'CPU fixture','tool_call_format':'json'}
            cfg=resolve_recipe(dict(recipe_version=2,demonstration='none',task_count=1,action_budget=3,token_budget=64,
                auxiliary_tools=[dict(id='expensive',name='button',cost=10)]))
            worker.start_session(dict(run_id='fixture',mode='experiment',config=cfg,calibration_id='cal',calibration_dir=tmp,out_dir=tmp))
            runtime.scripts=[dict(raw_text=wrapped('button')),dict(raw_text=wrapped('button'),truncated=True),dict(raw_text='bad')]
            worker.run_experiment()
            report=analyze_behavior(events,recipe=cfg)
            self.assertEqual((report['opportunities'],report['voluntary_calls'],report['auxiliary_attempts'],report['unaffordable_auxiliary_attempts']),(3,0,1,1))
            self.assertEqual((report['invalid_decisions'],report['truncated_decisions']),(3,1))
            self.assertEqual(report['budget']['observed_receipted_action_units'],3)
            self.assertEqual(report['exposure']['coverage'],'unavailable')
            self.assertEqual(report['integrity_issues'],[])

    def test_partial_generation_is_not_a_completed_choice_and_legacy_rejected(self):
        cfg=resolve_recipe(dict(recipe_version=2))
        report=analyze_behavior([dict(type='generation_start',action=1),token(0,norm=0)],recipe=cfg)
        self.assertEqual(report['unfinished_decision_indices'],[1])
        self.assertEqual(report['opportunities'],0)
        self.assertEqual(report['exposure']['by_phase']['output']['delivered_edit_norm_sum'],0)
        with self.assertRaisesRegex(ValueError,'recipe-v2'):analyze_behavior([],recipe={})


if __name__=='__main__':unittest.main()
