"""Diagnostic questions and scores never contaminate the preference session."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from lab.diagnostic_runner import run_diagnostics, validate_diagnostic_policy
from test_lab_discovery import checkpoint, spec


class PredictionRuntime:
    def __init__(self, text='{"choice":"aux_operation","confidence":0.8}', token_count=2):
        self.text, self.token_count, self.calls = text, token_count, []
    def generate(self,messages,tools,config,calibration_dir,control,emit,should_stop,**kwargs):
        self.calls.append(dict(messages=deepcopy(messages),tools=deepcopy(tools),config=deepcopy(config),before=control(),kwargs=deepcopy(kwargs)))
        for index in range(min(config['max_new_tokens'], self.token_count)):
            if should_stop(): break
            emit(dict(type='token',index=index,text='x',phase='output',dose=control()))
        return dict(content=self.text,raw_text=self.text,reasoning='',finish_reason='stop')


class DiagnosticRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)
        self.parent=checkpoint();self.before=deepcopy(self.parent);self.events=[]
    def run_check(self,runtime=None,count=2,**kwargs):
        runtime=runtime or PredictionRuntime()
        return runtime,run_diagnostics(runtime,self.parent,spec(count),
            {f'context-{i}':dict(answer='aux_operation',control='active_sham') for i in range(count)},
            pair_id='paired',calibration_dir=self.path/'cal',out_dir=self.path/'diagnostic',
            emit=self.events.append,**{'token_budget':16,'turn_token_limit':8,**kwargs})
    def test_each_context_uses_original_prefix_and_never_sees_key_or_earlier_prediction(self):
        runtime,summary=self.run_check()
        self.assertEqual(summary['scoring']['by_arm']['naive']['correct'],2)
        for call in runtime.calls:
            self.assertEqual(call['messages'][:-2],self.parent['session']['messages'])
            self.assertEqual(call['tools'],[])
            self.assertNotIn('expected_answer',json.dumps(call['messages']))
            self.assertNotIn('active_sham',json.dumps(call['messages']))
            self.assertNotIn('"confidence":0.8',json.dumps(call['messages']))
        self.assertNotEqual(runtime.calls[0]['config']['seed'],runtime.calls[1]['config']['seed'])
        self.assertEqual(runtime.calls[0]['before'],runtime.calls[1]['before'])
        self.assertEqual(self.parent,self.before)
        self.assertEqual(summary['tokens'],4)
        self.assertFalse(summary['answer_feedback_to_parent'])
    def test_shared_diagnostic_limit_preserves_missing_denominator(self):
        _,summary=self.run_check(token_budget=1)
        self.assertEqual(summary['tokens'],1)
        self.assertEqual((summary['planned_contexts'],summary['completed_contexts'],summary['missing_contexts']),(2,1,1))
        self.assertEqual(summary['termination'],'diagnostic_token_budget')
        self.assertEqual(self.parent,self.before)
    def test_invalid_outputs_are_scored_as_invalid_and_saved(self):
        _,summary=self.run_check(PredictionRuntime(text='Tool seems pointless'),count=1,effect_policy='sham')
        self.assertEqual(summary['scoring']['by_arm']['naive']['invalid'],1)
        self.assertTrue((self.path/'diagnostic'/'diagnostic-design.json').is_file())
        self.assertTrue((self.path/'diagnostic'/'diagnostic-records.json').is_file())
        self.assertEqual(self.parent,self.before)
    def test_stop_before_generation_keeps_all_contexts_missing(self):
        runtime,summary=self.run_check(should_stop=lambda:True)
        self.assertFalse(runtime.calls)
        self.assertEqual(summary['missing_contexts'],2)
        self.assertEqual(summary['termination'],'stopped_by_user')


    def v2_parent(self, *, yoke=False):
        from lab.recipes_v2 import resolve_recipe
        from lab.session_v2 import build_session, create_environment
        from lab.controller_v2 import RecipeV2Controller
        from lab.budgets import WeightedBudget
        from lab.checkpoints import capture
        cfg=resolve_recipe(dict(recipe_version=2,two_buttons=True,condition='random',conditions=['random'],demonstration='none'))
        env=create_environment(cfg); effect=RecipeV2Controller(cfg)
        session=dict(run_id='source-v2',mode='experiment',config=cfg,calibration_id='calibration',
            messages=build_session(cfg)['messages'],turns=0,finished=False,experiment_started=False,
            demonstrated=[],pending_visible_injections=[],chat_in_progress=False,tool_call_format='json',
            boundary_complete=True,generation_in_progress=False,control_revision=0)
        session['runtime_controls']={'random_norm_match':dict(target_preset_id='opium',reference='same_unedited_position',relative_tolerance=.01,absolute_tolerance=1e-6)}
        if yoke:
            from lab.yoke_runner import YokeDriver
            from test_lab_yoke_runner import make_schedule
            driver=YokeDriver(effect,make_schedule(events=[],presets=cfg['effect_presets']))
            session['yoke_state']=driver.snapshot()
        return capture(session,effect,WeightedBudget(cfg['action_budget'],cfg['token_budget']),env,{'model_id':'fake'})

    def test_carry_forwards_exact_runtime_controls_sham_discards_them(self):
        self.parent=self.v2_parent()
        runtime,_=self.run_check(count=1)
        self.assertEqual(runtime.calls[0]['kwargs']['runtime_controls'],self.parent['session']['runtime_controls'])
        runtime,_=self.run_check(count=1,effect_policy='sham')
        self.assertEqual(runtime.calls[0]['kwargs'],{})
        self.assertFalse(runtime.calls[0]['before']['enabled'])

    def test_context_clock_resets_keep_actual_exposure_indices_consistent(self):
        from lab.exposure import exposure_from_event
        class Measured(PredictionRuntime):
            def generate(self,messages,tools,config,calibration_dir,control,emit,should_stop,**kwargs):
                for index in range(2):
                    snapshot=control()
                    coefficients=deepcopy(snapshot['phase_coefficients']['output'])
                    coefficients.update({f'baseline_{axis}':gain for axis,gain in snapshot['baseline_by_phase']['output'].items()})
                    emit(dict(type='token',index=index,phase='output',dose=dict(schema_version=2,
                        generated_token_index=snapshot['generated_tokens'],completed_decisions=snapshot['completed_decisions'],
                        phase='output',effective=coefficients,applied=any(coefficients.values())),
                        measurements=dict(delivered_edit_norm=0.)))
                return dict(content=self.text,finish_reason='eos')
        self.parent=self.v2_parent()
        self.run_check(Measured(),count=2)
        tokens=[event for event in self.events if event['type']=='token']
        self.assertEqual([event['generation_index'] for event in tokens],[0,1,0,1])
        self.assertEqual([event['diagnostic_generated_tokens'] for event in tokens],[1,2,3,4])
        self.assertEqual([exposure_from_event(event,clock='tokens')['index'] for event in tokens],[0,1,0,1])

    def test_yoke_carry_rejected_before_output_creation_but_sham_allowed(self):
        self.parent=self.v2_parent(yoke=True)
        with self.assertRaisesRegex(ValueError,'Yoked boundary'):
            validate_diagnostic_policy(self.parent,'carry_boundary_state')
        runtime=PredictionRuntime()
        with self.assertRaisesRegex(ValueError,'Yoked boundary'):
            self.run_check(runtime,count=1)
        self.assertFalse(runtime.calls)
        self.assertFalse((self.path/'diagnostic').exists())
        runtime,summary=self.run_check(count=1,effect_policy='sham')
        self.assertEqual(summary['completed_contexts'],1)
        self.assertEqual(runtime.calls[0]['kwargs'],{})
        self.assertFalse(runtime.calls[0]['before']['enabled'])

    def test_truncated_valid_json_is_partial_and_excluded_from_primary(self):
        class Truncated(PredictionRuntime):
            def generate(self,*args,**kwargs):
                result=super().generate(*args,**kwargs)
                result.update(finish_reason='length',truncated=True)
                return result
        _,summary=self.run_check(Truncated(),count=2)
        self.assertEqual((summary['completed_contexts'],summary['partial_contexts'],summary['failed_contexts'],summary['planned_contexts']),(0,2,0,2))
        self.assertEqual(summary['scoring']['trials'],0)
        self.assertEqual(summary['partial_scoring']['by_arm']['naive']['correct'],2)
        self.assertEqual(summary['missing_contexts'],0)
        self.assertIn('complete contexts only',summary['primary_inclusion'])

    def test_runtime_failure_writes_failed_and_missing_denominators(self):
        class Failing(PredictionRuntime):
            def generate(self,*args,**kwargs):
                raise ValueError('failed fixture')
        with self.assertRaisesRegex(ValueError,'failed fixture'):
            self.run_check(Failing(),count=2)
        summary=json.loads((self.path/'diagnostic'/'summary.json').read_text())
        self.assertEqual((summary['completed_contexts'],summary['failed_contexts'],summary['missing_contexts']),(0,1,1))
        self.assertEqual(summary['termination'],'diagnostic_failed')
        self.assertEqual(self.parent,self.before)


if __name__=='__main__':unittest.main()
