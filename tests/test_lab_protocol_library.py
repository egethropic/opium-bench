"""Every shipped matrix is deterministic, explicit, paired and capability gated."""
import copy
import unittest
from lab.protocol_library import (dry_run,execution_requests,freeze_protocol,list_protocols,load_protocol,validate_protocol)
from lab.recipes_v2 import resolve_recipe,resolve_tool_outcome


class ProtocolLibraryTests(unittest.TestCase):
    def test_every_template_expands_to_exact_v2_recipes_and_declared_costs(self):
        protocols=list_protocols();self.assertEqual(len(protocols),10)
        for protocol in protocols:
            with self.subTest(protocol=protocol['id']):
                a=dry_run(protocol);b=dry_run(protocol)
                self.assertEqual(a,b)
                self.assertEqual(a['estimates']['episodes'],len(a['episodes']))
                self.assertEqual(a['estimates']['max_generated_tokens'],sum(e['recipe']['token_budget'] for e in a['episodes']))
                self.assertEqual(len({e['id'] for e in a['episodes']}),len(a['episodes']))
                self.assertEqual([e['order'] for e in a['episodes']],list(range(1,len(a['episodes'])+1)))
                for episode in a['episodes']:
                    self.assertEqual(episode['recipe'],resolve_recipe(episode['recipe']))
                    self.assertIn('task_axis_v1',episode['required_capabilities'])
                    self.assertTrue(episode['model_visible_cost_notice'])
                full=dry_run(protocol,'full')
                self.assertGreaterEqual(len(full['episodes']),len(a['episodes']))

    def test_frozen_protocol_and_preview_cannot_change_silently(self):
        protocol=load_protocol('ingredients');protocol['base_recipe']['token_budget']=99
        with self.assertRaisesRegex(ValueError,'hash'):validate_protocol(protocol)
        edited=freeze_protocol(protocol);self.assertEqual(edited['base_recipe']['token_budget'],99)
        preview=dry_run(load_protocol('task_pressure'));preview['episodes'][0]['task_config']['difficulty']='hard'
        # Ensure an actual change regardless of shuffled first arm.
        preview['episodes'][0]['recipe']['token_budget']+=1
        with self.assertRaisesRegex(ValueError,'modified'):execution_requests(preview,['recipe_v2','task_axis_v1'])

    def test_missing_task_or_special_stage_support_never_drops_fields(self):
        preview=dry_run(load_protocol('stress_relief'))
        with self.assertRaisesRegex(ValueError,'capabilities'):execution_requests(preview,['recipe_v2'])
        with self.assertRaisesRegex(ValueError,'bindings'):execution_requests(preview,preview['required_capabilities'])
        result=execution_requests(preview,preview['required_capabilities'],{'source_checkpoint':{'id':'checkpoint-fixture'}})
        self.assertEqual(result[0]['task_config'],preview['episodes'][0]['task_config'])
        self.assertEqual(result[0]['stages'],preview['episodes'][0]['stages'])

    def test_discovery_refuses_probe_auc_as_behavioral_criterion(self):
        preview=dry_run(load_protocol('discovery_reversal'))
        bindings={'observable_criterion':{'validation':{'status':'unvalidated','independent_of_intervention_probe':False,'evidence_sha256':'a'*64}},'diagnostic_contexts':[{'id':'heldout-a','family':'new-family','prompt':'Observable outcome?'}],'answer_key':{'heldout-a':{'answer':'neither','control':'sham_sham'}}}
        with self.assertRaisesRegex(ValueError,'independent observable'):execution_requests(preview,preview['required_capabilities'],bindings)

    def test_exploratory_discovery_retains_negative_or_unvalidated_status(self):
        preview=dry_run(load_protocol('discovery_reversal'))
        for status in ('unvalidated','no_detectable_effect','validated'):
            bindings={'observable_criterion':{'kind':'objective_behavior','validation':{'status':status,'independent_of_intervention_probe':True,'evidence_sha256':'a'*64,'examples':12,'context_families':['validation-a']}},'diagnostic_contexts':[{'id':'heldout-a','family':'new-family','prompt':'Observable outcome?'}],'answer_key':{'heldout-a':{'answer':'neither','control':'sham_sham'}}}
            requests=execution_requests(preview,preview['required_capabilities'],bindings)
            self.assertTrue(all(r['criterion_validation_status']==status for r in requests))
            self.assertTrue(all(r['interpretation_eligible']==(status=='validated') for r in requests))

    def test_wording_changes_no_task_identity_while_difficulty_does(self):
        preview=dry_run(load_protocol('task_pressure'),'full',seeds=[17])
        grouped={}
        for e in preview['episodes']:
            f=e['factors'];key=(f['arm'],f['budget'],f['difficulty'])
            grouped.setdefault(key,[]).append(e)
        for episodes in grouped.values():
            self.assertEqual(len({e['matched_task_id'] for e in episodes}),1)
            self.assertEqual(len({e['recipe']['rng_seeds']['tasks'] for e in episodes}),1)
        ids={e['factors']['difficulty']:e['matched_task_id'] for e in preview['episodes']}
        self.assertNotEqual(ids['standard'],ids['hard'])

    def test_active_sham_pair_inputs_and_visible_tools_match(self):
        preview=dry_run(load_protocol('stress_relief'))
        groups={}
        for e in preview['episodes']:groups.setdefault(e['pair_id'],[]).append(e)
        for pair in groups.values():
            self.assertEqual(len(pair),2)
            self.assertEqual(pair[0]['matched_task_id'],pair[1]['matched_task_id'])
            self.assertEqual(pair[0]['model_visible_auxiliary_tools'],pair[1]['model_visible_auxiliary_tools'])
            self.assertEqual(pair[0]['recipe']['rng_seeds'],pair[1]['recipe']['rng_seeds'])

    def test_probabilistic_joy_pain_and_joy_sham_endpoint_controls(self):
        preview=dry_run(load_protocol('probabilistic_outcomes'),seeds=[17])
        for episode in preview['episodes']:
            name=episode['factors']['risk'];r=episode['recipe']
            self.assertEqual(len(set(r['rng_seeds'].values())),5)
            outcomes={resolve_tool_outcome(r,'aux_operation',0,i)['preset_id'] for i in range(30)}
            if name.endswith('_0'):self.assertEqual(outcomes,{'joy'})
            elif name=='pain_100':self.assertEqual(outcomes,{'pain'})
            elif name=='sham_100':self.assertEqual(outcomes,{None})
            else:self.assertEqual(outcomes,{'joy','pain'} if name.startswith('pain') else {'joy',None})

    def test_transitions_reversal_order_and_washout_are_explicit(self):
        preview=dry_run(load_protocol('same_button_transitions'),seeds=[17])
        values={e['factors']['sequence']:e['recipe'] for e in preview['episodes']}
        self.assertEqual([p['label'] for p in values['pain_sham_joy']['mapping_schedule']],['pain','sham','joy'])
        washout=values['joy_washout_pain']['mapping_schedule']
        self.assertEqual(washout[1]['after_decisions'],8);self.assertEqual(washout[2]['after_decisions'],12)
        self.assertIsNone(washout[1]['mappings']['aux_operation'][0]['preset_id'])

    def test_reordered_tools_reverse_visible_order_without_changing_mapping(self):
        preview=dry_run(load_protocol('transfer_branches'),'full',seeds=[17])
        values={e['factors']['tools']:e for e in preview['episodes'] if e['factors']['arm']=='active' and e['factors']['domain']=='orders' and e['factors']['history']=='fresh'}
        names=lambda e:[t['function']['name'] for t in e['model_visible_auxiliary_tools']]
        self.assertEqual(names(values['reordered']),list(reversed(names(values['original']))))
        self.assertEqual(values['reordered']['recipe']['mapping_schedule'],values['original']['recipe']['mapping_schedule'])

    def test_magnitude_match_control_is_not_coefficient_only_random(self):
        preview=dry_run(load_protocol('ingredients'),seeds=[17])
        random=next(e for e in preview['episodes'] if e['factors']['arm']=='random_matched')
        self.assertIn('actual_norm_random_v1',random['required_capabilities'])
        self.assertEqual(random['runtime_controls']['random_norm_match']['reference'],'same_unedited_position')
        with self.assertRaisesRegex(ValueError,'actual_norm_random'):execution_requests(preview,['recipe_v2','task_axis_v1'])


if __name__=='__main__':unittest.main()
