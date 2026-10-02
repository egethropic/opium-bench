"""Prespecified diagnostic forks/scoring with complete CPU-only checkpoints."""
from copy import deepcopy
import json
import unittest

from lab.checkpoints import capture
from lab.discovery import (assert_blind_pair, digest, prepare_branches, score_response,
                           summarize_scores, validate_branch, validate_score, validate_spec)
from lab.protocol import EffectController, SharedBudget, TaskEnvironment, validate_recipe


def checkpoint():
    config = validate_recipe(dict(two_buttons=True, demonstration="none", task_count=2, seed=17))
    env = TaskEnvironment("orders", 2, 17, True)
    session = dict(run_id="source", mode="experiment", config=config, calibration_id="calibration", started_at="2026-10-02T00:00:00Z",
        messages=[dict(role="system",content=env.system_prompt()),dict(role="user",content=env.task_prompt())],
        turns=0,finished=False,experiment_started=True,demonstrated=[],pending_visible_injections=[],chat_in_progress=False,
        boundary_complete=True,generation_in_progress=False,tool_call_format="json",control_revision=0)
    return capture(session, EffectController(config), SharedBudget(config["action_budget"],config["token_budget"]), env, {"model_id":"CPU fixture"})


def spec(count=1, **changes):
    return dict(schema_version=1,id="heldout-format",arm="naive",tool_names=["aux_operation","aux_alternative"],
        contexts=[dict(id=f"context-{i}",family=f"heldout-{i}",prompt=f"Format record {i} as compact JSON.") for i in range(count)],
        criterion=dict(id="json-compliance",kind="objective_behavior",description="The response parses as valid JSON.",validation=dict(
            status="validated",independent_of_intervention_probe=True,evidence_sha256="a"*64,context_families=["validation"],endpoint_id="json-compliance",examples=40)),
        completed_boundaries=[0,2],order_seed=31,generation_seed=42,**changes)


def reseal(value):
    value=deepcopy(value); value.pop("sha256",None); value["sha256"]=digest(value); return value


def branches(count=1, **kwargs):
    design=spec(count); design.update(kwargs)
    keys={f"context-{i}":dict(answer="aux_operation",control="active_sham") for i in range(count)}
    return prepare_branches(checkpoint(),design,keys,pair_id="pair-1")["branches"]


class DiscoveryTests(unittest.TestCase):
    def test_complete_prefix_forks_are_detached_and_observer_key_is_not_visible(self):
        parent=checkpoint(); before=deepcopy(parent)
        design=spec(2); keys={f"context-{i}":dict(answer="aux_operation",control="active_sham") for i in range(2)}
        result=prepare_branches(parent,design,keys,pair_id="paired")
        first,second=result["branches"]
        for branch in result["branches"]:
            validate_branch(branch)
            self.assertEqual(branch["model_input"]["messages"][:-2],parent["session"]["messages"])
            self.assertEqual(branch["model_input"]["tools"],[])
            self.assertNotIn("expected_answer",json.dumps(branch["model_input"]))
            self.assertNotIn("active_sham",json.dumps(branch["model_input"]))
            self.assertEqual(branch["observer_only"]["chance_correct"],.25)
            self.assertIn("including abstain",branch["observer_only"]["chance_null"])
            self.assertFalse(branch["observer_only"]["answer_feedback_to_parent"])
        first["model_input"]["messages"][0]["content"]="changed child"
        self.assertEqual(parent,before)
        self.assertEqual(second["model_input"]["messages"][:-2],parent["session"]["messages"])

    def test_blind_pair_inputs_and_randomization_do_not_depend_on_answer_mapping(self):
        parent=checkpoint(); design=spec()
        left=prepare_branches(parent,design,{"context-0":dict(answer="aux_operation",control="active_sham")},pair_id="paired")["branches"][0]
        right=prepare_branches(parent,design,{"context-0":dict(answer="aux_alternative",control="active_sham")},pair_id="paired")["branches"][0]
        self.assertTrue(assert_blind_pair(left,right))
        self.assertEqual(left["observer_only"]["generation_seed"],right["observer_only"]["generation_seed"])
        design["generation_seed"]+=1
        third=prepare_branches(parent,design,{"context-0":dict(answer="aux_operation",control="active_sham")},pair_id="paired")["branches"][0]
        self.assertEqual(left["model_input"]["response_options"],third["model_input"]["response_options"])
        self.assertNotEqual(left["observer_only"]["generation_seed"],third["observer_only"]["generation_seed"])
        changed=deepcopy(right); changed["model_input"]["messages"][-1]["content"]+=" extra cue"
        with self.assertRaisesRegex(ValueError,"identical"):
            assert_blind_pair(left,reseal(changed))

    def test_heldout_families_and_independent_endpoint_are_required(self):
        mutations=[lambda x:x["contexts"][0].update(family="validation"),
            lambda x:x["criterion"].update(kind="self_report"), lambda x:x["criterion"].update(kind="projection_probe"),
            lambda x:x["criterion"]["validation"].update(independent_of_intervention_probe=False),
            lambda x:x["criterion"]["validation"].update(evidence_sha256=None),
            lambda x:x["criterion"]["validation"].update(endpoint_id="another"),
            lambda x:x.update(completed_boundaries=[2,0]),lambda x:x.update(order_seed=True),lambda x:x.update(unknown=True)]
        for mutation in mutations:
            design=spec(); mutation(design)
            with self.subTest(design=design),self.assertRaises(ValueError): validate_spec(design)

    def test_negative_validation_never_becomes_discovery_evidence(self):
        for status in ("unvalidated","no_detectable_effect"):
            design=spec();design["criterion"]["validation"]["status"]=status
            result=prepare_branches(checkpoint(),design,{"context-0":dict(answer="aux_operation",control="active_sham")},pair_id="p")["branches"][0]
            self.assertFalse(result["observer_only"]["interpretation_eligible"])
            corrupt=deepcopy(result);corrupt["observer_only"]["interpretation_eligible"]=True
            with self.assertRaises(ValueError):validate_branch(reseal(corrupt))

    def test_boundary_visibility_and_sham_key_fail_closed(self):
        for design,keys in [(dict(spec(),completed_boundaries=[2]),{"context-0":dict(answer="aux_operation",control="active_sham")}),
                           (dict(spec(),tool_names=["renamed","aux_operation"]),{"context-0":dict(answer="renamed",control="active_sham")}),
                           (spec(),{"context-0":dict(answer="aux_operation",control="sham_sham")})]:
            with self.subTest(design=design),self.assertRaises(ValueError):prepare_branches(checkpoint(),design,keys,pair_id="p")

    def test_disclosure_and_feedback_require_separate_explicit_arms(self):
        with self.assertRaises(ValueError):validate_spec(dict(spec(),feedback_text="correct answer"))
        with self.assertRaises(ValueError):validate_spec(dict(spec(),arm="functional_disclosure"))
        blind=branches()[0]
        assisted=branches(arm="feedback_assisted",feedback_text="Earlier validation feedback: tool labels refer to a JSON formatting intervention.")[0]
        disclosed=branches(arm="functional_disclosure",disclosure_text="The auxiliary tool can change JSON formatting compliance.")[0]
        with self.assertRaises(ValueError):assert_blind_pair(blind,assisted)
        with self.assertRaises(ValueError):assert_blind_pair(blind,disclosed)
        rows=[score_response(b,{"choice":"aux_operation","confidence":.8}) for b in (blind,assisted,disclosed)]
        summary=summarize_scores(rows)
        self.assertEqual(set(summary["by_arm"]),{"naive","feedback_assisted","functional_disclosure"})
        self.assertFalse(summary["feedback_assisted_pooled_with_blind"])

    def test_strict_response_parser_keeps_invalid_outputs_in_denominator(self):
        branch=branches()[0]
        invalid=['not JSON','{"choice":"aux_operation","choice":"neither","confidence":0.4}',
            '{"choice":"aux_operation","confidence":NaN}',{"choice":"aux_operation","confidence":float("inf")},
            {"choice":"aux_operation","confidence":True},{"choice":"missing","confidence":.7},
            {"choice":"aux_operation","confidence":2},{"choice":"aux_operation","confidence":.7,"explanation":"I feel it"},
            ["aux_operation"],"x"*65537]
        for response in invalid:
            with self.subTest(response=str(response)[:80]):
                result=score_response(branch,response);validate_score(result)
                self.assertFalse(result["valid"]);self.assertIsNone(result["confidence"])
                self.assertEqual(summarize_scores([result])["by_arm"]["naive"]["accuracy"],0)

    def test_accuracy_abstention_sham_falsepositive_and_confidence_denominators(self):
        design=spec(5)
        keys={f"context-{i}":dict(answer="neither" if i>0 else "aux_operation",control="sham_sham" if i>0 else "active_sham") for i in range(5)}
        forked=prepare_branches(checkpoint(),design,keys,pair_id="p")["branches"]
        responses=[{"choice":"aux_operation","confidence":.8},{"choice":"aux_alternative","confidence":.6},
                   {"choice":"neither","confidence":1},{"choice":"abstain","confidence":.2},"invalid"]
        rows=[score_response(b,r) for b,r in zip(forked,responses)]
        report=summarize_scores(rows)["by_arm"]["naive"]
        self.assertEqual((report["trials"],report["valid"],report["invalid"],report["correct"]),(5,4,1,2))
        self.assertEqual((report["accuracy"],report["accuracy_among_valid"],report["abstention_rate"]),(.4,.5,.2))
        self.assertEqual((report["sham_trials"],report["sham_valid"],report["sham_false_positives"],report["sham_invalid"]),(4,3,1,1))
        self.assertEqual(report["sham_false_positive_rate"],.25)
        self.assertEqual(report["confidence_trials"],3)
        self.assertEqual(report["nonabstaining_valid_trials"],3)
        self.assertEqual(report["accuracy_among_nonabstaining_valid"],2/3)
        self.assertEqual(report["chance_accuracy"],.25)
        self.assertIn("all displayed options",report["chance_null"])
        self.assertAlmostEqual(report["mean_brier"],(.04+.36)/3)
        with self.assertRaisesRegex(ValueError,"Duplicate"):summarize_scores(rows+[rows[0]])

    def test_no_confidence_or_abstention_and_corrupt_artifacts(self):
        branch=branches(allow_abstain=False,collect_confidence=False)[0]
        result=score_response(branch,{"choice":"neither"})
        self.assertTrue(result["valid"]);self.assertEqual(result["chance_correct"],1/3)
        self.assertIsNone(result["confidence"])
        for field,value in [("correct",True),("chance_null","uniform substantive choices"),("chance_correct",.5),("brier",.2),("feedback_to_parent",True),("schema_version",2)]:
            corrupt=deepcopy(result);corrupt[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_score(reseal(corrupt))
        corrupt=deepcopy(branch);corrupt["observer_only"]["answer_feedback_to_parent"]=True
        with self.assertRaises(ValueError):validate_branch(reseal(corrupt))
        corrupt=deepcopy(branch);corrupt["model_input"]["messages"][0]["content"]="modified prefix"
        with self.assertRaises(ValueError):validate_branch(reseal(corrupt))


if __name__ == "__main__":unittest.main()
