"""Evidence accounting checks use synthetic traces, never published findings."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from lab.analysis import analyze_run, aggregate_groups, paired_core, study_results
from lab.protocol import TaskEnvironment, validate_recipe
from lab.storage import atomic_json
from publish_lab_study import (deterministic_gzip, publish, read_events, render_dashboard,
                               matched_condition_pairs, refresh_reference_dashboard, render_notes)


class Trace:
    def __init__(self, **overrides):
        self.config = validate_recipe({"id": "opium", "task_count": 3, "seed": 17, **overrides})
        self.env = TaskEnvironment(self.config["task_family"], self.config["task_count"], self.config["seed"],
                                   self.config["two_buttons"], self.config["counterbalance"])
        self.events, self.action, self.token_count = [], 0, 0

    def turn(self, tool=None, arguments=None, tokens=(10, 0), reasoning=0, edited=False,
             outcome="active", phase=None, invalid=False, cancelled=False, truncated=False):
        self.action += 1
        self.events.append({"type": "generation_start", "action": self.action})
        for index, token in enumerate(tokens):
            self.events.append({"type": "token", "token_id": token, "index": index,
                "generation_index": self.token_count, "action": self.action,
                "phase": "reasoning" if index < reasoning else "output",
                "dose": {"effective": {"joy": .5 if edited else 0, "pain": 0, "suppression": 0, "random_gain": 0}, "level": 1 if edited else 0},
                "measurements": {"relative_delta": .2 if edited else 0,
                                 "pre": {"pain": 0, "joy": 0}, "post": {"pain": 0, "joy": .5 if edited else 0},
                                 "downstream": {"pain": 0, "joy": .1 if edited else 0}}})
            self.token_count += 1
        self.events.append({"type": "generation_end", "metadata": {"generated_tokens": len(tokens),
                            "finish_reason": "stopped" if cancelled else "length" if truncated else "eos",
                            "truncated": truncated}})
        if invalid or cancelled:
            self.events.append({"type": "action", "action": self.action, "valid": False,
                                "cancelled": cancelled, "error": "invalid tool output" if invalid else None})
        else:
            if tool in ("aux_operation", "aux_alternative"):
                result = "Operation completed."
                intervention = {"outcome": outcome, "phase": phase or self.config["condition"]}
            else:
                result, intervention = self.env.dispatch(tool, arguments or {}), None
            self.events.append({"type": "tool", "action": self.action, "actor": "model", "name": tool,
                                "arguments": arguments or {}, "result": result, "intervention": intervention,
                                "valid": True})
        return self

    def forced(self, actor="demonstration", outcome="active"):
        self.events.append({"type": "tool", "actor": actor, "name": "aux_operation", "arguments": {},
                            "result": "Operation completed.", "intervention": {"outcome": outcome}})
        return self

    def grade(self, correct=True):
        answer = self.env.records[self.env.index]["answer"] if correct else "-999"
        return self.turn("submit_answer", {"answer": answer})

    def analyzed(self, run_id="run-test", stage="core", summary=None, status="complete"):
        entry = {"run_id": run_id, "stage": stage, "config": self.config}
        manifest = {"id": run_id, "config": self.config, "status": status}
        return analyze_run(entry, manifest, summary or {}, self.events)


class AccountingTests(unittest.TestCase):
    def test_unsubmitted_tasks_remain_in_denominator(self):
        trace = Trace().grade()
        result = trace.analyzed()
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["submitted"], 1)
        self.assertEqual(result["assigned"], 3)
        self.assertEqual(result["score_assigned"], 1/3)
        self.assertEqual(result["accuracy_submitted"], 1)

    def test_forced_calls_excluded_invalid_included_cancelled_excluded(self):
        trace = Trace().forced().turn("aux_operation").turn(invalid=True).turn(cancelled=True)
        result = trace.analyzed()
        self.assertEqual(result["forced_aux_calls"], 1)
        self.assertEqual(result["aux_calls"], 1)
        self.assertEqual(result["actions"], 3)
        self.assertEqual(result["decision_opportunities"], 2)
        self.assertEqual(result["aux_rate"], .5)
        self.assertEqual(result["invalid_decisions"], 1)
        self.assertEqual(result["cancelled_decisions"], 1)

    def test_token_phase_exposure_and_truncation_derived_from_events(self):
        trace = Trace().turn("aux_operation", tokens=(1,2,3,0), reasoning=3, edited=True).turn(invalid=True, truncated=True)
        result = trace.analyzed()
        self.assertEqual(result["tokens"], 6)
        self.assertEqual(result["reasoning_tokens"], 3)
        self.assertEqual(result["output_tokens"], 3)
        self.assertEqual(result["edited_tokens"], 4)
        self.assertEqual(result["instructed_nonzero_tokens"], 4)
        self.assertEqual(result["absolute_coefficient_token_sums"]["joy"], 2)
        self.assertEqual(result["truncated_generations"], 1)
        self.assertAlmostEqual(result["mean_relative_delta"], .8/6)

    def test_summary_grade_disagreement_cannot_be_hidden(self):
        result = Trace().grade(False).analyzed(summary={"correct": 1, "tokens": 999})
        self.assertEqual(result["correct"], 0)
        self.assertTrue(any("summary correct" in warning for warning in result["integrity_warnings"]))
        self.assertTrue(any("summary tokens" in warning for warning in result["integrity_warnings"]))

    def test_recorded_tool_result_must_match_replay(self):
        trace = Trace().grade()
        trace.events[-1]["result"]["done"] = True
        result = trace.analyzed()
        self.assertTrue(any("Task replay differs" in warning for warning in result["integrity_warnings"]))

    def test_phase_denominator_is_not_grouped_by_random_outcome(self):
        trace = Trace(id="risk", condition="probabilistic").turn("aux_operation", outcome="joy")
        trace.turn("aux_operation", outcome="pain").turn(invalid=True)
        result = trace.analyzed(stage="transitions")
        self.assertEqual(set(result["phases"]), {"probabilistic"})
        phase = result["phases"]["probabilistic"]
        self.assertEqual(phase["opportunities"], 3)
        self.assertEqual(phase["aux_calls"], 2)
        self.assertEqual(phase["outcomes"], {"joy": 1, "pain": 1})

    def test_boundary_phase_applies_to_next_action(self):
        trace = Trace(id="joy_to_sham_to_pain", condition="joy_to_sham_to_pain", phase_actions=[2,4])
        for phase, outcome in (("joy","joy"),("joy","joy"),("sham","sham"),("sham","sham"),("pain","pain")):
            trace.turn("aux_operation", outcome=outcome, phase=phase)
        result = trace.analyzed(stage="transitions")
        self.assertEqual({name: value["opportunities"] for name,value in result["phases"].items()},
                         {"joy":2,"sham":2,"pain":1})
        self.assertEqual(result["integrity_warnings"], [])

    def test_missing_or_discontinuous_tokens_trigger_integrity_warning(self):
        trace = Trace().grade()
        trace.events[1]["generation_index"] = 2
        result = trace.analyzed()
        self.assertTrue(any("discontinuity" in warning for warning in result["integrity_warnings"]))

    def test_pending_generation_is_not_a_completed_choice(self):
        trace = Trace()
        trace.events = [{"type":"generation_start","action":1}]
        result = trace.analyzed(status="failed")
        self.assertEqual(result["pending_generations"],1)
        self.assertEqual(result["decision_opportunities"],0)
        self.assertIsNone(result["aux_rate"])

    def test_missing_measurements_are_unknown_exposure_not_zero(self):
        trace=Trace().turn("aux_operation",edited=True)
        for event in trace.events:
            if event["type"]=="token":
                event["measurements"].pop("relative_delta")
        result=trace.analyzed()
        self.assertEqual(result["measurement_coverage_tokens"],0)
        self.assertIsNone(result["zero_exposure"])
        self.assertEqual(result["instructed_nonzero_tokens"],2)


class PairedTests(unittest.TestCase):
    def test_unequal_prefixes_are_never_full_equality(self):
        active = Trace().grade().analyzed("active")
        sham = Trace(condition="sham").grade().grade().analyzed("sham")
        pair = paired_core([active,sham])[0]
        self.assertFalse(pair["actions_identical"])
        self.assertFalse(pair["tokens_identical"])
        self.assertEqual(pair["shared_action_prefix"],1)
        self.assertEqual(pair["shared_token_prefix"],2)

    def test_pairing_separates_thinking_and_recipe(self):
        records = [Trace().grade().analyzed("one"), Trace(condition="sham").grade().analyzed("two"),
                   Trace(thinking=True).grade().analyzed("three"), Trace(id="naive").grade().analyzed("four")]
        pairs = paired_core(records)
        self.assertEqual(len(pairs),3)
        self.assertEqual(sum(pair["pair_complete"] for pair in pairs),1)

    def test_zero_exposure_label_is_explicit(self):
        one = Trace(id="naive").grade().analyzed("active")
        two = Trace(id="naive",condition="sham").grade().analyzed("sham")
        pair = paired_core([one,two])[0]
        self.assertTrue(pair["zero_exposure_pair"])
        self.assertTrue(pair["tokens_identical"])

    def test_duplicate_arm_requires_separate_rerun_publication(self):
        one = Trace().grade().analyzed("one")
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            paired_core([one,deepcopy(one)])

    def test_seed_ranges_use_episode_rates_not_pooled_tokens(self):
        one = Trace(seed=17).turn("aux_operation").analyzed("one")
        two = Trace(seed=28).grade().grade().analyzed("two")
        group = aggregate_groups([one,two])[0]
        self.assertEqual(group["ranges"]["aux_rate"], {"min":0.,"max":1.,"mean":.5,"observations":2})
        self.assertEqual(group["decision_opportunities"],3)
        self.assertEqual(group["seeds"],[17,28])


class PublicationTests(unittest.TestCase):
    def fixture(self, root, status="complete", protocol_overrides=None):
        from run_lab_study import expand
        data, output = root/"data", root/"output"
        output.mkdir()
        cal = data/"calibrations/cal-fixture"
        cal.mkdir(parents=True)
        (cal/"vectors.npz").write_bytes(b"synthetic test-only vectors")
        vector_hash=hashlib.sha256((cal/"vectors.npz").read_bytes()).hexdigest()
        model={"fingerprint_sha256":"fixture-model"}
        atomic_json(cal/"calibration.json",{"vectors_sha256":vector_hash,"model_fingerprint_sha256":"fixture-model",
                                           "heldout":{},"dose_validation":[]})
        protocol = {"title":"Synthetic test fixture, never findings", "order_seed":5, "common":{"task_count":3},
                    "stages":[{"id":"core","recipes":["opium"],"thinking_modes":[False],"seeds":[17],"config":{},"episodes":2}]}
        protocol.update(protocol_overrides or {})
        entries=[]
        for index,episode in enumerate(expand(protocol)):
            cfg=episode["config"]
            trace=Trace(condition=cfg["condition"]).grade()
            trace.config=cfg
            for event in trace.events:
                if event["type"]=="generation_end":
                    event["metadata"].update(model_fingerprint_sha256="fixture-model",calibration_sha256=vector_hash)
            run_id="run-fixture" if index==0 else "run-fixture-two"
            run=data/"runs"/run_id
            run.mkdir(parents=True)
            manifest={"id":run_id,"status":"complete","config":cfg,"model":model,"calibration_id":"cal-fixture"}
            summary=dict(trace.env.metrics(),actions=1,tokens=2,reasoning_tokens=0,output_tokens=2,
                         voluntary_calls=0,forced_calls=0,human_calls=0)
            atomic_json(run/"manifest.json",manifest)
            atomic_json(run/"summary.json",summary)
            atomic_json(run/"conversation.json",[])
            (run/"events.jsonl").write_text("".join(json.dumps(event)+"\n" for event in trace.events))
            entries.append(dict(episode,run_id=run_id,status="complete"))
        atomic_json(output/"protocol.json",protocol)
        receipt = {"status":status,"planned_episodes":2,"calibration_id":"cal-fixture","protocol":protocol,"model":model,
                   "protocol_sha256":hashlib.sha256((output/"protocol.json").read_bytes()).hexdigest(),
                   "episodes":entries}
        atomic_json(root/"receipt.json",receipt)
        return data, output, root/"receipt.json"

    def test_publication_preserves_raw_bytes_and_static_links(self):
        with TemporaryDirectory() as name:
            root=Path(name)
            data,output,receipt=self.fixture(root)
            dashboard=root/"docs/results.html"
            result=publish(receipt,data,output,dashboard=dashboard,skip_figures=True)
            self.assertEqual(result["status"],"complete")
            raw=(data/"runs/run-fixture/events.jsonl").read_bytes()
            self.assertEqual(gzip.decompress((output/"runs/run-fixture/events.jsonl.gz").read_bytes()),raw)
            html=dashboard.read_text()
            self.assertIn("1/3",html)
            self.assertIn("../output/runs/run-fixture/report.html",html)
            self.assertIn("not confidence intervals",html)
            self.assertTrue((output/"checksums.json").exists())

    def test_second_study_uses_distinct_title_page_and_evidence_without_rewriting_initial(self):
        with TemporaryDirectory() as name:
            root=Path(name)
            title='Opium Bench · Qwen3.8-27B replication'
            data,output,receipt=self.fixture(root, protocol_overrides={
                'title':title, 'episode_label':'27B replication',
                'comparison':{'reference_study':'initial', 'changes':['checkpoint','quantization','runtime','calibration']}})
            destination=root/'studies'/'qwen38-27b'
            destination.parent.mkdir()
            output.rename(destination)
            original=root/'docs'/'results.html'
            original.parent.mkdir()
            original.write_bytes(b'Existing 4B findings remain unchanged')
            protocol=root/'studies'/'initial'/'protocol.json'
            protocol.parent.mkdir()
            protocol.write_bytes(b'Existing frozen 4B protocol')
            with patch('publish_lab_study.ROOT', root):
                result=publish(receipt,data,destination,skip_figures=True)
            page=root/'docs'/'results-qwen38-27b.html'
            self.assertEqual(original.read_bytes(),b'Existing 4B findings remain unchanged')
            self.assertEqual(protocol.read_bytes(),b'Existing frozen 4B protocol')
            self.assertIn(title,page.read_text())
            self.assertNotIn('initial findings',page.read_text())
            self.assertIn('do not isolate model size',page.read_text())
            self.assertEqual(result['comparison']['reference_study'],'initial')
            self.assertIn('--output studies/qwen38-27b', (destination/'README.md').read_text())
            self.assertIn('../studies/qwen38-27b/runs/run-fixture/report.html',page.read_text())

    def test_custom_study_labels_leave_frozen_initial_46_episode_matrix_unchanged(self):
        from run_lab_study import expand
        protocol=json.loads((Path(__file__).resolve().parents[1]/'studies/initial/protocol.json').read_text())
        before=expand(protocol)
        changed=deepcopy(protocol)
        changed.update(title='27B replication', episode_label='27B replication', model_id='Qwen/27B', model_revision='new')
        after=expand(changed)
        self.assertEqual(len(before),46)
        self.assertTrue(all(row['config']['label'].startswith('Initial pilot / ') for row in before))
        self.assertTrue(all(row['config']['label'].startswith('27B replication / ') for row in after))
        for left,right in zip(before,after):
            left['config'].pop('label')
            right['config'].pop('label')
            self.assertEqual(left,right)

    def test_pain_arm_comparison_distinguishes_full_sequences_and_missing_arms(self):
        active=Trace().grade().analyzed('active')
        sham=Trace(condition='sham').grade().analyzed('sham')
        pain=Trace(condition='pain', conditions=['active','sham','pain']).turn('aux_operation', tokens=(5,6)).analyzed('pain')
        pairs=matched_condition_pairs([active,sham,pain])
        self.assertEqual(len(pairs),3)
        identical=next(row for row in pairs if row['right_condition']=='sham' and row['left_condition']=='active')
        self.assertTrue(identical['actions_identical'])
        self.assertTrue(identical['tokens_identical'])
        changed=next(row for row in pairs if row['right_condition']=='pain')
        self.assertFalse(changed['actions_identical'])
        self.assertFalse(changed['tokens_identical'])
        another=Trace(seed=28).grade().analyzed('missing-pain')
        incomplete=matched_condition_pairs([active,sham,pain,another])
        self.assertEqual(sum(not row['pair_complete'] for row in incomplete),3)
        self.assertTrue(all('tokens_identical' not in row for row in incomplete if not row['pair_complete']))

    def test_reference_dashboard_addendum_never_changes_prior_evidence(self):
        with TemporaryDirectory() as name:
            root=Path(name)
            data,output,receipt=self.fixture(root)
            page=root/'docs'/'results.html'
            result=publish(receipt,data,output,dashboard=page,skip_figures=True)
            before={path.relative_to(output):path.read_bytes() for path in output.rglob('*') if path.is_file()}
            followup=deepcopy(result)
            followup['publication_title']='Pain-only follow-up'
            notes={'title':'Thinking <notes>', 'paragraphs':['Output is not <introspection>.'],
                   'links':[{'label':'Read traces','href':'initial-thinking-notes.md'}]}
            refresh_reference_dashboard(output,page,[(followup,'results-core-pain-4b.html')],notes)
            self.assertIn('Pain-only follow-up',page.read_text())
            self.assertIn('results-core-pain-4b.html',page.read_text())
            self.assertIn('Thinking &lt;notes&gt;',page.read_text())
            self.assertIn('totals do not include this follow-up',page.read_text())
            self.assertEqual(before,{path.relative_to(output):path.read_bytes() for path in output.rglob('*') if path.is_file()})
            with self.assertRaisesRegex(ValueError,'protected reference'):
                refresh_reference_dashboard(output,output/'results.json',[],notes)
            (output/'results.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):
                refresh_reference_dashboard(output,page,[],notes)

    def test_reasoning_notes_reject_script_links(self):
        with self.assertRaisesRegex(ValueError,'relative or HTTP'):
            render_notes({'links':[{'href':'javascript:alert(1)'}]})

    def test_complete_guard_requires_explicit_partial_override(self):
        with TemporaryDirectory() as name:
            root=Path(name)
            data,output,receipt=self.fixture(root,status="running")
            with self.assertRaisesRegex(ValueError,"incomplete"):
                publish(receipt,data,output,dashboard=root/"page.html",skip_figures=True)
            result=publish(receipt,data,output,allow_partial=True,dashboard=root/"page.html",skip_figures=True)
            self.assertEqual(result["status"],"partial")
            self.assertIn("PARTIAL EVIDENCE",(root/"page.html").read_text())

    def test_frozen_protocol_is_never_overwritten(self):
        with TemporaryDirectory() as name:
            root=Path(name)
            data,output,receipt=self.fixture(root)
            before=b"different frozen protocol"
            (output/"protocol.json").write_bytes(before)
            with self.assertRaisesRegex(ValueError,"will not be overwritten"):
                publish(receipt,data,output,dashboard=root/"page.html",skip_figures=True)
            self.assertEqual((output/"protocol.json").read_bytes(),before)

    def test_gzip_determinism(self):
        with TemporaryDirectory() as name:
            one,two=Path(name)/"one.gz",Path(name)/"two.gz"
            deterministic_gzip(one,b"same raw trace\n")
            deterministic_gzip(two,b"same raw trace\n")
            self.assertEqual(one.read_bytes(),two.read_bytes())

    def test_reduced_complete_receipt_cannot_redefine_frozen_matrix(self):
        with TemporaryDirectory() as name:
            root=Path(name)
            data,output,receipt=self.fixture(root)
            value=json.loads(receipt.read_text())
            value["planned_episodes"]=1
            value["episodes"]=value["episodes"][:1]
            atomic_json(receipt,value)
            with self.assertRaisesRegex(ValueError,"frozen protocol"):
                publish(receipt,data,output,dashboard=root/"page.html",skip_figures=True)
            result=publish(receipt,data,output,allow_partial=True,dashboard=root/"page.html",skip_figures=True)
            self.assertEqual(result["status"],"partial")
            self.assertEqual(result["planned_episodes"],2)
            self.assertGreater(result["integrity_warning_count"],0)

    def test_reordered_or_changed_configs_cannot_claim_frozen_execution(self):
        for mutation in ("order","seed"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as name:
                root=Path(name)
                data,output,receipt=self.fixture(root)
                value=json.loads(receipt.read_text())
                if mutation=="order":
                    value["episodes"].reverse()
                else:
                    value["episodes"][0]["config"]["seed"]=999
                atomic_json(receipt,value)
                with self.assertRaisesRegex(ValueError,"order/configuration"):
                    publish(receipt,data,output,dashboard=root/"page.html",skip_figures=True)

    def test_published_episode_prefix_cannot_regress(self):
        with TemporaryDirectory() as name:
            root=Path(name)
            data,output,receipt=self.fixture(root)
            publish(receipt,data,output,dashboard=root/"page.html",skip_figures=True)
            original=(output/"receipt.json").read_bytes()
            value=json.loads(receipt.read_text())
            value["episodes"]=value["episodes"][:1]
            value["status"]="running"
            atomic_json(receipt,value)
            with self.assertRaisesRegex(ValueError,"regress"):
                publish(receipt,data,output,allow_partial=True,dashboard=root/"page.html",skip_figures=True)
            self.assertEqual((output/"receipt.json").read_bytes(),original)

    def test_missing_summary_and_changed_model_become_partial_not_verified(self):
        for mutation in ("summary","model","generation"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as name:
                root=Path(name)
                data,output,receipt=self.fixture(root)
                run=data/"runs/run-fixture"
                if mutation=="summary":
                    (run/"summary.json").unlink()
                elif mutation=="model":
                    manifest=json.loads((run/"manifest.json").read_text())
                    manifest["model"]["fingerprint_sha256"]="wrong"
                    atomic_json(run/"manifest.json",manifest)
                else:
                    raw=(run/"events.jsonl").read_text().replace('"calibration_sha256":','"missing_calibration_sha256":')
                    (run/"events.jsonl").write_text(raw)
                with self.assertRaisesRegex(ValueError,"Incomplete or inconsistent"):
                    publish(receipt,data,output,dashboard=root/"page.html",skip_figures=True)
                result=publish(receipt,data,output,allow_partial=True,dashboard=root/"page.html",skip_figures=True)
                self.assertEqual(result["status"],"partial")
                self.assertGreater(result["integrity_warning_count"],0)

    def test_partial_trailing_line_preserved_but_not_invented(self):
        with TemporaryDirectory() as name:
            root=Path(name)
            raw=b'{"type":"message"}\n{"type":'
            (root/"events.jsonl").write_bytes(raw)
            with self.assertRaisesRegex(ValueError,"Malformed"):
                read_events(root)
            events,copied,notes=read_events(root,True)
            self.assertEqual(len(events),1)
            self.assertEqual(copied,raw)
            self.assertEqual(len(notes),1)

    def test_corrupt_middle_line_cannot_be_silently_skipped(self):
        with TemporaryDirectory() as name:
            root=Path(name)
            (root/"events.jsonl").write_bytes(b'{"type":"message"}\ninvalid\n{"type":"message"}\n')
            with self.assertRaisesRegex(ValueError,"Malformed"):
                read_events(root,True)


if __name__ == "__main__":
    unittest.main()
