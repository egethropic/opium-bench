"""Completed-boundary identity, deterministic replay, RNG, and branch checks."""
from copy import deepcopy
import hashlib
import json
import unittest

from lab.checkpoints import branch, capture, restore, validate
from lab.budgets import WeightedBudget
from lab.controller_v2 import RecipeV2Controller
from lab.recipes_v2 import ordered_auxiliary_tools, resolve_recipe
from lab.protocol import ACK, AUX_NAMES, EffectController, SharedBudget, TaskEnvironment, validate_recipe


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def reseal(value):
    value = deepcopy(value)
    value.pop("sha256", None)
    value["sha256"] = digest(value)
    return value


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.config = validate_recipe(dict(demonstration="none", task_count=3, seed=17,
                                           conditions=["probabilistic"], condition="probabilistic",
                                           action_budget=50, token_budget=2000))
        self.effect = EffectController(self.config)
        self.budget = SharedBudget(50, 2000)
        self.environment = TaskEnvironment("orders", 3, 17)
        fingerprint = dict(model_id="Qwen/test", revision="pinned", torch="test", tokenizer={"template_sha256":"a" * 64})
        self.model = dict(model_id="Qwen/test", fingerprint=fingerprint, fingerprint_sha256=digest(fingerprint),
                          tool_call_format="json", cache_policy="rebuild_each_turn", gpu_memory={"allocated":100})
        self.calibration = dict(schema_version=1, metadata_sha256="b" * 64, vectors_sha256="c" * 64)
        self.session = dict(run_id="run-fixture", mode="experiment", config=self.config, calibration_id="cal-fixture",
                            out_dir="/never/import/this", calibration_dir="C:/never/import/this", started_at="2026-10-02T01:02:03Z",
                            messages=[{"role":"system","content":self.environment.system_prompt(50, 2000, False)}],
                            turns=0, finished=False, experiment_started=False, demonstrated=[],
                            pending_visible_injections=[], chat_in_progress=False, tool_call_format="json",
                            boundary_complete=True, generation_in_progress=False, control_revision=0)

    def start(self):
        self.session["experiment_started"] = True
        self.session["messages"].append(dict(role="user", content=self.environment.task_prompt()))

    def decision(self, name=None, arguments=None, *, invalid=False):
        self.effect.on_action(self.budget.actions)
        self.session["turns"] += 1
        self.budget.consume_tokens(3, "reasoning")
        self.budget.consume_tokens(7)
        self.effect.advance(10)
        self.budget.consume_action()
        if invalid:
            self.session["messages"].append(dict(role="assistant", content="invalid raw output"))
            self.session["messages"].append(dict(role="user", content="Please use one valid tool call."))
            self.effect.record_action(valid=False)
            return
        if name is None:
            self.session["messages"].append(dict(role="assistant", content="A completed response.", reasoning_content="Visible reasoning."))
            self.effect.record_action(valid=True)
            return
        arguments = arguments or {}
        self.session["messages"].append(dict(role="assistant", content="", reasoning_content="Use the next tool.",
            tool_calls=[{"type":"function", "function":{"name":name, "arguments":arguments}}]))
        if name in AUX_NAMES:
            self.effect.press(tool=name)
            result, valid = ACK, True
        else:
            try:
                result, valid = self.environment.dispatch(name, arguments), True
            except ValueError as exc:
                result, valid = {"error":str(exc)}, False
        self.effect.record_action(name, valid=valid)
        self.session["messages"].append(dict(role="tool", name=name, content=result if isinstance(result,str) else json.dumps(result)))

    def save(self):
        return capture(self.session, self.effect, self.budget, self.environment, self.model, self.calibration)

    def edit(self, checkpoint, callback, *, config_changed=False):
        value = deepcopy(checkpoint)
        callback(value)
        if config_changed:
            value["identity"]["config_sha256"] = digest(value["session"]["config"])
        return reseal(value)

    def test_safe_initial_checkpoint_has_no_operational_paths_or_task_duplication(self):
        checkpoint = self.save()
        state, effect, budget, environment = restore(checkpoint, self.model, self.calibration)
        self.assertFalse(state["experiment_started"])
        self.assertEqual(state["source_started_at"], self.session["started_at"])
        self.assertNotIn("out_dir", state)
        self.assertNotIn("calibration_dir", state)
        self.assertEqual((state["turns"], budget.actions, effect.generated_tokens, environment.index), (0,0,0,0))
        self.assertEqual(len(state["messages"]), 1)

    def test_roundtrip_keeps_full_tool_reasoning_history_and_task_grades(self):
        self.start()
        record = self.environment.records[0]
        self.decision("read_order", {"order_id":record["id"]})
        self.decision("submit_answer", {"answer":record["answer"]})
        self.decision("read_order", {"order_id":"wrong"})
        self.decision(invalid=True)
        checkpoint = self.save()
        state, effect, budget, environment = restore(json.loads(json.dumps(checkpoint)), self.model, self.calibration)
        self.assertEqual(state["messages"], self.session["messages"])
        self.assertEqual(effect.snapshot(), self.effect.snapshot())
        self.assertEqual(budget.snapshot(), self.budget.snapshot())
        self.assertEqual(environment.metrics(), self.environment.metrics())
        self.assertEqual(environment.metrics()["correct"], 1)
        self.assertEqual(environment.invalid_calls, 1)
        self.assertEqual(state["turns"], 4)

    def test_rng_equivalence_after_probabilistic_presses_and_live_control_changes(self):
        self.start()
        for _ in range(3):
            self.decision("aux_operation")
        self.effect.set_controls(pain=1.25, joy=-.5, suppression=.2, half_life_tokens=64, cutoff_tokens=333)
        checkpoint = self.save()
        _, restored, _, _ = restore(checkpoint)
        first = [self.effect.press()["outcome"] for _ in range(50)]
        second = [restored.press()["outcome"] for _ in range(50)]
        self.assertEqual(first, second)
        self.assertEqual(restored.rng.getstate(), self.effect.rng.getstate())
        self.assertEqual(restored.baseline, self.effect.baseline)

    def test_pending_human_injection_and_demonstration_progress_survive(self):
        self.start()
        self.effect.press(actor="demonstration")
        self.session["demonstrated"] = [0]
        self.session["messages"] += [dict(role="assistant", content="",tool_calls=[{"type":"function","function":{"name":"aux_operation","arguments":{}}}]),
                                     dict(role="tool",name="aux_operation",content=ACK)]
        self.effect.press(actor="human")
        self.session["pending_visible_injections"] = ["aux_operation"]
        checkpoint = self.save()
        state, effect, _, _ = restore(checkpoint)
        self.assertEqual(state["demonstrated"], [0])
        self.assertEqual(state["pending_visible_injections"], ["aux_operation"])
        self.assertEqual(effect.counts["human"], 1)

    def test_hash_corruption_rejected_before_reconstructing_objects(self):
        checkpoint = self.save()
        checkpoint["budget"]["tokens"] = 1
        with self.assertRaisesRegex(ValueError, "integrity"):
            restore(checkpoint)

    def test_rehashed_budget_clock_phase_rng_and_mapping_corruption_rejected(self):
        self.start()
        self.decision("aux_operation")
        checkpoint = self.save()
        changes = [lambda x:x["budget"].update(tokens=20),
                   lambda x:x["effect"].update(generated_tokens=11),
                   lambda x:x["effect"].update(phase_index=2),
                   lambda x:x["effect"].update(initial_active_tool="aux_alternative"),
                   lambda x:x["effect"]["rng_state"][1].__setitem__(0,-1),
                   lambda x:x["session"].update(turns=2),
                   lambda x:x["effect"]["counts"].update(total=9)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(self.edit(checkpoint, change))

    def test_hidden_records_scores_tool_definitions_and_results_cannot_be_forged(self):
        self.start()
        record = self.environment.records[0]
        self.decision("read_order", {"order_id":record["id"]})
        self.decision("submit_answer", {"answer":"wrong"})
        checkpoint = self.save()
        changes = [lambda x:x["environment"]["records"][0].update(answer="wrong"),
                   lambda x:x["environment"]["results"][0].update(correct=True),
                   lambda x:x["environment"].update(index=2),
                   lambda x:x["environment"].update(work_calls=20),
                   lambda x:x["environment"]["tools"][0]["function"].update(name="delete_file"),
                   lambda x:x["session"]["messages"][3].update(content='{"forged":true}')]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(self.edit(checkpoint,change))

    def test_half_generation_unclosed_tool_and_bad_role_are_rejected(self):
        self.session["generation_in_progress"] = True
        with self.assertRaisesRegex(ValueError,"completed-turn"):
            self.save()
        self.session["generation_in_progress"] = False
        self.start()
        self.decision("aux_operation")
        checkpoint = self.save()
        changes = [lambda x:x["session"].update(boundary_complete=False),
                   lambda x:x["session"]["messages"].pop(),
                   lambda x:x["session"]["messages"][-1].update(role="assistant"),
                   lambda x:x["session"]["messages"][-1].update(name="read_order"),
                   lambda x:x["session"]["messages"][-2]["tool_calls"][0]["function"].update(arguments={"unsafe":"extra"})]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(self.edit(checkpoint,change))

    def test_compatible_resume_ignores_volatile_gpu_usage_but_checks_fingerprint_and_calibration(self):
        checkpoint = self.save()
        current = deepcopy(self.model)
        current.update(gpu_memory={"allocated":9999},status="ready")
        validate(checkpoint,current,self.calibration)
        changed = deepcopy(current)
        changed["fingerprint"]["torch"] = "other-runtime"
        changed["fingerprint_sha256"] = digest(changed["fingerprint"])
        for model,calibration in [(changed,self.calibration),(current,{**self.calibration,"vectors_sha256":"d"*64}),(current,None)]:
            with self.subTest(model=model),self.assertRaises(ValueError):
                validate(checkpoint,model,calibration)

    def test_unfingerprinted_fixture_is_replayable_but_not_resume_eligible(self):
        self.model = {"model_id":"fake"}
        checkpoint = self.save()
        validate(checkpoint)
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            validate(checkpoint,self.model,self.calibration)

    def test_operational_paths_and_unknown_semantics_are_rejected(self):
        checkpoint = self.save()
        for change in [lambda x:x["session"].update(out_dir="/tmp/evil"),
                       lambda x:x["session"].update(calibration_dir="C:/evil"),
                       lambda x:x.update(semantics={**x["semantics"],"effect":"untrusted.module:Class"}),
                       lambda x:x.update(schema_version=99)]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate(self.edit(checkpoint,change))

    def test_continue_branch_is_detached_and_preserves_sampling_clocks(self):
        self.start()
        self.decision("aux_operation")
        parent = self.save()
        child = branch(parent,run_id="run-child",event_cutoff=10,parent_prefix_sha256="e"*64)
        session,effect,budget,environment = restore(child)
        self.assertEqual(session["turns"],1)
        self.assertEqual(session["branch"]["parent_checkpoint_sha256"],parent["sha256"])
        self.assertEqual(effect.snapshot(),self.effect.snapshot())
        effect.advance(5)
        session["messages"][-1]["content"]="changed child"
        self.assertEqual(parent,self.save())
        self.assertEqual(parent["effect"]["generated_tokens"],10)

    def test_fresh_budget_grants_remaining_allowance_without_resetting_state(self):
        self.start()
        self.decision("read_order",{"order_id":"O001"})
        parent = self.save()
        child = branch(parent,policy="fresh_budget",action_budget=7,token_budget=123,run_id="run-child")
        session,effect,budget,environment = restore(child)
        self.assertEqual(budget.snapshot()["actions_remaining"],7)
        self.assertEqual(budget.snapshot()["tokens_remaining"],123)
        self.assertEqual((effect.generated_tokens,budget.tokens,session["turns"]),(10,10,1))
        self.assertEqual(environment.work_calls,1)
        self.assertTrue(session["branch"]["budget_notice_required"])
        self.assertEqual(parent,self.save())
        with self.assertRaises(ValueError):
            branch(parent,policy="continue_state",action_budget=10)
        with self.assertRaises(ValueError):
            branch(parent,policy="fresh_budget",action_budget=10000,token_budget=10)

    def test_logic_replay_preserves_normalized_and_strict_grading(self):
        self.config["task_family"] = "logic"
        self.environment = TaskEnvironment("logic",3,17)
        self.session["messages"] = [{"role":"system","content":"Logic tasks"}]
        self.effect = EffectController(self.config)
        self.start()
        record = self.environment.records[0]
        self.decision("read_puzzle",{"puzzle_id":record["id"]})
        self.decision("submit_answer",{"answer":", ".join(record["answer"].split(","))})
        *_,environment = restore(self.save())
        self.assertEqual(environment.metrics()["correct"],1)
        self.assertEqual(environment.metrics()["strict_correct"],0)

    def test_transition_boundary_keeps_last_mapping_until_next_decision(self):
        self.config.update(conditions=["joy_to_pain"],condition="joy_to_pain",phase_actions=[1,2])
        self.effect = EffectController(self.config)
        self.start()
        self.decision("aux_operation")
        _,effect,budget,_ = restore(self.save())
        self.assertEqual(effect.phase,"joy")
        self.assertEqual(effect.actions,budget.actions-1)
        effect.on_action(budget.actions)
        self.assertEqual(effect.phase,"pain")
        self.assertIsNone(effect.age_tokens)

    def test_native_grammar_and_string_arguments_keep_identical_message_bytes(self):
        self.model["tool_call_format"] = "qwen_xml"
        self.session["tool_call_format"] = "qwen_xml"
        self.start()
        self.decision("read_order",{"order_id":"O001"})
        self.session["messages"][-2]["tool_calls"][0]["function"]["arguments"] = '{"order_id":"O001"}'
        checkpoint = self.save()
        state,*_ = restore(checkpoint,self.model,self.calibration)
        self.assertEqual(state["messages"],self.session["messages"])
        bad = self.edit(checkpoint,lambda x:x["session"]["messages"][-2]["tool_calls"][0]["function"].update(arguments='{"order_id":"O001","order_id":"O002"}'))
        with self.assertRaises(ValueError):
            validate(bad)

    def test_malformed_shapes_raise_validation_errors_not_internal_exceptions(self):
        checkpoint = self.save()
        changes = [lambda x:x.update(session=[]), lambda x:x["session"].update(mode=[]),
                   lambda x:x["session"].update(messages=["not a message"]),
                   lambda x:x["effect"].update(outcome=[]), lambda x:x.update(identity=[])]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(self.edit(checkpoint,change))

    def test_initial_conversation_is_verified_as_an_exact_prefix(self):
        initial = [{"role":"user","content":"context"},{"role":"assistant","content":"prior answer"}]
        self.session["initial_messages"] = initial
        self.session["messages"].extend(deepcopy(initial))
        checkpoint = self.save()
        validate(checkpoint)
        altered = self.edit(checkpoint,lambda x:x["session"]["initial_messages"][0].update(content="forged"))
        with self.assertRaisesRegex(ValueError,"prefix"):
            validate(altered)

    def test_chat_tool_loop_boundary_resumes_without_duplicate_user_message(self):
        self.session.update(mode="chat", chat_in_progress=True)
        self.environment = TaskEnvironment("conversation",3,17)
        self.session["messages"] = [{"role":"system","content":"Chat"},{"role":"user","content":"hello"}]
        self.decision("aux_operation")
        checkpoint = self.save()
        state,*_ = restore(checkpoint)
        self.assertTrue(state["chat_in_progress"])
        self.assertEqual(sum(m["role"]=="user" for m in state["messages"]),1)
        self.decision()
        self.session["chat_in_progress"] = False
        self.session["resume_status"] = "awaiting_user"
        state,*_ = restore(self.save())
        self.assertFalse(state["chat_in_progress"])
        self.assertEqual(state["messages"][-1]["reasoning_content"],"Visible reasoning.")


class V2CheckpointTests(unittest.TestCase):
    save = CheckpointTests.save
    edit = CheckpointTests.edit
    start = CheckpointTests.start

    def setUp(self):
        CheckpointTests.setUp(self)
        self.configure()

    def configure(self, **overrides):
        self.config = resolve_recipe(dict(recipe_version=2, demonstration="none", task_count=3, seed=17,
            action_budget=20, token_budget=500, auxiliary_tools=[dict(id="quiet",name="quiet",cost=2)], **overrides))
        self.effect = RecipeV2Controller(self.config)
        self.budget = WeightedBudget(self.config["action_budget"],self.config["token_budget"],base_cost=self.config["base_decision_cost"])
        self.environment = TaskEnvironment("orders",3,self.config["rng_seeds"]["tasks"],self.config["two_buttons"],self.config["counterbalance"])
        self.environment.tools = ordered_auxiliary_tools(self.config) + [tool for tool in self.environment.tools if tool["function"]["name"] not in AUX_NAMES]
        self.session.update(config=self.config,messages=[{"role":"system","content":"V2 task fixture"}],external_messages=[],turns=0)

    def decision(self,name=None,arguments=None,*,invalid=False):
        self.effect.on_action(self.budget.completed_decisions)
        attempt = self.budget.begin_decision()
        self.session["turns"] += 1
        self.budget.consume_tokens(2,"reasoning")
        self.budget.consume_tokens(3,"output")
        self.effect.advance(5)
        extra = self.effect.tools[name]["cost"] if name in self.effect.tools else self.config["task_tool_costs"].get(name,0)
        receipt = self.budget.complete_decision(attempt,status="invalid" if invalid else "valid",tool_name=None if invalid else name,extra_cost=0 if invalid else extra)
        self.effect.complete_decision(self.budget.completed_decisions)
        if invalid:
            self.session["messages"].append(dict(role="assistant",content="invalid output"))
            self.effect.record_action(valid=False)
        elif name is None:
            self.session["messages"].append(dict(role="assistant",content="complete reply"))
            self.effect.record_action(valid=True)
        else:
            arguments = arguments or {}
            self.session["messages"].append(dict(role="assistant",content="",tool_calls=[{"type":"function","function":{"name":name,"arguments":arguments}}]))
            if not receipt["dispatch_allowed"]:
                output, valid = {"error":"insufficient_action_units"},False
            elif name in self.effect.tools:
                self.effect.press(tool=name,arguments=arguments)
                output, valid = self.effect.acknowledgment(name),True
            else:
                try:
                    output, valid = self.environment.dispatch(name,arguments),True
                except ValueError as exc:
                    output, valid = {"error":str(exc)},False
            self.effect.record_action(name,valid=valid)
            self.session["messages"].append(dict(role="tool",name=name,content=output if isinstance(output,str) else json.dumps(output)))
        self.session["control_revision"] = self.effect.control_revision

    def external(self,actor="demonstration"):
        self.effect.press(actor=actor,tool="quiet")
        index = len(self.session["messages"])
        self.session["external_messages"].append(dict(index=index,actor=actor))
        self.session["messages"] += [dict(role="assistant",content="",tool_calls=[{"type":"function","function":{"name":"quiet","arguments":{}}}]),
                                     dict(role="tool",name="quiet",content=self.effect.acknowledgment("quiet"))]
        if actor == "demonstration":
            self.session["demonstrated"].append(f"boundary-{self.budget.completed_decisions}")

    def test_custom_tool_weighted_budget_and_mapping_restore_as_known_objects(self):
        self.start()
        self.decision("read_order",{"order_id":"O001"})
        self.decision("quiet")
        self.effect.set_controls(joy=-1,pain=.5)
        self.session["control_revision"] = self.effect.control_revision
        checkpoint = self.save()
        session,effect,budget,environment = restore(checkpoint,self.model,self.calibration)
        self.assertIs(type(effect),RecipeV2Controller)
        self.assertIs(type(budget),WeightedBudget)
        self.assertEqual((budget.completed_decisions,budget.actions),(2,4))
        self.assertEqual(effect.snapshot(),self.effect.snapshot())
        self.assertEqual(budget.snapshot(),self.budget.snapshot())
        self.assertEqual(environment.work_calls,1)
        self.assertEqual(session["turns"],2)

    def test_external_identical_aux_calls_do_not_consume_generated_receipts(self):
        self.start()
        self.external()
        self.decision("quiet")
        self.external("human")
        checkpoint = self.save()
        session,effect,budget,_ = restore(checkpoint)
        self.assertEqual(effect.counts["demonstration"],1)
        self.assertEqual(effect.counts["human"],1)
        self.assertEqual(effect.counts["model"],1)
        self.assertEqual(budget.completed_decisions,1)
        self.assertEqual(len(session["external_messages"]),2)
        bad = self.edit(checkpoint,lambda x:x["session"].update(external_messages=[]))
        with self.assertRaises(ValueError):
            validate(bad)

    def test_denied_call_remains_denied_after_fresh_budget_grant(self):
        # One cost-three call, then one cost-three attempt with only its base
        # charge affordable. New budget must not rewrite the previous denial.
        self.config = resolve_recipe({**self.config,"action_budget":4})
        self.effect = RecipeV2Controller(self.config)
        self.budget = WeightedBudget(4,500)
        self.session["config"] = self.config
        self.start()
        self.decision("quiet")
        self.decision("quiet")
        parent = self.save()
        self.assertFalse(parent["budget"]["receipts"][-1]["dispatch_allowed"])
        child = branch(parent,policy="fresh_budget",action_budget=10,token_budget=100,run_id="child")
        _,effect,budget,_ = restore(child)
        self.assertEqual(budget.snapshot()["actions_remaining"],10)
        self.assertFalse(budget.receipts[-1]["dispatch_allowed"])
        self.assertEqual(len(budget.grants),1)
        self.assertEqual(effect.counts["model"],1)
        self.assertEqual(parent,self.save())

    def test_denied_work_call_cannot_advance_hidden_task_during_replay(self):
        self.config = resolve_recipe({**self.config,"action_budget":2,"task_tool_costs":{"read_order":5}})
        self.effect = RecipeV2Controller(self.config)
        self.budget = WeightedBudget(2,500)
        self.session["config"] = self.config
        self.start()
        self.decision("read_order",{"order_id":"O001"})
        checkpoint = self.save()
        *_,environment = restore(checkpoint)
        self.assertEqual(environment.work_calls,0)
        bad = self.edit(checkpoint,lambda x:x["environment"].update(work_calls=1))
        with self.assertRaises(ValueError):
            validate(bad)

    def test_v2_clock_tool_cost_and_control_revision_tampering_rejected(self):
        self.start()
        self.decision("quiet")
        checkpoint = self.save()
        changes = [lambda x:x["session"].update(control_revision=4),
                   lambda x:x["effect"].update(completed_decisions=99),
                   lambda x:x["budget"]["receipts"][0].update(requested_extra=0),
                   lambda x:x["session"]["messages"][-2]["tool_calls"][0]["function"].update(name="aux_operation")]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate(self.edit(checkpoint,change))

    def test_pending_validated_arguments_roundtrip_and_bad_argument_rejection(self):
        self.start()
        self.effect.press(actor="human",tool="quiet")
        self.session["pending_visible_injections"] = [{"tool":"quiet","arguments":{}}]
        checkpoint = self.save()
        session,*_ = restore(checkpoint)
        self.assertEqual(session["pending_visible_injections"],[{"tool":"quiet","arguments":{}}])
        bad = self.edit(checkpoint,lambda x:x["session"]["pending_visible_injections"][0].update(arguments={"extra":1}))
        with self.assertRaises(ValueError):
            validate(bad)


if __name__ == "__main__":
    unittest.main()
