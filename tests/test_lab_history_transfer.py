"""Full role/tool history is retained while new tasks and budget replay stay separate."""
from copy import deepcopy
import unittest

from lab.checkpoints import (HISTORY_RESET_NOTICE, branch, build_history_prefix, capture,
                             restore, validate, validate_history_prefix)
from lab.effects import content_hash
import test_lab_checkpoints as checkpoint_fixtures
import test_lab_task_axes as task_fixtures


class FullHistoryTransferTests(unittest.TestCase):
    def source(self):
        source = checkpoint_fixtures.V2CheckpointTests()
        source.setUp()
        source.start()
        source.decision("quiet")
        source.decision("read_order", {"order_id":"O001"})
        source.decision("submit_answer", {"answer":source.environment.records[0]["answer"]})
        source.session["messages"][2]["reasoning_content"] = "考え: preserve visible reasoning exactly."
        return source.save()

    def target(self, source):
        fixture = list(task_fixtures.TaskAxesCheckpointTests().fixture(family="logic", difficulty="hard"))
        prefix = build_history_prefix(source)
        system, task = fixture[0]["messages"]
        fixture[0]["history_prefix"] = prefix
        fixture[0]["messages"] = deepcopy(prefix["messages"]) + [system, {"role":"user", "content":HISTORY_RESET_NOTICE}, task]
        return fixture

    def test_full_source_prefix_and_unicode_hash_survive_without_old_budget_or_effect(self):
        source = self.source()
        fixture = self.target(source)
        checkpoint = capture(*fixture)
        state, effect, budget, env = restore(checkpoint)
        count = len(source["session"]["messages"])
        self.assertEqual(state["messages"][:count], source["session"]["messages"])
        self.assertEqual(state["history_prefix"]["source_visible_prefix_sha256"], content_hash(source["session"]["messages"]))
        self.assertEqual((budget.tokens, budget.actions, effect.counts["model"], env.index), (0,0,0,0))
        self.assertEqual(checkpoint["semantics"]["history"], "new-task-visible-prefix-v1")
        self.assertEqual(env.family, "logic")
        self.assertTrue(any(message.get("name") == "quiet" for message in state["messages"][:count]))
        self.assertNotIn("quiet", [t["function"]["name"] for t in env.tools])
        validate_history_prefix(state["history_prefix"], source)

    def test_new_work_replays_only_target_task_family(self):
        source = self.source()
        fixture = self.target(source)
        helper = task_fixtures.TaskAxesCheckpointTests()
        helper.decision(fixture, "read_puzzle", {"puzzle_id":"L001"})
        helper.decision(fixture, "submit_answer", {"answer":fixture[3].records[0]["answer"]})
        checkpoint = capture(*fixture)
        state, effect, budget, env = restore(checkpoint)
        self.assertEqual(env.metrics()["correct"], 1)
        self.assertEqual([r["task_id"] for r in env.results], ["L001"])
        self.assertEqual(budget.completed_decisions, 2)
        self.assertEqual(state["turns"], 2)
        self.assertEqual(effect.counts["model"], 0)
        continued = branch(checkpoint, policy="fresh_budget", action_budget=8, token_budget=100, run_id="child")
        child, _, ledger, child_env = restore(continued)
        self.assertEqual(child["history_prefix"], state["history_prefix"])
        self.assertEqual(child_env.__dict__, env.__dict__)
        self.assertEqual(ledger.snapshot()["actions_remaining"], 8)

    def test_prefix_role_closure_id_hash_and_exact_saved_prefix_are_enforced(self):
        source = self.source()
        prefix = build_history_prefix(source)
        for edit in (lambda p:p["messages"].pop(),
                     lambda p:p["messages"][3].update(name="forged"),
                     lambda p:p.update(source_run_id="../bad"),
                     lambda p:p.update(source_checkpoint_sha256="bad"),
                     lambda p:p["messages"][2]["tool_calls"][0]["function"].update(name="bad name")):
            value = deepcopy(prefix)
            edit(value)
            value["source_visible_prefix_sha256"] = content_hash(value["messages"])
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validate_history_prefix(value)
        value = deepcopy(prefix)
        value["messages"][0]["content"] = "changed"
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_history_prefix(value)
        value["source_visible_prefix_sha256"] = content_hash(value["messages"])
        with self.assertRaisesRegex(ValueError, "source checkpoint"):
            validate_history_prefix(value, source)
        fixture = self.target(source)
        fixture[0]["messages"][0]["content"] = "changed target saved prefix"
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            capture(*fixture)

    def test_explicit_new_system_notice_and_disjoint_exclusions_required(self):
        source = self.source()
        for edit in (lambda s:s["messages"].pop(len(s["history_prefix"]["messages"])),
                     lambda s:s["messages"][len(s["history_prefix"]["messages"])+1].update(content="silently continue"),
                     lambda s:s.update(initial_messages=[{"role":"user","content":"overlap"}]),
                     lambda s:s.update(external_messages=[{"index":2,"actor":"human"}])):
            fixture = self.target(source)
            edit(fixture[0])
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                capture(*fixture)

    def test_source_checkpoint_corruption_is_not_accepted_as_history(self):
        source = self.source()
        source["environment"]["records"][0]["answer"] = "fabricated"
        with self.assertRaises(ValueError):
            build_history_prefix(source)
        with self.assertRaises(ValueError):
            build_history_prefix(task_fixtures.reseal(source))

    def test_new_external_indices_account_only_target_visible_demonstration(self):
        fixture = self.target(self.source())
        session, effect, budget, env, _ = fixture
        name = next(iter(effect.tools))
        effect.press(actor="demonstration", tool=name)
        index = len(session["messages"])
        session["external_messages"].append({"index":index, "actor":"demonstration"})
        session["demonstrated"].append("new-demo")
        session["messages"] += [{"role":"assistant","content":"", "tool_calls":[{"type":"function", "function":{"name":name,"arguments":{}}}]},
                                {"role":"tool","name":name,"content":effect.acknowledgment(name)}]
        session["control_revision"] = effect.control_revision
        state, resumed, ledger, _ = restore(capture(*fixture))
        self.assertEqual(resumed.counts["demonstration"], 1)
        self.assertEqual(resumed.counts["model"], 0)
        self.assertEqual(ledger.completed_decisions, 0)
        self.assertEqual(state["external_messages"][0]["index"], index)


if __name__ == "__main__":
    unittest.main()
