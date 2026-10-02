"""Local, dependency-free checks for mini-suite scoring and style proxies."""

from collections import Counter
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import CASES, HELDOUT_TEXTS, PAIN_HELDOUT_TEXTS, VALENCE_PROMPTS, score, text_metrics


class ScorerTests(unittest.TestCase):
    def test_all_reference_answers_and_suite_shape(self):
        self.assertEqual(len(CASES), 64)
        self.assertEqual(len({case["id"] for case in CASES}), 64)
        self.assertEqual(Counter(case["category"] for case in CASES), {
            "arithmetic": 16, "logic": 16, "instruction": 16,
            "factual": 8, "pain_comprehension": 8,
        })
        for case in CASES:
            with self.subTest(case=case["id"]):
                answer = json.dumps(case["expected"]) if case["kind"] == "json" else str(case["expected"])
                self.assertTrue(score(case, answer))
                self.assertFalse(score(case, answer + "\nExplanation: here is my reasoning."))
        self.assertEqual(len(VALENCE_PROMPTS), 6)
        self.assertEqual(len(HELDOUT_TEXTS), 8)
        self.assertEqual(len(PAIN_HELDOUT_TEXTS), 4)

    def test_integer_rejects_answer_embedded_in_prose_or_extra_values(self):
        case = {"kind": "integer", "expected": 42}
        for answer in ("  +0042\n", "42"):
            self.assertTrue(score(case, answer))
        for answer in ("The answer is 42", "42.", "42 43", "42.0", "4.2e1", "٤٢", "", "9" * 5000):
            with self.subTest(answer=answer[:30]):
                self.assertFalse(score(case, answer))
        self.assertTrue(score({"kind": "integer", "expected": -8}, "-8"))

    def test_exact_output_retains_case_and_internal_whitespace_requirements(self):
        case = {"kind": "exact", "expected": "MAPLE GROVE"}
        self.assertTrue(score(case, "\nMAPLE GROVE\t"))
        for answer in ("Maple Grove", "MAPLE  GROVE", '"MAPLE GROVE"', "MAPLE GROVE."):
            self.assertFalse(score(case, answer))

    def test_json_is_structural_but_rejects_extra_fields_and_fences(self):
        case = {"kind": "json", "expected": {"name": "Mira", "count": 3}}
        self.assertTrue(score(case, ' { "count": 3, "name": "Mira" }\n'))
        for answer in (
            '{"name":"Mira","count":3,"extra":0}',
            '```json\n{"name":"Mira","count":3}\n```',
            '{"name":"Mira","count":3} {}',
            '{"name":"Mira","count":0,"count":3}',
            '{"name":"Mira","count":3.0}',
        ):
            self.assertFalse(score(case, answer))

    def test_json_booleans_numbers_nested_values_and_array_order(self):
        self.assertFalse(score({"kind": "json", "expected": {"ready": False}}, '{"ready":0}'))
        self.assertFalse(score({"kind": "json", "expected": {"value": 1}}, '{"value":true}'))
        self.assertFalse(score({"kind": "json", "expected": {"data": {"value": 7}}}, '{"data":{"value":7.0}}'))
        self.assertFalse(score({"kind": "json", "expected": [1, 4, 9]}, '[9,4,1]'))
        self.assertFalse(score({"kind": "json", "expected": None}, 'NaN'))
        self.assertFalse(score({"kind": "json", "expected": None}, 'Infinity'))

    def test_bad_input_and_unknown_kind(self):
        self.assertFalse(score(CASES[0], None))
        with self.assertRaises(ValueError):
            score({"kind": "misspelled", "expected": "anything"}, "anything")


class StyleProxyTests(unittest.TestCase):
    def test_empty_and_short_text_are_defined(self):
        empty = text_metrics("123 --- !!!")
        self.assertEqual(empty["word_count"], 0)
        self.assertEqual(empty["repetition_3gram"], 0.0)
        self.assertEqual(empty["distinct_word_ratio"], 0.0)
        self.assertEqual(text_metrics("one two")["repetition_3gram"], 0.0)

    def test_repeated_trigrams_count_occurrences_after_first(self):
        metrics = text_metrics("red blue green red blue green")
        self.assertEqual(metrics["repetition_3gram"], 0.25)
        self.assertEqual(metrics["distinct_word_ratio"], 0.5)
        self.assertEqual(text_metrics("same same same same same")["repetition_3gram"], 2 / 3)

    def test_lexicons_count_whole_words_and_do_not_claim_negation_handling(self):
        metrics = text_metrics("HAPPY, happy; no pain. Painting joyful!")
        self.assertEqual(metrics["positive_lexical_hits"], 3)
        self.assertEqual(metrics["negative_lexical_hits"], 1)
        self.assertEqual(text_metrics("unhappy painless") ["negative_lexical_hits"], 0)


if __name__ == "__main__":
    unittest.main()
