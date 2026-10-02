import unittest
from lab.compatibility import canonical_hash, verify_legacy_contract


class PublishedCompatibilityTests(unittest.TestCase):
    def test_all_published_protocols_and_visible_inputs_remain_exact(self):
        result = verify_legacy_contract(include_archives=False)
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["checked"], dict(protected_files=0, protocols=3,
                                              recipes=10, visible_payloads=24))

    def test_canonical_comparison_preserves_values_not_dict_order(self):
        self.assertEqual(canonical_hash({"a": 1, "b": [2, 3]}),
                         canonical_hash({"b": [2, 3], "a": 1}))
        self.assertNotEqual(canonical_hash({"a": 1}), canonical_hash({"a": 1.0}))
        with self.assertRaises(ValueError):
            canonical_hash({"x": float("nan")})


if __name__ == "__main__":
    unittest.main()
