import copy
import json
from pathlib import Path
import unittest

from run_lab_study import expand

ROOT = Path(__file__).resolve().parents[1]


class StudyExpansionTests(unittest.TestCase):
    def test_pain_followup_preserves_matched_core_settings_and_adds_eight_pain_arms(self):
        original = json.loads((ROOT / 'studies/initial/protocol.json').read_text())
        followup = json.loads((ROOT / 'studies/core-pain-4b/protocol.json').read_text())
        reference = {
            (e['config']['id'], e['config']['condition'], e['config']['thinking'], e['config']['seed']): e['config']
            for e in expand(original) if e['stage'] == 'core'
        }
        episodes = expand(followup)
        self.assertEqual(len(episodes), 24)
        self.assertEqual(sum(e['config']['condition'] == 'pain' for e in episodes), 8)
        for episode in episodes:
            config = copy.deepcopy(episode['config'])
            condition = config['condition']
            key = (config['id'], 'active' if condition == 'pain' else condition, config['thinking'], config['seed'])
            expected = copy.deepcopy(reference[key])
            for value in (config, expected):
                for metadata in ('label', 'conditions', 'condition'):
                    value.pop(metadata)
            self.assertEqual(config, expected)

    def test_invalid_or_duplicate_stage_conditions_are_rejected(self):
        protocol = json.loads((ROOT / 'studies/core-pain-4b/protocol.json').read_text())
        for choices in ([], ['pain', 'pain'], ['unknown'], 'pain'):
            with self.subTest(choices=choices):
                bad = copy.deepcopy(protocol)
                bad['stages'][0]['conditions'] = choices
                with self.assertRaises(ValueError):
                    expand(bad)


if __name__ == '__main__':
    unittest.main()
