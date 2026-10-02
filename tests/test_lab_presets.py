"""Local recipe asset round trips, versioning and evidence identity."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from lab.presets import save, load, catalog
from lab.recipes_v2 import default_recipe


class PresetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def test_recipe_roundtrip_is_immutable_and_detached(self):
        recipe = default_recipe()
        record = save(self.root, 'recipe', 'custom-v1', recipe)
        record['value']['action_budget'] = 1
        saved = load(self.root, 'recipe', 'custom-v1')
        self.assertEqual(saved['value']['action_budget'], recipe['action_budget'])
        with self.assertRaisesRegex(ValueError, 'already exists'): save(self.root, 'recipe', 'custom-v1', recipe)
        self.assertEqual(catalog(self.root)[0]['content_sha256'], saved['content_sha256'])
        self.assertNotIn('value', catalog(self.root)[0])

    def test_effect_and_tool_bundles_keep_references_and_validate_on_composition(self):
        recipe = default_recipe()
        for kind, value in [('effect_presets',recipe['effect_presets']), ('auxiliary_tools',recipe['auxiliary_tools'])]:
            record = save(self.root, kind, 'saved', value)
            self.assertEqual(load(self.root, kind, 'saved')['value'], record['value'])
        bad = deepcopy(recipe['auxiliary_tools']); bad[0]['name'] = 'read_order'
        with self.assertRaises(ValueError): save(self.root, 'auxiliary_tools', 'invalid', bad)

    def test_tampering_traversal_and_future_versions_fail(self):
        record = save(self.root, 'recipe', 'custom', default_recipe())
        path = self.root/'presets'/'recipe'/'custom.json'
        for version in (99, 1):
            changed = deepcopy(record); changed['schema_version'] = version; changed['value']['action_budget'] += 1
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): load(self.root, 'recipe', 'custom')
        self.assertEqual(catalog(self.root), [])
        for identifier in ('../escape','/absolute','',None):
            with self.assertRaises(ValueError): save(self.root, 'recipe', identifier, default_recipe())


if __name__ == '__main__': unittest.main()
