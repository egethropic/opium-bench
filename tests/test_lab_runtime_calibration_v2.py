"""Actual research hooks, token offsets and immutable calibration lifecycle on CPU."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from types import SimpleNamespace

from test_lab_calibration import fixture, configuration
from test_lab_runtime import Tokenizer, runtime
from lab.runtime import Cancelled
from lab.calibration_validation import diagnostic_specs


class ResearchTokenizer(Tokenizer):
    def __init__(self, corpus):
        super().__init__()
        self.rows = corpus['rows']
        self.incomplete = False

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False):
        row = next((r for r in self.rows if text.startswith(r['text'])), None)
        label = (10 if row['concept'] == 'pain' else 20) if row and row['label'] else 30
        result = SimpleNamespace(input_ids=torch.tensor([[label, 30, 30]]), attention_mask=torch.ones((1, 3), dtype=torch.long))
        if return_offsets_mapping:
            start, end = row['span'] if row else (0, len(text))
            offsets = [[0, start], [start, end], [end, len(text)]]
            if self.incomplete: offsets[1] = [0, 0]
            result.offset_mapping = torch.tensor([offsets])
        return result


def research_runtime(corpus=None):
    corpus = corpus or fixture()
    rt = runtime()
    rt.tokenizer = ResearchTokenizer(corpus)
    return rt, corpus


def config(corpus):
    return {**configuration(poolings=['final', 'mean', 'span'], ridge_alphas=[1.]),
            'schema_version': 2, 'preset': 'research', 'corpus': corpus, 'doses': [0., .25],
            'validation_pairs_per_concept': 1, 'continuation_tokens': 8}


class ResearchRuntimeTests(unittest.TestCase):
    def test_research_extraction_saves_separate_fit_and_loadable_evidence(self):
        rt, corpus = research_runtime(); events = []
        with TemporaryDirectory() as temp:
            target = Path(temp)/'fitted'
            metadata = rt.calibrate(config(corpus), target, events.append)
            package = rt._package(target)
            self.assertEqual(metadata['schema_version'], 2)
            self.assertEqual(metadata['status'], 'unvalidated')
            self.assertEqual(metadata['selected_dose'], 0.)
            self.assertIn('probe_pain_weight', package['vectors'])
            self.assertEqual(metadata['corpus']['splits']['heldout']['families'], 2)
            self.assertEqual(metadata['supported_thinking_modes'], [])
            self.assertEqual(sum(e.get('stage') == 'research_extract' for e in events), len(corpus['rows']))
            with np.load(target/'activations.npz') as arrays:
                self.assertEqual(set(arrays.files), {'1:final', '1:mean', '1:span', '2:final', '2:mean', '2:span'})
            self.assertTrue(all(not block._forward_hooks for block in rt.blocks))
            original_hash = hashlib.sha256((target/'calibration.json').read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'immutable'): rt.calibrate(config(corpus), target)
            self.assertEqual(hashlib.sha256((target/'calibration.json').read_bytes()).hexdigest(), original_hash)

    def test_incomplete_spans_and_cancellation_leave_partial_evidence_not_success(self):
        for kind in ('offset', 'cancel'):
            rt, corpus = research_runtime(); events = []
            rt.tokenizer.incomplete = kind == 'offset'
            with TemporaryDirectory() as temp:
                target = Path(temp)/'partial'
                with self.assertRaises((ValueError, Cancelled)):
                    rt.calibrate(config(corpus), target, events.append,
                                 lambda: kind == 'cancel' and sum(e.get('stage') == 'research_extract' for e in events) >= 2)
                self.assertFalse((target/'calibration.json').exists())
                progress = json.loads((target/'progress.json').read_text())
                self.assertEqual(progress['status'], 'cancelled' if kind == 'cancel' else 'failed')
                self.assertTrue(all(not block._forward_hooks for block in rt.blocks))

    def test_loader_rejects_changed_template_corpus_config_or_vectors(self):
        rt, corpus = research_runtime()
        with TemporaryDirectory() as temp:
            target = Path(temp)/'fitted'; rt.calibrate(config(corpus), target)
            rt._package(target)  # Populate cache before tampering checks.
            original_template = rt.tokenizer.chat_template
            rt.tokenizer.chat_template = 'changed'
            # Cache is normally cleared with model load/unload; direct test
            # mutation emulates a new template without the normal load path.
            rt._calibration_cache.clear()
            with self.assertRaisesRegex(ValueError, 'tokenizer/template'): rt._package(target)
            rt.tokenizer.chat_template = original_template
            for name in ('corpus.json', 'activations.npz', 'vectors.npz'):
                original = (target/name).read_bytes(); rt._calibration_cache.clear()
                with (target/name).open('ab') as file: file.write(b' ')
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'checksum'): rt._package(target)
                (target/name).write_bytes(original)
            manifest = json.loads((target/'calibration.json').read_text())
            manifest['config']['seed'] += 1
            (target/'calibration.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'identity'): rt._package(target)

    def test_schema2_generation_affine_readout_is_actual_and_sham_bitwise(self):
        rt, corpus = research_runtime()
        with TemporaryDirectory() as temp:
            target = Path(temp)/'fitted'; rt.calibrate(config(corpus), target)
            package = rt._package(target)
            baseline = rt.generate([{'role': 'user', 'content': 'hello'}], [], {'temperature': 0, 'max_new_tokens': 3}, None)
            events = []
            measured = rt.generate([{'role': 'user', 'content': 'hello'}], [], {'temperature': 0, 'max_new_tokens': 3}, target, emit=events.append)
            self.assertEqual(baseline['token_ids'], measured['token_ids'])
            self.assertEqual(events[0]['measurements']['relative_delta'], 0.)
            before = np.array([1., 1., 2., 1.]) * 1.1**(package['metadata']['layer']+1)
            expected = before @ package['vectors']['probe_pain_weight'] + package['vectors']['probe_pain_bias']
            self.assertAlmostEqual(events[0]['measurements']['pre']['pain'], expected, places=5)

    def test_validation_records_real_signs_matching_and_blinded_outputs(self):
        rt, corpus = research_runtime(); events = []
        with TemporaryDirectory() as temp:
            fitted, validated = Path(temp)/'fitted', Path(temp)/'validated'
            rt.calibrate(config(corpus), fitted)
            source_hash = hashlib.sha256((fitted/'calibration.json').read_bytes()).hexdigest()
            metadata = rt.validate_calibration({'calibration_dir': str(fitted)}, validated, events.append)
            rt._package(validated)
            self.assertEqual(hashlib.sha256((fitted/'calibration.json').read_bytes()).hexdigest(), source_hash)
            data = json.loads((validated/'diagnostics.json').read_text())
            self.assertEqual(len(data['records']), 36)
            self.assertTrue(all(r['next_token_kl'] == 0 and r['relative_delta'] == 0 for r in data['records'] if r['condition'] == 'zero'))
            self.assertTrue(any(r['next_token_kl'] > 0 for r in data['records']))
            random = [r for r in data['records'] if r['specification']['operator'] == 'random_matched']
            self.assertTrue(all(r['random_norm_match_error'] < 1e-6 for r in random))
            self.assertEqual(metadata['operating_range']['split'], 'selection')
            self.assertEqual(metadata['continuation_scoring']['scored'], 0)
            self.assertEqual(metadata['continuation_scoring']['status'], 'awaiting_independent_human_ratings')
            sheet = json.loads((validated/'scoring-sheet.json').read_text())
            self.assertTrue(sheet['samples'])
            self.assertTrue(all('condition' not in sample and 'generation' not in sample for sample in sheet['samples']))
            self.assertTrue(all(not block._forward_hooks for block in rt.blocks))
            stages = [e.get('stage') for e in events]
            self.assertLess(stages.index('operating_range_locked'), next(i for i, e in enumerate(events) if e.get('split') == 'heldout'))

    def test_validation_cancel_is_partial_and_does_not_rewrite_source(self):
        rt, corpus = research_runtime(); events = []
        with TemporaryDirectory() as temp:
            fitted, partial = Path(temp)/'fitted', Path(temp)/'partial'
            rt.calibrate(config(corpus), fitted)
            source = (fitted/'calibration.json').read_bytes()
            with self.assertRaises(Cancelled):
                rt.validate_calibration({'calibration_dir': str(fitted)}, partial, events.append,
                                        lambda: sum(e.get('stage') == 'research_diagnostics' for e in events) >= 2)
            self.assertFalse((partial/'calibration.json').exists())
            self.assertEqual((fitted/'calibration.json').read_bytes(), source)
            self.assertEqual(json.loads((partial/'diagnostics.json').read_text())['status'], 'cancelled')
            self.assertTrue(all(not block._forward_hooks for block in rt.blocks))


if __name__ == '__main__': unittest.main()
