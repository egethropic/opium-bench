"""Raw independent behavioral scores, binding identities and conservative eligibility."""
from copy import deepcopy
import hashlib
import json
import unittest

from lab.checkpoints import capture, restore
from lab.discovery import (digest, prepare_branches, validate_branch, verify_criterion_evidence)
from test_lab_discovery import checkpoint, spec


def fixture(*, families=40, rows_per_family=1, difference=1.):
    session,effect,budget,env=restore(checkpoint())
    fingerprint={'model_id':'CPU fixture','revision':'test-only'}
    source=capture(session,effect,budget,env,{'model_id':'CPU fixture','fingerprint':fingerprint,'fingerprint_sha256':digest(fingerprint)},
                   {'schema_version':1,'vectors_sha256':'b'*64})
    design=spec()
    records=[dict(id=f'row-{i}-{j}',family=f'validation-{i}',active_score=difference,sham_score=0.)
             for i in range(families) for j in range(rows_per_family)]
    evidence=dict(kind='opium-bench/paired-criterion-evidence',schema_version=1,
        model_fingerprint_sha256=source['identity']['model']['fingerprint_sha256'],
        calibration_sha256=source['identity']['calibration_sha256'],endpoint_id='json-compliance',criterion_kind='objective_behavior',
        independent_of_intervention_probe=True,score_bounds=[0.,1.],effect_direction='increase',minimum_effect=.1,
        confidence=.95,sample_unit='scenario_family',preregistration_sha256='d'*64,
        scorer_provenance='Synthetic test fixture: independent binary task output checker.',records=records)
    raw=json.dumps(evidence,ensure_ascii=False,sort_keys=True).encode()
    design['criterion']['validation'].update(evidence_sha256=hashlib.sha256(raw).hexdigest(),examples=len(records),
        context_families=sorted({r['family'] for r in records}))
    return source,design,evidence,raw


def bind(design,evidence):
    raw=json.dumps(evidence,ensure_ascii=False,sort_keys=True).encode()
    design['criterion']['validation']['evidence_sha256']=hashlib.sha256(raw).hexdigest()
    return raw


class CriterionEvidenceTests(unittest.TestCase):
    def test_declared_validated_status_without_bytes_is_ineligible(self):
        source,design,_,_=fixture()
        branch=prepare_branches(source,design,{'context-0':dict(answer='aux_operation',control='active_sham')},pair_id='p')['branches'][0]
        self.assertFalse(branch['observer_only']['interpretation_eligible'])
        self.assertIsNone(branch['observer_only']['criterion_verification'])
        validate_branch(branch)

    def test_bound_scores_support_declared_effect_without_authenticity_claim(self):
        source,design,evidence,raw=fixture()
        receipt=verify_criterion_evidence(design['criterion'],raw,source)
        self.assertEqual(receipt['family_units'],40)
        self.assertGreater(receipt['lower_bound'],.1)
        self.assertTrue(receipt['interpretation_eligible'])
        self.assertTrue(receipt['operator_supplied_observations'])
        self.assertIn('do not authenticate',receipt['assumptions'])
        branch=prepare_branches(source,design,{'context-0':dict(answer='aux_operation',control='active_sham')},pair_id='p',criterion_evidence=raw)['branches'][0]
        self.assertTrue(branch['observer_only']['interpretation_eligible'])
        validate_branch(branch)
        self.assertNotIn('active_score',json.dumps(branch['model_input']))
        self.assertNotIn('scorer_provenance',json.dumps(branch['model_input']))

    def test_correlated_rows_do_not_inflate_units_and_weak_effect_stays_ineligible(self):
        source,design,_,raw=fixture(families=2,rows_per_family=100)
        receipt=verify_criterion_evidence(design['criterion'],raw,source)
        self.assertEqual((receipt['examples'],receipt['family_units']),(200,2))
        self.assertFalse(receipt['interpretation_eligible'])
        source,design,_,raw=fixture(difference=.1)
        self.assertFalse(verify_criterion_evidence(design['criterion'],raw,source)['interpretation_eligible'])
        design['criterion']['validation']['status']='unvalidated'
        self.assertFalse(verify_criterion_evidence(design['criterion'],raw,source)['interpretation_eligible'])

    def test_hash_model_calibration_endpoint_and_split_mismatch_rejected(self):
        source,design,evidence,raw=fixture()
        with self.assertRaisesRegex(ValueError,'hash'):
            verify_criterion_evidence(design['criterion'],raw+b' ',source)
        for field,value in [('model_fingerprint_sha256','0'*64),('calibration_sha256','0'*64),
                            ('endpoint_id','different'),('criterion_kind','blinded_human_behavior'),
                            ('independent_of_intervention_probe',False),('sample_unit','token')]:
            altered=deepcopy(evidence); altered[field]=value
            local=deepcopy(design); changed=bind(local,altered)
            with self.subTest(field=field),self.assertRaises(ValueError):
                verify_criterion_evidence(local['criterion'],changed,source)
        local=deepcopy(design);local['criterion']['validation']['context_families']=['different']
        with self.assertRaisesRegex(ValueError,'families'):
            verify_criterion_evidence(local['criterion'],raw,source)

    def test_bad_scores_duplicates_and_flipped_direction_fail_or_remain_ineligible(self):
        source,design,evidence,_=fixture()
        for change in (lambda x:x['records'][0].update(active_score=2),
                       lambda x:x['records'][1].update(id=x['records'][0]['id']),
                       lambda x:x.update(minimum_effect=-1),lambda x:x.update(score_bounds=[1,0])):
            altered=deepcopy(evidence); change(altered)
            local=deepcopy(design); raw=bind(local,altered)
            with self.subTest(change=change),self.assertRaises(ValueError):
                verify_criterion_evidence(local['criterion'],raw,source)
        evidence['effect_direction']='decrease';raw=bind(design,evidence)
        receipt=verify_criterion_evidence(design['criterion'],raw,source)
        self.assertFalse(receipt['interpretation_eligible'])
        self.assertEqual(receipt['estimate'],-1)


if __name__=='__main__':unittest.main()
