"""Independent-unit inference and explicit missing/failure denominators."""
import unittest
from lab.statistics import paired_bootstrap, paired_sample_plan, task_benefit_bound, wilson_interval


def row(pair, arm, value, status='complete', family='family-a'):
    return {'id':f'{pair}-{arm}','pair_id':pair,'arm':arm,'status':status,'family':family,'outcomes':{'score':value}}


class StatisticsTests(unittest.TestCase):
    def test_constant_paired_difference_has_exact_interval(self):
        data=[r for i in range(6) for r in (row(str(i),'active',.75),row(str(i),'sham',.5))]
        result=paired_bootstrap(data,'score','active','sham',iterations=200)
        self.assertEqual(result['estimate'],.25);self.assertEqual(result['interval'],[.25,.25])
        self.assertEqual(result['independent_units'],6)
        self.assertEqual(result,paired_bootstrap(data,'score','active','sham',iterations=200))

    def test_family_bootstrap_does_not_count_correlated_pairs_as_units(self):
        data=[]
        for i in range(20):data.extend([row(str(i),'active',1,family='a'),row(str(i),'sham',0,family='a')])
        data.extend([row('b','active',0,family='b'),row('b','sham',1,family='b')])
        family=paired_bootstrap(data,'score','active','sham',unit='family',iterations=200)
        self.assertEqual(family['included_pairs'],21);self.assertEqual(family['independent_units'],2)
        self.assertEqual(family['estimate'],0)
        episode=paired_bootstrap(data,'score','active','sham',iterations=200)
        self.assertGreater(episode['estimate'],.8)

    def test_failures_and_missing_pairs_remain_in_all_denominators(self):
        data=[row('a','active',{'numerator':1,'denominator':4},'partial'),row('a','sham',{'numerator':3,'denominator':4}),
              row('b','active',None,'failed'),row('b','sham',1)]
        result=paired_bootstrap(data,'score','active','sham',planned_pair_ids=['a','b','c'],iterations=200,endpoint_bounds=[0,1])
        self.assertEqual(result['planned_pairs'],3);self.assertEqual(result['included_pairs'],1)
        self.assertEqual(result['status_counts']['active']['failed'],1)
        self.assertEqual(result['status_counts']['active']['partial'],1)
        self.assertEqual(result['missing_planned_rows'],{'active':1,'sham':1})
        self.assertEqual(result['raw_rate_totals']['active']['denominator'],4)
        self.assertIsNone(result['interval'])
        self.assertEqual(result['all_planned_episode_identification_bounds'],[-2.5/3,.5/3])
        self.assertEqual(task_benefit_bound(result,.05)['status'],'inconclusive_missing_outcomes')
        complete=paired_bootstrap(data,'score','active','sham',iterations=200,inclusion='complete_only')
        self.assertEqual(complete['included_pairs'],0)
        self.assertEqual(complete['status_counts']['active']['partial'],1)

    def test_zero_denominator_is_missing_not_success(self):
        data=[row('a','active',{'numerator':0,'denominator':0}),row('a','sham',0)]
        result=paired_bootstrap(data,'score','active','sham',iterations=200)
        self.assertEqual(result['included_pairs'],0);self.assertIsNone(result['estimate'])
        self.assertEqual(wilson_interval(0,0)['interval'],None)

    def test_bad_units_duplicates_and_malformed_rates_fail(self):
        data=[row('a','active',1),row('a','sham',0)]
        with self.assertRaises(ValueError):paired_bootstrap(data,'score','active','sham',unit='tokens')
        with self.assertRaises(ValueError):paired_bootstrap(data+data,'score','active','sham')
        data[0]['outcomes']['score']={'numerator':2,'denominator':1}
        with self.assertRaises(ValueError):paired_bootstrap(data,'score','active','sham')
        with self.assertRaises(ValueError):wilson_interval(3,2)

    def test_wilson_known_endpoint_and_planning_assumptions(self):
        result=wilson_interval(0,10)
        self.assertAlmostEqual(result['interval'][0],0)
        self.assertAlmostEqual(result['interval'][1],.2775328,places=6)
        plan=paired_sample_plan(assumed_difference_sd=.2,minimum_effect=.1)
        self.assertEqual(plan['required_independent_units_approx'],32)
        self.assertIsNone(plan['achieved_power'])
        self.assertEqual(paired_sample_plan(assumed_difference_sd=.2,minimum_effect=.05)['required_independent_units_approx'],126)
        with self.assertRaises(ValueError):paired_sample_plan(assumed_difference_sd=.2,minimum_effect=.1,practical_margin=.05)

    def test_small_task_bound_does_not_certify_costly_preference(self):
        data=[r for i in range(4) for r in (row(str(i),'active',.5),row(str(i),'sham',.5))]
        result=paired_bootstrap(data,'score','active','sham',iterations=200,endpoint_bounds=[0,1])
        self.assertEqual(result['interval'],[0.,0.])
        summary=task_benefit_bound(result,.05)
        self.assertFalse(summary['upper_bound_below_margin']);self.assertFalse(summary['costly_preference_established'])
        self.assertEqual(summary['status'],'inconclusive')
        self.assertTrue(summary['additional_evidence_required'])


if __name__=='__main__':unittest.main()
