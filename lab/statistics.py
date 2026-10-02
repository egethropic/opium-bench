"""Small auditable statistics on independent episodes/families, never tokens.

Missing outcomes stay missing. A planning approximation is not a declaration
that an existing study has power or that an insignificant contrast is equivalent.
"""
from collections import Counter, defaultdict
import math
import random
from statistics import NormalDist, mean

STATUSES = {'complete', 'failed', 'partial', 'cancelled', 'stopped', 'pending'}
UNITS = {'episode_pair', 'family'}


def _number(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValueError(f'{name} is outside its declared bounds')
    return float(value)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value


def _quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high]-ordered[low]) * (position-low)


def wilson_interval(successes, denominator, confidence=.95):
    """A binomial interval only for a declared independent Bernoulli unit."""
    denominator = _integer(denominator, 'denominator', 0, 10**12)
    successes = _integer(successes, 'successes', 0, denominator)
    confidence = _number(confidence, 'confidence', .5, .9999)
    if denominator == 0:
        return {'successes': successes, 'denominator': denominator, 'rate': None, 'interval': None, 'confidence': confidence, 'method': 'Wilson score'}
    z = NormalDist().inv_cdf((1+confidence)/2)
    p, n = successes/denominator, denominator
    center = (p + z*z/(2*n))/(1+z*z/n)
    width = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return {'successes': successes, 'denominator': n, 'rate': p, 'interval': [max(0., center-width), min(1., center+width)], 'confidence': confidence, 'method': 'Wilson score'}


def _endpoint(record, name):
    outcomes = record.get('outcomes', {})
    if not isinstance(outcomes, dict): raise ValueError('Outcomes must be an object')
    value = outcomes.get(name)
    if value is None:
        return None, None
    if isinstance(value, dict):
        if set(value) != {'numerator', 'denominator'}:
            raise ValueError('Rate endpoints require explicit numerator and denominator')
        numerator = _number(value['numerator'], 'endpoint numerator', 0)
        denominator = _number(value['denominator'], 'endpoint denominator', 0)
        if numerator > denominator:
            raise ValueError('Endpoint numerator exceeds its denominator')
        return (numerator/denominator if denominator else None), {'numerator': numerator, 'denominator': denominator}
    return _number(value, 'endpoint value'), None


def paired_bootstrap(records, endpoint, arm_a, arm_b, *, unit='episode_pair', seed=1729,
                     iterations=5000, confidence=.95, planned_pair_ids=None,
                     inclusion='all_observed', endpoint_bounds=None):
    """Estimate arm_a minus arm_b using paired episode/family resampling.

    all_observed retains measured outcomes from failed/partial episodes. The
    optional complete_only rule is explicit and reports excluded denominators.
    Neither rule converts missing failures into successes or zeroes. Endpoint
    rates are first calculated per episode; the estimand is the equally weighted
    paired difference, not a token-weighted or pooled-binomial rate.
    """
    if unit not in UNITS or inclusion not in {'all_observed', 'complete_only'}:
        raise ValueError('Declare episode_pair/family and a supported inclusion rule')
    if not isinstance(records, list) or len(records) > 100000 or not isinstance(endpoint, str) or not endpoint or not isinstance(arm_a, str) or not isinstance(arm_b, str) or not arm_a or not arm_b or arm_a == arm_b:
        raise ValueError('Invalid paired observations/endpoint/arms')
    seed = _integer(seed, 'analysis seed', 0, 2**63-1)
    iterations = _integer(iterations, 'bootstrap iterations', 100, 100000)
    confidence = _number(confidence, 'confidence', .5, .9999)
    by_pair = defaultdict(dict); ids = set(); status_counts = {arm: Counter() for arm in (arm_a, arm_b)}
    rate_totals = {arm: {'numerator': 0., 'denominator': 0., 'observed_rate_episodes': 0} for arm in (arm_a, arm_b)}
    for row in records:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k] for k in ('id', 'pair_id', 'arm', 'status')):
            raise ValueError('Each observation needs id, pair_id, arm and status')
        if row['id'] in ids or row['status'] not in STATUSES:
            raise ValueError('Duplicate observation ID or unknown status')
        ids.add(row['id'])
        if row['arm'] not in (arm_a, arm_b):
            continue
        if row['arm'] in by_pair[row['pair_id']]:
            raise ValueError('Duplicate arm within one episode pair')
        if unit == 'family' and (not isinstance(row.get('family'), str) or not row['family']):
            raise ValueError('Family-level inference requires an explicit family for every observation')
        value, counts = _endpoint(row, endpoint)
        by_pair[row['pair_id']][row['arm']] = (row, value)
        status_counts[row['arm']][row['status']] += 1
        if counts is not None:
            for key in ('numerator', 'denominator'): rate_totals[row['arm']][key] += counts[key]
            rate_totals[row['arm']]['observed_rate_episodes'] += 1
    if planned_pair_ids is None:
        planned = sorted(by_pair)
    else:
        if not isinstance(planned_pair_ids, list) or any(not isinstance(p, str) or not p for p in planned_pair_ids) or len(planned_pair_ids) != len(set(planned_pair_ids)):
            raise ValueError('Planned pair IDs must be distinct nonempty strings')
        planned = list(planned_pair_ids)
        if set(by_pair) - set(planned):
            raise ValueError('Observed pair absent from the frozen plan')
    if not planned:
        raise ValueError('At least one planned or observed pair is required')
    differences, excluded = [], []
    for pair_id in planned:
        pair = by_pair.get(pair_id, {})
        missing = [arm for arm in (arm_a, arm_b) if arm not in pair or pair[arm][1] is None]
        if missing:
            excluded.append({'pair_id': pair_id, 'reason': 'missing_outcome', 'arms': missing})
            continue
        a, b = pair[arm_a], pair[arm_b]
        if unit == 'family' and a[0]['family'] != b[0]['family']:
            raise ValueError('Paired arms have inconsistent family IDs')
        if inclusion == 'complete_only' and any(item[0]['status'] != 'complete' for item in (a, b)):
            excluded.append({'pair_id': pair_id, 'reason': 'noncomplete_status', 'arms': [arm for arm in (arm_a, arm_b) if pair[arm][0]['status'] != 'complete']})
            continue
        differences.append({'pair_id': pair_id, 'family': a[0].get('family'), 'difference': a[1]-b[1], 'a': a[1], 'b': b[1]})
    if unit == 'family':
        groups = defaultdict(list)
        for row in differences: groups[row['family']].append(row['difference'])
        units = [{'id': key, 'difference': mean(values), 'pairs': len(values)} for key, values in sorted(groups.items())]
    else:
        units = [{'id': row['pair_id'], 'difference': row['difference'], 'pairs': 1} for row in differences]
    rng = random.Random(seed)
    estimate = mean(row['difference'] for row in units) if units else None
    interval = None
    if len(units) >= 2:
        samples = [mean(units[rng.randrange(len(units))]['difference'] for _ in units) for _ in range(iterations)]
        tail = (1-confidence)/2
        interval = [_quantile(samples, tail), _quantile(samples, 1-tail)]
    result = {'endpoint': endpoint, 'contrast': f'{arm_a} minus {arm_b}', 'arms': [arm_a, arm_b],
              'unit': unit, 'inclusion': inclusion, 'method': 'paired percentile bootstrap', 'seed': seed,
              'iterations': iterations, 'confidence': confidence, 'estimate': estimate, 'interval': interval,
              'planned_pairs': len(planned), 'observed_rows': {arm: sum(status_counts[arm].values()) for arm in (arm_a, arm_b)},
              'status_counts': {arm: {status: status_counts[arm][status] for status in sorted(STATUSES)} for arm in (arm_a, arm_b)},
              'missing_planned_rows': {arm: sum(arm not in by_pair.get(p, {}) for p in planned) for arm in (arm_a, arm_b)},
              'included_pairs': len(differences), 'independent_units': len(units), 'excluded_pairs': excluded,
              'pair_differences': differences, 'resampling_units': units, 'raw_rate_totals': rate_totals,
              'limitations': 'Percentile bootstrap uncertainty may be unreliable with few independent units; this estimate does not establish power or equivalence.'}
    if endpoint_bounds is not None:
        if not isinstance(endpoint_bounds, (list, tuple)) or len(endpoint_bounds) != 2:
            raise ValueError('Endpoint bounds require [low, high]')
        low = _number(endpoint_bounds[0], 'endpoint lower bound'); high = _number(endpoint_bounds[1], 'endpoint upper bound')
        if low >= high or any(not low <= value <= high for row in by_pair.values() for _, value in row.values() if value is not None):
            raise ValueError('Invalid endpoint bounds or observed value outside bounds')
        # Conservative range across all planned episode pairs. This is an
        # identification bound, NOT a confidence interval or family-weighted CI.
        lower, upper = [], []
        for pair_id in planned:
            pair = by_pair.get(pair_id, {})
            values = {arm: pair.get(arm, (None, None))[1] for arm in (arm_a, arm_b)}
            a, b = values[arm_a], values[arm_b]
            lower.append((a if a is not None else low) - (b if b is not None else high))
            upper.append((a if a is not None else high) - (b if b is not None else low))
        result['all_planned_episode_identification_bounds'] = [mean(lower), mean(upper)]
        # Distribution-free mean bound for independent bounded units. Unlike a
        # degenerate small-sample bootstrap, all-zero differences do not imply
        # zero population uncertainty. Missing-outcome interpretation remains
        # restricted by the complete-pair gate below.
        width = high-low
        radius = width * math.sqrt(2*math.log(2/(1-confidence))/len(units)) if units else None
        result['bounded_mean_interval'] = ([max(-width, estimate-radius), min(width, estimate+radius)] if units else None)
        result['bounded_interval_method'] = 'Hoeffding bound for independent bounded paired-unit means'
    return result


def paired_sample_plan(*, assumed_difference_sd, minimum_effect=None, practical_margin=None,
                       alpha=.05, power=.8, two_sided=True, unit='episode_pair', design_effect=1.):
    """Normal-approximation planning from explicit, external assumptions.

    A margin-based entry is precision/planning guidance, not a complete TOST or
    noninferiority design. Specify the intended hypothesis in preregistration.
    """
    if unit not in UNITS or type(two_sided) is not bool or (minimum_effect is None) == (practical_margin is None):
        raise ValueError('Declare unit, sidedness, and exactly one effect or practical margin')
    sd = _number(assumed_difference_sd, 'assumed paired-difference SD', 1e-12)
    target = _number(minimum_effect if minimum_effect is not None else practical_margin, 'planning target', 1e-12)
    alpha = _number(alpha, 'alpha', .0001, .5); power = _number(power, 'power', .5001, .9999)
    design_effect = _number(design_effect, 'design effect', 1., 1e6)
    z_alpha = NormalDist().inv_cdf(1-alpha/(2 if two_sided else 1))
    z_power = NormalDist().inv_cdf(power)
    count = max(2, math.ceil(((z_alpha+z_power)*sd/target)**2 * design_effect))
    return {'required_independent_units_approx': count, 'unit': unit, 'assumed_difference_sd': sd,
            'minimum_effect': minimum_effect, 'practical_margin': practical_margin, 'alpha': alpha,
            'target_power_assumption': power, 'two_sided': two_sided, 'design_effect': design_effect,
            'method': 'normal approximation for a paired difference', 'achieved_power': None,
            'limitations': 'Assumptions must come from external/pilot variance and a prespecified scientific target. Validate the final design with simulation; this does not confer power on a completed small sample or establish equivalence.'}


def task_benefit_bound(result, practical_margin):
    """A task-benefit interval alone never certifies costly preference."""
    margin = _number(practical_margin, 'practical task-benefit margin', 0.)
    interval = result.get('bounded_mean_interval')
    complete = not result.get('excluded_pairs') and not any(result.get('missing_planned_rows', {}).values())
    status = ('inconclusive_missing_outcomes' if not complete else 'no_bounded_endpoint_interval' if interval is None else
              'harm_bounded_away_from_zero' if interval[1] < 0 else 'benefit_below_margin' if interval[1] < margin else 'inconclusive')
    return {'status': status, 'margin': margin, 'interval': interval, 'method': result.get('bounded_interval_method'),
            'upper_bound_below_margin': bool(complete and interval is not None and interval[1] < margin),
            'costly_preference_established': False,
            'additional_evidence_required': ['voluntary preference with opportunity denominators', 'intact comprehension and work-tool use', 'independent manipulation validation', 'prespecified inclusion and termination handling']}
