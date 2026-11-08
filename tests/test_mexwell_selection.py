import copy

import numpy as np

from eris_ai.baseline import MODEL_BUILDERS, stable_top_fraction_metrics
from eris_ai.mexwell_selection import (
    CRITERIA, candidate_configs, paired_intervals, passes, screening_checks,
)


def test_search_changes_only_depth_and_leaf_and_is_bounded():
    configs = candidate_configs()
    assert len(configs) == 8
    for algorithm, builder in MODEL_BUILDERS.items():
        selected = [c for c in configs.values() if c['algorithm'] == algorithm]
        assert len(selected) == 4
        original = builder().get_params(deep=False)
        for config in selected:
            params = config['parameters']
            assert params['max_depth'] < original['max_depth']
            assert params['min_samples_leaf'] > original['min_samples_leaf']
            assert all(params[k] == v for k, v in original.items()
                       if k not in {'max_depth', 'min_samples_leaf'})


def test_all_screening_gates_required_including_joint_fold_evidence():
    reference = {'ap': 0.10, 'brier': 0.09, 'log_loss': 0.33}
    pooled = {'ap': 0.12, 'lift_at_10': 1.2, 'brier': 0.09, 'log_loss': 0.33}
    folds = [{'validation': {'ap': 0.12, 'lift_at_10': 1.1}} for _ in range(3)]
    refs = [reference] * 3
    assert passes(screening_checks(pooled, reference, folds, refs, CRITERIA))
    for key, failing in [('ap', 0.109), ('lift_at_10', 1.199),
                         ('brier', 0.09001), ('log_loss', 0.33001)]:
        changed = {**pooled, key: failing}
        assert not passes(screening_checks(changed, reference, folds, refs, CRITERIA))
    changed = copy.deepcopy(folds)
    changed[0]['validation']['ap'] = reference['ap']
    changed[1]['validation']['lift_at_10'] = 1.0
    assert not passes(screening_checks(pooled, reference, changed, refs, CRITERIA))


def test_paired_bootstrap_identical_predictions_zero_difference_and_stable_ties():
    y = np.tile([0, 0, 0, 0, 1], 6)
    folds = np.repeat([1, 2, 3], 10)
    ids = np.arange(len(y)).astype(str)
    p = np.full(len(y), 0.2)
    first = paired_intervals(y, p, p, ids, folds, repetitions=30)
    assert first['delta_ap'] == [0.0, 0.0]
    assert first == paired_intervals(y, p, p, ids, folds, repetitions=30)
    rng = np.random.default_rng(42)
    strata = [np.flatnonzero((folds == f) & (y == label))
              for f in (1, 2, 3) for label in (0, 1)]
    expected_lifts = []
    for _ in range(30):
        indices = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        expected_lifts.append(stable_top_fraction_metrics(
            y[indices], p[indices], ids[indices])['lift_at_10'])
    np.testing.assert_allclose(first['lift_at_10'], np.percentile(expected_lifts, [2.5, 97.5]))
