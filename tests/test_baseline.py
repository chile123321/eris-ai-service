import math

import numpy as np

from eris_ai.baseline import make_folds, probability_metrics, stable_top_fraction_metrics
from eris_ai.preprocessing import build_preprocessor
from eris_ai.baseline_audit import bootstrap_oof


def test_top_ten_percent_uses_ceil_and_stable_employee_id_tie_break():
    y_true = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    probability = np.full(len(y_true), 0.2)
    employee_ids = np.array(["9", "1", "8", "2", "7", "6", "5", "4", "3", "11", "10"])
    metrics = stable_top_fraction_metrics(y_true, probability, employee_ids)
    assert metrics["top_10_count"] == math.ceil(0.10 * len(y_true)) == 2
    assert metrics["precision_at_10"] == 1.0
    assert metrics["recall_at_10"] == 2 / 3


def test_folds_assign_every_record_once_and_preserve_classes():
    target = np.array([0] * 90 + [1] * 10)
    folds, assignments = make_folds(target)
    assert len(folds) == 3
    assert set(assignments) == {0, 1, 2}
    for train_indices, validation_indices in folds:
        assert not set(train_indices) & set(validation_indices)
        assert set(target[train_indices]) == {0, 1}
        assert set(target[validation_indices]) == {0, 1}


def test_probability_metrics_reject_non_finite_or_out_of_range():
    target = np.array([0, 1])
    ids = np.array(["1", "2"])
    for probability in [np.array([np.nan, 0.5]), np.array([-0.1, 1.1])]:
        try:
            probability_metrics(target, probability, ids)
        except ValueError as error:
            assert "within [0, 1]" in str(error)
        else:
            raise AssertionError("Invalid probabilities were accepted")


def test_tree_preprocessor_disables_scaling_without_changing_default():
    unscaled = build_preprocessor(scale_numeric=False)
    scaled = build_preprocessor()
    assert "scaler" not in unscaled.transformers[1][1].named_steps
    assert "scaler" not in unscaled.transformers[2][1].named_steps
    assert "scaler" not in unscaled.transformers[3][1].named_steps
    assert "scaler" in scaled.transformers[1][1].named_steps


def test_reference_probability_metrics_are_computable():
    target = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    probability = np.full(len(target), 0.2)
    metrics = probability_metrics(target, probability, np.arange(len(target)).astype(str))
    assert metrics["roc_auc"] == 0.5
    assert metrics["ap"] == 0.2
    assert np.isfinite(list(metrics.values())).all()


def test_stratified_bootstrap_is_reproducible_and_preserves_prevalence():
    target = np.array([0] * 16 + [1] * 4)
    ids = np.arange(len(target)).astype(str)
    probabilities = {"constant": np.full(len(target), 0.2)}
    first = bootstrap_oof(target, probabilities, ids, repetitions=10)
    second = bootstrap_oof(target, probabilities, ids, repetitions=10)
    assert first == second
    assert first["constant"]["roc_auc"] == [0.5, 0.5]
    np.testing.assert_allclose(first["constant"]["ap"], [0.2, 0.2])
