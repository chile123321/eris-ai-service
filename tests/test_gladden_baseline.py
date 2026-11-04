import numpy as np
import pandas as pd

from eris_ai.gladden_baseline import NUMERIC, preprocessor, weekly_priorities


def weekly_frame():
    return pd.DataFrame({
        "sub_ID": [str(i) for i in range(10)] * 2,
        "snapshot_date": [pd.Timestamp("2021-12-05")] * 10
        + [pd.Timestamp("2021-12-12")] * 10,
        "target": [1] + [0] * 9 + [1] + [0] * 9,
    })


def test_dummy_random_expectation_and_distinct_case_detection():
    frame = weekly_frame()
    result, selected, chances = weekly_priorities(frame, np.full(20, 0.2), random_expectation=True)
    assert result["alert_count"] == 2 and not selected.any()
    assert result["unique_resigning_people"] == 1
    np.testing.assert_allclose(result["unique_resigning_people_alerted"], 1 - 0.9 ** 2)
    np.testing.assert_allclose(chances, 0.1)
    assert result["macro_weekly"] == {"precision_at_10": 0.1, "recall_at_10": 0.1, "lift_at_10": 1.0}


def test_weekly_ranking_deduplicates_cases_and_uses_stable_ties():
    frame = weekly_frame().iloc[::-1].reset_index(drop=True)
    result, selected, _ = weekly_priorities(frame, np.full(20, 0.2))
    assert result["alert_count"] == result["true_positive_alerts"] == 2
    assert result["unique_resigning_people_alerted"] == 1
    assert set(frame.loc[selected, "sub_ID"]) == {"0"}


def test_zero_positive_week_has_undefined_recall_and_lift():
    frame = weekly_frame().iloc[:10].copy()
    frame["target"] = 0
    result, _, _ = weekly_priorities(frame, np.full(10, 0.1), random_expectation=True)
    assert result["macro_weekly"] == {
        "precision_at_10": 0.0, "recall_at_10": None, "lift_at_10": None,
    }
    assert result["unique_case_coverage"] is None


def test_preprocessing_learns_only_training_categories_and_scaling():
    train = pd.DataFrame({"role": ["Laborer", "Team Leader"], "shift": ["1", "1"],
                          "team": ["1", "2"], **{c: [0.0, 2.0] for c in NUMERIC}})
    valid = pd.DataFrame({"role": ["Unseen"], "shift": ["9"], "team": ["99"],
                          **{c: [100.0] for c in NUMERIC}})
    transformer = preprocessor().fit(train)
    scaler = transformer.named_transformers_["numeric"].named_steps["scale"]
    np.testing.assert_array_equal(scaler.mean_, np.ones(5))
    transformed = transformer.transform(valid)
    np.testing.assert_array_equal(transformed[0, :5], np.zeros(5))
    np.testing.assert_array_equal(scaler.mean_, np.ones(5))
