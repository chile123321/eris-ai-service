import numpy as np
import pandas as pd
import pytest

from eris_ai.feature_diagnostic import ADDED_FEATURES, join_development_features, metric_comparison
from eris_ai.preprocessing import build_preprocessor


def test_join_preserves_original_values_and_order_and_rejects_missing_or_duplicate_ids():
    development = pd.DataFrame({"Employee_ID": ["b", "a"], "Hire_Date": ["2020-01-01", "2021-02-03"],
                                "Resigned": [0, 1]})
    additions = pd.DataFrame({"Employee_ID": ["a", "b"], **{
        col: [1, 2] for col in ADDED_FEATURES
    }})
    joined = join_development_features(development, additions)
    pd.testing.assert_frame_equal(joined[development.columns], development)
    assert joined["Performance_Score"].tolist() == [2, 1]
    with pytest.raises(ValueError, match="missing development"):
        join_development_features(development, additions.iloc[:1])
    with pytest.raises(ValueError, match="unique"):
        join_development_features(development, pd.concat([additions, additions]))


def test_diagnostic_imputation_and_encoding_fit_only_training_rows():
    preprocessor = build_preprocessor(scale_numeric=False, include_diagnostic_features=True)
    categorical = preprocessor.transformers[0][1]
    numeric = preprocessor.transformers[1][1]
    categorical.fit(pd.DataFrame({"Job_Title": ["Analyst", "Manager"]}))
    encoded = categorical.transform(pd.DataFrame({"Job_Title": ["Unseen"]}))
    np.testing.assert_array_equal(encoded, [[0, 0]])
    numeric.fit(pd.DataFrame({"Training_Hours": [2.0, 4.0]}))
    np.testing.assert_array_equal(numeric.transform(pd.DataFrame({"Training_Hours": [np.nan]})), [[3]])
    assert "Job_Title" in preprocessor.transformers[0][2]
    assert set(ADDED_FEATURES[1:]) <= set(preprocessor.transformers[1][2])


def test_identical_predictions_have_zero_paired_metric_difference():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.3, 0.2])
    result = metric_comparison(y, p, p.copy(), np.array(["a", "b", "c", "d"]))
    assert set(result["delta_extended_minus_baseline"].values()) == {0.0}
