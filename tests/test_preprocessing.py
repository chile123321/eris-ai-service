import numpy as np
import pandas as pd
import pytest

from eris_ai.data_preparation import make_split_assignments, map_target_value
from eris_ai.preprocessing import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    HireDateToDaysTransformer,
    build_preprocessor,
    select_model_features,
)


def test_hire_date_conversion_uses_fixed_epoch_and_preserves_missing():
    transformed = HireDateToDaysTransformer().transform(
        pd.DataFrame({"Hire_Date": ["1970-01-01", "1970-01-02", None]})
    )
    assert transformed[0, 0] == 0
    assert transformed[1, 0] == 1
    assert np.isnan(transformed[2, 0])


def test_hire_date_rejects_ambiguous_format():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        HireDateToDaysTransformer().transform(pd.DataFrame({"Hire_Date": ["01/02/2020"]}))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(False, 0), (True, 1), ("False", 0), (" True ", 1)],
)
def test_target_mapping(raw, expected):
    assert map_target_value(raw) == expected


def test_target_mapping_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unrecognized Resigned"):
        map_target_value("yes")


def test_feature_selection_has_exact_order_and_excludes_forbidden_columns():
    all_columns = ["Employee_ID", *FEATURE_COLUMNS, *EXCLUDED_COLUMNS, "Resigned"]
    frame = pd.DataFrame([{column: 0 for column in all_columns}])
    selected = select_model_features(frame)
    assert list(selected.columns) == FEATURE_COLUMNS
    assert "Employee_ID" not in selected
    assert "Resigned" not in selected
    assert not set(EXCLUDED_COLUMNS) & set(selected.columns)


def test_preprocessor_is_returned_unfitted():
    preprocessor = build_preprocessor()
    assert not hasattr(preprocessor, "transformers_")


def test_preprocessor_fits_only_when_explicitly_used_on_training_data():
    training = pd.DataFrame(
        {
            "Department": ["IT", "HR", "IT", "HR"],
            "Gender": ["Female", "Male", "Female", "Male"],
            "Age": [25, 30, 35, 40],
            "Hire_Date": ["2020-01-01", "2021-01-01", None, "2022-01-01"],
            "Years_At_Company": [1, 2, 3, 4],
            "Education_Level": ["High School", "Bachelor", "Master", "PhD"],
            "Monthly_Salary": [4000, 5000, 6000, 7000],
            "Work_Hours_Per_Week": [40, 41, 42, 43],
            "Overtime_Hours": [0, 1, 2, 3],
            "Sick_Days": [1, 2, 3, 4],
            "Promotions": [0, 0, 1, 1],
            "Employee_Satisfaction_Score": [2.0, 3.0, 4.0, 5.0],
        }
    )
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(select_model_features(training))
    assert transformed.shape == (4, 14)
    assert np.isfinite(transformed).all()


def test_unique_employee_split_is_stratified_and_disjoint():
    frame = pd.DataFrame(
        {
            "Employee_ID": [str(index) for index in range(100)],
            "Resigned": [0] * 80 + [1] * 20,
        }
    )
    assignments, strategy = make_split_assignments(frame)
    assert strategy == "stratified_train_test_split"
    assert assignments.value_counts().to_dict() == {"development": 80, "final_test": 20}
    development_ids = set(frame.loc[assignments == "development", "Employee_ID"])
    final_ids = set(frame.loc[assignments == "final_test", "Employee_ID"])
    assert not development_ids & final_ids
    assert frame.loc[assignments == "final_test", "Resigned"].sum() == 4
