"""Reusable, unfitted preprocessing for the ERIS employee dataset."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "Department",
    "Gender",
    "Age",
    "Hire_Date",
    "Years_At_Company",
    "Education_Level",
    "Monthly_Salary",
    "Work_Hours_Per_Week",
    "Overtime_Hours",
    "Sick_Days",
    "Promotions",
    "Employee_Satisfaction_Score",
]

EXCLUDED_COLUMNS = [
    "Job_Title",
    "Performance_Score",
    "Projects_Handled",
    "Remote_Work_Frequency",
    "Team_Size",
    "Training_Hours",
]

NUMERIC_COLUMNS = [
    "Age",
    "Years_At_Company",
    "Monthly_Salary",
    "Work_Hours_Per_Week",
    "Overtime_Hours",
    "Sick_Days",
    "Promotions",
    "Employee_Satisfaction_Score",
]

CATEGORICAL_COLUMNS = ["Department", "Gender"]
EDUCATION_ORDER = ["High School", "Bachelor", "Master", "PhD"]
DATE_EPOCH = pd.Timestamp("1970-01-01")


class HireDateToDaysTransformer(BaseEstimator, TransformerMixin):
    """Convert ISO calendar dates to days from 1970-01-01 without changing the date."""

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        values = np.asarray(X, dtype=object).reshape(-1)
        converted = np.empty(len(values), dtype=float)
        for index, value in enumerate(values):
            if value is None or pd.isna(value) or str(value).strip() == "":
                converted[index] = np.nan
                continue
            if isinstance(value, datetime):
                parsed = pd.Timestamp(value.date())
            elif isinstance(value, date):
                parsed = pd.Timestamp(value)
            else:
                try:
                    parsed = pd.to_datetime(str(value).strip(), format="%Y-%m-%d", errors="raise")
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Hire_Date must be an ISO date in YYYY-MM-DD format; got {value!r}"
                    ) from error
            converted[index] = float((parsed.normalize() - DATE_EPOCH).days)
        return converted.reshape(-1, 1)


class EducationOrdinalTransformer(BaseEstimator, TransformerMixin):
    """Apply the documented education order after fold-local imputation."""

    def fit(self, X, y=None):  # noqa: N803
        values = np.asarray(X, dtype=object).reshape(-1)
        unknown = sorted({str(value) for value in values if str(value) not in EDUCATION_ORDER})
        if unknown:
            raise ValueError(f"Unknown Education_Level values: {unknown}")
        return self

    def transform(self, X):  # noqa: N803
        values = np.asarray(X, dtype=object).reshape(-1)
        mapping = {value: index for index, value in enumerate(EDUCATION_ORDER)}
        unknown = sorted({str(value) for value in values if str(value) not in mapping})
        if unknown:
            raise ValueError(f"Unknown Education_Level values: {unknown}")
        return np.asarray([mapping[str(value)] for value in values], dtype=float).reshape(-1, 1)


def select_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return frame.loc[:, FEATURE_COLUMNS].copy()


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted transformer suitable for use inside cross-validation."""

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    date_pipeline = Pipeline(
        steps=[
            ("to_days_since_1970_01_01", HireDateToDaysTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    education_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", EducationOrdinalTransformer()),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("hire_date", date_pipeline, ["Hire_Date"]),
            ("education", education_pipeline, ["Education_Level"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
