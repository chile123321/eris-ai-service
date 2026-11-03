"""Prepare the ERIS employee dataset without fitting preprocessing or training a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from eris_ai.preprocessing import EDUCATION_ORDER, EXCLUDED_COLUMNS, FEATURE_COLUMNS


TARGET_COLUMN = "Resigned"
ID_COLUMN = "Employee_ID"
RANDOM_STATE = 42
FEATURE_SET_VERSION = "employee-performance-v1"
EXPECTED_SOURCE_SHA256 = "2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb"
SOURCE_HIRE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
EXPECTED_SOURCE_COLUMNS = [
    ID_COLUMN,
    "Department",
    "Gender",
    "Age",
    "Job_Title",
    "Hire_Date",
    "Years_At_Company",
    "Education_Level",
    "Performance_Score",
    "Monthly_Salary",
    "Work_Hours_Per_Week",
    "Projects_Handled",
    "Overtime_Hours",
    "Sick_Days",
    "Remote_Work_Frequency",
    "Team_Size",
    "Training_Hours",
    "Promotions",
    "Employee_Satisfaction_Score",
    TARGET_COLUMN,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_target_value(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized == "False":
            return 0
        if normalized == "True":
            return 1
    raise ValueError(f"Unrecognized Resigned value: {value!r}")


def normalize_hire_dates(values: pd.Series) -> tuple[pd.Series, dict]:
    stripped = values.astype("string").str.strip()
    missing = stripped.isna() | stripped.eq("")
    parsed = pd.to_datetime(
        stripped.mask(missing),
        format=SOURCE_HIRE_DATE_FORMAT,
        errors="coerce",
    )
    invalid = ~missing & parsed.isna()
    if invalid.any():
        raise ValueError(
            f"Hire_Date contains {int(invalid.sum())} values outside the required "
            f"format {SOURCE_HIRE_DATE_FORMAT}"
        )
    if missing.any():
        raise ValueError(f"Hire_Date contains {int(missing.sum())} missing values")
    time_components_removed = int(
        (
            (parsed.dt.hour != 0)
            | (parsed.dt.minute != 0)
            | (parsed.dt.second != 0)
            | (parsed.dt.microsecond != 0)
        ).sum()
    )
    normalized = parsed.dt.strftime("%Y-%m-%d").astype("string")
    stats = {
        "source_format": "YYYY-MM-DD HH:MM:SS.ffffff",
        "calendar_dates_preserved": len(normalized),
        "values_converted_to_iso": len(normalized),
        "time_components_removed": time_components_removed,
        "unresolved": 0,
        "minimum_date": normalized.min(),
        "maximum_date": normalized.max(),
        "unique_dates": int(normalized.nunique()),
    }
    return normalized, stats


def load_source(source: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(source)
    if list(frame.columns) != EXPECTED_SOURCE_COLUMNS:
        raise ValueError(
            "Unexpected source columns. "
            f"Expected {EXPECTED_SOURCE_COLUMNS}, got {list(frame.columns)}"
        )
    frame["Hire_Date"], date_stats = normalize_hire_dates(frame["Hire_Date"])
    return frame, date_stats


def normalize_and_validate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    output = frame.copy()
    output[ID_COLUMN] = output[ID_COLUMN].astype("string").str.strip()
    if output[ID_COLUMN].isna().any() or (output[ID_COLUMN] == "").any():
        raise ValueError("Employee_ID contains missing values")

    string_columns = ["Department", "Gender", "Education_Level"]
    trimmed_changes = 0
    for column in string_columns:
        before = output[column].astype("string")
        after = before.str.strip()
        trimmed_changes += int((before.fillna("") != after.fillna("")).sum())
        output[column] = after

    education_values = set(output["Education_Level"].dropna())
    unknown_education = sorted(education_values - set(EDUCATION_ORDER))
    if unknown_education:
        raise ValueError(f"Unknown Education_Level values: {unknown_education}")

    numeric_columns = [
        "Age",
        "Years_At_Company",
        "Monthly_Salary",
        "Work_Hours_Per_Week",
        "Overtime_Hours",
        "Sick_Days",
        "Promotions",
        "Employee_Satisfaction_Score",
    ]
    ranges = {}
    for column in numeric_columns:
        converted = pd.to_numeric(output[column], errors="raise")
        if not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains non-finite values")
        output[column] = converted
        ranges[column] = {"min": float(converted.min()), "max": float(converted.max())}

    output[TARGET_COLUMN] = output[TARGET_COLUMN].map(map_target_value)
    missing_counts = {column: int(count) for column, count in output.isna().sum().items()}
    quality = {
        "missing_by_column": missing_counts,
        "duplicate_employee_ids": int(output[ID_COLUMN].duplicated().sum()),
        "duplicate_rows_including_id": int(output.duplicated().sum()),
        "duplicate_rows_excluding_id": int(output.drop(columns=[ID_COLUMN]).duplicated().sum()),
        "trimmed_category_values": trimmed_changes,
        "department_vocabulary": sorted(output["Department"].dropna().unique().tolist()),
        "gender_vocabulary": sorted(output["Gender"].dropna().unique().tolist()),
        "education_vocabulary": sorted(education_values),
        "numeric_ranges": ranges,
        "age_minus_years_at_company_below_18": int(
            ((output["Age"] - output["Years_At_Company"]) < 18).sum()
        ),
        "years_zero_with_promotions": int(
            ((output["Years_At_Company"] == 0) & (output["Promotions"] >= 1)).sum()
        ),
    }
    return output, quality


def make_split_assignments(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    target = frame[TARGET_COLUMN].to_numpy()
    ids = frame[ID_COLUMN].astype(str)
    assignments = pd.Series(index=frame.index, dtype="string")
    if ids.is_unique:
        development_indices, final_indices = train_test_split(
            frame.index.to_numpy(),
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=target,
        )
        strategy = "stratified_train_test_split"
    else:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        development_positions, final_positions = next(
            splitter.split(frame, target, groups=ids)
        )
        development_indices = frame.index.to_numpy()[development_positions]
        final_indices = frame.index.to_numpy()[final_positions]
        strategy = "stratified_group_5_fold_first_holdout"
    assignments.loc[development_indices] = "development"
    assignments.loc[final_indices] = "final_test"
    if assignments.isna().any():
        raise AssertionError("Every row must receive exactly one split assignment")
    development_ids = set(ids.loc[assignments == "development"])
    final_ids = set(ids.loc[assignments == "final_test"])
    if development_ids & final_ids:
        raise AssertionError("Employee_ID overlap detected between splits")
    return assignments, strategy


def distribution(series: pd.Series) -> dict[str, int]:
    return {str(int(key)): int(value) for key, value in series.value_counts().sort_index().items()}


def prepare_dataset(
    source: Path,
    output_dir: Path,
    expected_sha256: str = EXPECTED_SOURCE_SHA256,
) -> dict:
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Source SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    raw, date_stats = load_source(source)
    validated, quality = normalize_and_validate(raw)

    prepared = validated.loc[:, [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    assignments, split_strategy = make_split_assignments(prepared)
    development = prepared.loc[assignments == "development"].copy()
    final_test = prepared.loc[assignments == "final_test"].copy()
    assignment_table = pd.DataFrame(
        {ID_COLUMN: prepared[ID_COLUMN], "split": assignments},
        index=prepared.index,
    )

    if len(development) + len(final_test) != len(prepared):
        raise AssertionError("Split row counts do not reconcile")
    overlap = set(development[ID_COLUMN]) & set(final_test[ID_COLUMN])
    if overlap:
        raise AssertionError("Employee_ID overlap detected")
    if ID_COLUMN in FEATURE_COLUMNS or TARGET_COLUMN in FEATURE_COLUMNS:
        raise AssertionError("ID and target must not be model features")
    if set(EXCLUDED_COLUMNS) & set(FEATURE_COLUMNS):
        raise AssertionError("Excluded columns must not be model features")

    metadata = {
        "status": "DATA_PREPARATION_READY",
        "data_version": actual_sha256,
        "source": {
            "filename": source.name,
            "sha256": actual_sha256,
            "expected_sha256": expected_sha256,
            "hash_verified": True,
            "rows": len(raw),
            "columns": len(raw.columns),
        },
        "feature_set_version": FEATURE_SET_VERSION,
        "feature_order": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "target_mapping": {"False": 0, "True": 1},
        "target_distribution": distribution(prepared[TARGET_COLUMN]),
        "date_handling": {
            "prepared_format": "YYYY-MM-DD",
            "preprocessing_representation": "days since 1970-01-01",
            **date_stats,
        },
        "quality": quality,
        "split": {
            "strategy": split_strategy,
            "random_state": RANDOM_STATE,
            "development_rows": len(development),
            "final_test_rows": len(final_test),
            "development_target_distribution": distribution(development[TARGET_COLUMN]),
            "final_test_target_distribution": distribution(final_test[TARGET_COLUMN]),
            "employee_id_overlap": 0,
        },
        "prepared_columns": [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN],
        "prepared_rows": len(prepared),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    development.to_csv(output_dir / "development.csv", index=False, na_rep="")
    final_test.to_csv(output_dir / "final_test.csv", index=False, na_rep="")
    assignment_table.to_csv(output_dir / "split_assignments.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    arguments = parser.parse_args()
    metadata = prepare_dataset(
        arguments.source_csv,
        arguments.output_dir,
    )
    summary = {
        "status": metadata["status"],
        "rows": metadata["prepared_rows"],
        "development_rows": metadata["split"]["development_rows"],
        "final_test_rows": metadata["split"]["final_test_rows"],
        "calendar_dates_preserved": metadata["date_handling"]["calendar_dates_preserved"],
        "unresolved_hire_dates": metadata["date_handling"]["unresolved"],
        "age_tenure_anomalies": metadata["quality"][
            "age_minus_years_at_company_below_18"
        ],
        "trimmed_category_values": metadata["quality"]["trimmed_category_values"],
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
