"""Prepare the ERIS employee dataset without fitting preprocessing or training a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from eris_ai.preprocessing import EDUCATION_ORDER, EXCLUDED_COLUMNS, FEATURE_COLUMNS


TARGET_COLUMN = "Resigned"
ID_COLUMN = "Employee_ID"
RANDOM_STATE = 42
FEATURE_SET_VERSION = "employee-performance-v1"
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


def normalize_hire_date(value) -> tuple[str | None, str]:
    if value is None or (not isinstance(value, (date, datetime, time)) and pd.isna(value)):
        return None, "missing"
    if isinstance(value, datetime):
        return value.date().isoformat(), "calendar_date"
    if isinstance(value, date):
        return value.isoformat(), "calendar_date"
    if isinstance(value, time):
        return None, "time_only"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None, "missing"
        try:
            parsed = datetime.strptime(stripped, "%Y-%m-%d").date()
        except ValueError:
            return None, "unrecognized"
        return parsed.isoformat(), "calendar_date"
    return None, "unrecognized"


def read_hire_dates(source: Path) -> tuple[pd.Series, dict]:
    workbook = load_workbook(source, read_only=True, data_only=False)
    worksheet = workbook.active
    header_cells = next(worksheet.iter_rows(min_row=1, max_row=1))
    headers = [cell.value for cell in header_cells]
    if "Hire_Date" not in headers:
        raise ValueError("Source workbook does not contain Hire_Date")
    column_index = headers.index("Hire_Date") + 1
    normalized: list[str | None] = []
    statuses: Counter[str] = Counter()
    number_formats: Counter[str] = Counter()
    for row in worksheet.iter_rows(
        min_row=2,
        min_col=column_index,
        max_col=column_index,
    ):
        cell = row[0]
        value, status = normalize_hire_date(cell.value)
        normalized.append(value)
        statuses[status] += 1
        number_formats[cell.number_format] += 1

    xml_stats = inspect_hire_date_serials(source, column_index)
    stats = {
        "calendar_dates_preserved": statuses["calendar_date"],
        "values_converted_to_iso": statuses["calendar_date"],
        "unresolved_time_only": statuses["time_only"],
        "unresolved_missing": statuses["missing"],
        "unresolved_unrecognized": statuses["unrecognized"],
        "excel_number_formats": dict(sorted(number_formats.items())),
        "raw_xml_serials": xml_stats,
    }
    return pd.Series(normalized, dtype="string"), stats


def inspect_hire_date_serials(source: Path, column_index: int) -> dict:
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    column_letter = get_column_letter(column_index)
    with ZipFile(source) as archive:
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    values: list[float] = []
    for cell in root.findall(".//main:c", namespace):
        reference = cell.attrib.get("r", "")
        if not reference.startswith(column_letter):
            continue
        row_text = reference[len(column_letter) :]
        if not row_text.isdigit() or int(row_text) < 2:
            continue
        value_node = cell.find("main:v", namespace)
        if value_node is not None:
            values.append(float(value_node.text))
    return {
        "numeric_cells": len(values),
        "date_serial_ge_1": sum(value >= 1 for value in values),
        "fraction_between_0_and_1": sum(0 <= value < 1 for value in values),
        "negative": sum(value < 0 for value in values),
    }


def load_source(source: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_excel(source, engine="openpyxl")
    frame = frame.rename(columns={frame.columns[0]: ID_COLUMN})
    if list(frame.columns) != EXPECTED_SOURCE_COLUMNS:
        raise ValueError(
            "Unexpected source columns. "
            f"Expected {EXPECTED_SOURCE_COLUMNS}, got {list(frame.columns)}"
        )
    hire_dates, date_stats = read_hire_dates(source)
    if len(hire_dates) != len(frame):
        raise ValueError("Hire_Date cell count does not match source row count")
    frame["Hire_Date"] = hire_dates
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


def validate_clean_reference(
    prepared: pd.DataFrame,
    clean_reference: Path,
) -> dict:
    reference = pd.read_csv(
        clean_reference,
        usecols=[ID_COLUMN, "Hire_Date"],
        dtype={ID_COLUMN: "string", "Hire_Date": "string"},
    )
    reference[ID_COLUMN] = reference[ID_COLUMN].str.strip()
    if reference[ID_COLUMN].duplicated().any():
        raise ValueError("Clean reference contains duplicate Employee_ID values")
    joined = prepared[[ID_COLUMN, "Hire_Date"]].merge(
        reference,
        on=ID_COLUMN,
        how="outer",
        suffixes=("_raw", "_reference"),
        validate="many_to_one",
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        raise ValueError("Employee_ID sets differ between raw workbook and clean reference")
    grounded = joined["Hire_Date_raw"].notna()
    reference_dates = pd.to_datetime(
        joined.loc[grounded, "Hire_Date_reference"],
        format="%Y-%m-%d",
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    matches = joined.loc[grounded, "Hire_Date_raw"].reset_index(drop=True).eq(
        reference_dates.reset_index(drop=True)
    )
    if not matches.all():
        raise ValueError("Grounded Hire_Date values disagree with clean reference")
    return {
        "employee_ids_matched": len(joined),
        "grounded_dates_matched": int(matches.sum()),
        "reconstructed_reference_dates_not_used": int((~grounded).sum()),
    }


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


def prepare_dataset(source: Path, clean_reference: Path, output_dir: Path) -> dict:
    raw, date_stats = load_source(source)
    validated, quality = normalize_and_validate(raw)
    reference_stats = validate_clean_reference(validated, clean_reference)

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
        "status": "DATE_REVIEW_REQUIRED" if date_stats["unresolved_time_only"] else "READY",
        "data_version": sha256_file(source),
        "source": {
            "filename": source.name,
            "sha256": sha256_file(source),
            "rows": len(raw),
            "columns": len(raw.columns),
        },
        "clean_reference": {
            "filename": clean_reference.name,
            "sha256": sha256_file(clean_reference),
            **reference_stats,
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
    parser.add_argument("--source-xlsx", type=Path, required=True)
    parser.add_argument("--clean-reference-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    arguments = parser.parse_args()
    metadata = prepare_dataset(
        arguments.source_xlsx,
        arguments.clean_reference_csv,
        arguments.output_dir,
    )
    summary = {
        "status": metadata["status"],
        "rows": metadata["prepared_rows"],
        "development_rows": metadata["split"]["development_rows"],
        "final_test_rows": metadata["split"]["final_test_rows"],
        "calendar_dates_preserved": metadata["date_handling"]["calendar_dates_preserved"],
        "unresolved_hire_dates": metadata["date_handling"]["unresolved_time_only"],
        "age_tenure_anomalies": metadata["quality"][
            "age_minus_years_at_company_below_18"
        ],
        "trimmed_category_values": metadata["quality"]["trimmed_category_values"],
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
