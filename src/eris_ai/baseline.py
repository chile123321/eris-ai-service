"""Run development-only cross-validation for the two ERIS baseline candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from eris_ai.data_preparation import ID_COLUMN, RANDOM_STATE, TARGET_COLUMN, sha256_file
from eris_ai.preprocessing import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    build_preprocessor,
    select_model_features,
)


CV_CONFIG = {"n_splits": 3, "shuffle": True, "random_state": RANDOM_STATE}
MODEL_BUILDERS = {
    "gradient_boosting": lambda: GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=10,
        subsample=1.0,
        max_features=None,
        n_iter_no_change=None,
        random_state=RANDOM_STATE,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=10,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=4,
    ),
}


def validate_development(frame: pd.DataFrame) -> None:
    expected_columns = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"Development columns/order must be exactly {expected_columns}; got {list(frame.columns)}"
        )
    if frame[ID_COLUMN].isna().any() or not frame[ID_COLUMN].is_unique:
        raise ValueError("Development Employee_ID must be present and unique")
    target_values = set(frame[TARGET_COLUMN].dropna().unique().tolist())
    if target_values != {0, 1} or frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"Resigned must contain only 0 and 1; got {sorted(target_values)}")
    if ID_COLUMN in FEATURE_COLUMNS or TARGET_COLUMN in FEATURE_COLUMNS:
        raise AssertionError("Employee_ID and Resigned cannot be model features")
    if set(EXCLUDED_COLUMNS) & set(FEATURE_COLUMNS):
        raise AssertionError("Excluded columns cannot be model features")
    if any(column.endswith("_Scaled") for column in frame.columns):
        raise ValueError("Pre-scaled columns are not allowed")


def stable_top_fraction_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    employee_ids: np.ndarray,
    fraction: float = 0.10,
) -> dict[str, float | int]:
    count = math.ceil(fraction * len(y_true))
    ranking = pd.DataFrame(
        {
            "probability": probability,
            "employee_id": employee_ids.astype(str),
            "y_true": y_true,
        }
    ).sort_values(
        ["probability", "employee_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected = ranking.iloc[:count]
    positives = int(np.sum(y_true))
    selected_positives = int(selected["y_true"].sum())
    precision = selected_positives / count
    recall = selected_positives / positives
    prevalence = positives / len(y_true)
    return {
        "top_10_count": count,
        "precision_at_10": float(precision),
        "recall_at_10": float(recall),
        "lift_at_10": float(precision / prevalence),
    }


def probability_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    employee_ids: np.ndarray,
) -> dict[str, float | int]:
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    predicted = (probability >= 0.5).astype(int)
    metrics: dict[str, float | int] = {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "ap": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "prevalence": float(np.mean(y_true)),
    }
    metrics.update(stable_top_fraction_metrics(y_true, probability, employee_ids))
    return metrics


def class_one_probability(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(pipeline.named_steps["model"].classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(f"Estimator classes must contain class 1 exactly once; got {classes.tolist()}")
    return pipeline.predict_proba(features)[:, int(matches[0])]


def make_folds(target: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    splitter = StratifiedKFold(**CV_CONFIG)
    folds = list(splitter.split(np.zeros(len(target)), target))
    assignments = np.full(len(target), -1, dtype=int)
    for fold_number, (train_indices, validation_indices) in enumerate(folds):
        if np.intersect1d(train_indices, validation_indices).size:
            raise AssertionError("Fold train and validation indices overlap")
        if np.any(assignments[validation_indices] != -1):
            raise AssertionError("A development record was assigned to multiple validation folds")
        assignments[validation_indices] = fold_number
    if np.any(assignments < 0):
        raise AssertionError("Every development record must have one validation fold")
    return folds, assignments


def fold_assignment_sha256(employee_ids: pd.Series, assignments: np.ndarray) -> str:
    canonical = pd.DataFrame(
        {ID_COLUMN: employee_ids.astype(str), "fold": assignments}
    ).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def summarize_folds(folds: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    keys = [
        key
        for key, value in folds[0].items()
        if key not in {"fold", "train_rows", "validation_rows"}
        and isinstance(value, (int, float))
    ]
    means = {key: float(np.mean([fold[key] for fold in folds])) for key in keys}
    stds = {key: float(np.std([fold[key] for fold in folds], ddof=1)) for key in keys}
    return means, stds


def validate_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    forbidden = [ID_COLUMN, TARGET_COLUMN, *EXCLUDED_COLUMNS]
    for column in forbidden:
        if any(column in name for name in names):
            raise AssertionError(f"Forbidden column reached transformed features: {column}")
    return names


def run_baseline(development_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("Output directory already exists; use a new run directory")
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    frame = pd.read_csv(development_path, dtype={ID_COLUMN: "string"})
    validate_development(frame)
    features = select_model_features(frame)
    target = frame[TARGET_COLUMN].to_numpy(dtype=int)
    employee_ids = frame[ID_COLUMN].astype(str)
    folds, fold_assignments = make_folds(target)
    fold_hash = fold_assignment_sha256(employee_ids, fold_assignments)

    oof = pd.DataFrame(
        {
            ID_COLUMN: employee_ids,
            "fold": fold_assignments,
            "y_true": target,
        }
    )
    reference_probability = np.full(len(frame), np.nan, dtype=float)
    reference_folds = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds):
        prevalence = float(np.mean(target[train_indices]))
        probabilities = np.full(len(validation_indices), prevalence, dtype=float)
        reference_probability[validation_indices] = probabilities
        fold_metrics = probability_metrics(
            target[validation_indices],
            probabilities,
            employee_ids.iloc[validation_indices].to_numpy(),
        )
        reference_folds.append(
            {
                "fold": fold_number,
                "train_rows": len(train_indices),
                "validation_rows": len(validation_indices),
                "training_prevalence": prevalence,
                **fold_metrics,
            }
        )
    oof["reference_probability"] = reference_probability

    model_results = {}
    transformed_feature_names: list[str] | None = None
    for model_name, builder in MODEL_BUILDERS.items():
        oof_probability = np.full(len(frame), np.nan, dtype=float)
        fold_results = []
        for fold_number, (train_indices, validation_indices) in enumerate(folds):
            print(f"[{model_name}] fold {fold_number + 1}/{len(folds)}: fitting", flush=True)
            pipeline = Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(scale_numeric=False)),
                    ("model", builder()),
                ]
            )
            fit_started = time.perf_counter()
            pipeline.fit(features.iloc[train_indices], target[train_indices])
            fit_seconds = time.perf_counter() - fit_started
            if transformed_feature_names is None:
                transformed_feature_names = validate_transformed_feature_names(pipeline)

            prediction_started = time.perf_counter()
            validation_probability = class_one_probability(
                pipeline, features.iloc[validation_indices]
            )
            prediction_seconds = time.perf_counter() - prediction_started
            train_probability = class_one_probability(pipeline, features.iloc[train_indices])
            oof_probability[validation_indices] = validation_probability
            validation_metrics = probability_metrics(
                target[validation_indices],
                validation_probability,
                employee_ids.iloc[validation_indices].to_numpy(),
            )
            fold_results.append(
                {
                    "fold": fold_number,
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "train_roc_auc": float(
                        roc_auc_score(target[train_indices], train_probability)
                    ),
                    "train_ap": float(
                        average_precision_score(target[train_indices], train_probability)
                    ),
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    "fit_time_seconds": fit_seconds,
                    "prediction_time_seconds": prediction_seconds,
                }
            )
            print(
                f"[{model_name}] fold {fold_number + 1}/{len(folds)}: "
                f"AP={validation_metrics['ap']:.6f} ROC-AUC={validation_metrics['roc_auc']:.6f}",
                flush=True,
            )
        if np.isnan(oof_probability).any():
            raise AssertionError(f"Missing OOF predictions for {model_name}")
        oof[f"{model_name}_probability"] = oof_probability
        means, stds = summarize_folds(fold_results)
        model_results[model_name] = {
            "folds": fold_results,
            "fold_mean": means,
            "fold_std": stds,
            "pooled_oof": probability_metrics(
                target,
                oof_probability,
                employee_ids.to_numpy(),
            ),
            "total_fit_time_seconds": float(
                sum(fold["fit_time_seconds"] for fold in fold_results)
            ),
        }

    if oof[[f"{name}_probability" for name in MODEL_BUILDERS]].isna().any().any():
        raise AssertionError("Every record must have one OOF probability per model")
    reference_mean, reference_std = summarize_folds(reference_folds)
    metrics = {
        "scope": "development_3_fold_cv_oof",
        "models": model_results,
        "reference": {
            "description": "constant training-fold Resigned=1 prevalence",
            "folds": reference_folds,
            "fold_mean": reference_mean,
            "fold_std": reference_std,
            "pooled_oof": probability_metrics(
                target,
                reference_probability,
                employee_ids.to_numpy(),
            ),
        },
    }
    ordered_oof = oof[
        [
            ID_COLUMN,
            "fold",
            "y_true",
            "gradient_boosting_probability",
            "random_forest_probability",
            "reference_probability",
        ]
    ]
    completed_at = datetime.now(timezone.utc)
    model_parameters = {name: builder().get_params(deep=False) for name, builder in MODEL_BUILDERS.items()}
    metadata = {
        "scope": "development_only_no_final_test_access",
        "development": {
            "filename": development_path.name,
            "sha256": sha256_file(development_path),
            "rows": len(frame),
        },
        "feature_order": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "preprocessing": {
            "builder": "eris_ai.preprocessing.build_preprocessor",
            "fit_inside_each_training_fold": True,
            "scale_numeric": False,
            "dense_one_hot": True,
            "date_epoch": "1970-01-01",
            "transformed_feature_names": transformed_feature_names,
        },
        "cross_validation": CV_CONFIG,
        "fold_assignment_sha256": fold_hash,
        "fold_assignment_saved_in": "oof_predictions.csv",
        "models": model_parameters,
        "random_seed": RANDOM_STATE,
        "dependency_versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
        },
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - started_clock,
        "command_python": sys.executable,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ordered_oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"metrics": metrics, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_baseline(arguments.development, arguments.output_dir)
    summary = {
        name: {
            "oof_roc_auc": values["pooled_oof"]["roc_auc"],
            "oof_ap": values["pooled_oof"]["ap"],
            "total_fit_time_seconds": values["total_fit_time_seconds"],
        }
        for name, values in result["metrics"]["models"].items()
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
