"""Paired development-only feature diagnostic; reuse frozen baseline folds and models."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from eris_ai.baseline import (
    CV_CONFIG,
    MODEL_BUILDERS,
    class_one_probability,
    fold_assignment_sha256,
    probability_metrics,
    validate_development,
)
from eris_ai.data_preparation import ID_COLUMN, RANDOM_STATE, TARGET_COLUMN, sha256_file
from eris_ai.preprocessing import EXCLUDED_COLUMNS, FEATURE_COLUMNS, build_preprocessor

SOURCE_SHA256 = "2b1e07caf58e7678a780d428c1a8893a95d9c4a556134fd8e7368db5ed596cdb"
ADDED_FEATURES = list(EXCLUDED_COLUMNS)
DIAGNOSTIC_FEATURES = [*FEATURE_COLUMNS, *ADDED_FEATURES]
METRICS = ["ap", "roc_auc", "lift_at_10", "brier", "log_loss"]


def join_development_features(development: pd.DataFrame, additions: pd.DataFrame) -> pd.DataFrame:
    """Left join without replacing any existing feature, target, or row ordering."""
    if list(additions.columns) != [ID_COLUMN, *ADDED_FEATURES]:
        raise ValueError("Only Employee_ID and the six authorized columns may be joined")
    if additions[ID_COLUMN].isna().any() or not additions[ID_COLUMN].is_unique:
        raise ValueError("Source IDs must be present and unique")
    merged = development.merge(additions, on=ID_COLUMN, how="left", sort=False,
                               validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Source is missing development IDs")
    pd.testing.assert_frame_equal(
        merged.loc[:, development.columns], development.reset_index(drop=True)
    )
    return merged.drop(columns="_merge")


def read_additions(source: Path, employee_ids: pd.Series) -> pd.DataFrame:
    # Discard all non-development rows immediately; never read raw targets or old features.
    pieces = []
    for chunk in pd.read_csv(source, usecols=[ID_COLUMN, *ADDED_FEATURES],
                             dtype={ID_COLUMN: "string"}, chunksize=10000):
        pieces.append(chunk.loc[chunk[ID_COLUMN].isin(employee_ids), [ID_COLUMN, *ADDED_FEATURES]])
    return pd.concat(pieces, ignore_index=True)


def metric_comparison(y, old_probability, new_probability, ids) -> dict:
    old = probability_metrics(y, old_probability, ids)
    new = probability_metrics(y, new_probability, ids)
    return {
        "baseline_12": {key: old[key] for key in METRICS},
        "extended_18": {key: new[key] for key in METRICS},
        "delta_extended_minus_baseline": {key: new[key] - old[key] for key in METRICS},
    }


def run_diagnostic(development_path: Path, source: Path, baseline: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Use a new diagnostic run; never overwrite a prior run")
    started = datetime.now(timezone.utc).isoformat()
    old_paths = [baseline / name for name in ("metadata.json", "metrics.json", "oof_predictions.csv")]
    old_hashes = {path.name: sha256_file(path) for path in old_paths}
    metadata = json.loads(old_paths[0].read_text(encoding="utf-8"))
    old_metrics = json.loads(old_paths[1].read_text(encoding="utf-8"))
    development_hash = sha256_file(development_path)
    if development_hash != metadata["development"]["sha256"]:
        raise ValueError("Development hash differs from baseline")
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("Original CSV hash mismatch")
    if old_hashes["oof_predictions.csv"] != metadata["oof_sha256"]:
        raise ValueError("Baseline OOF hash mismatch")
    parameters = {name: builder().get_params(deep=False) for name, builder in MODEL_BUILDERS.items()}
    assert parameters == metadata["models"]
    assert metadata["cross_validation"] == CV_CONFIG
    assert metadata["random_seed"] == RANDOM_STATE == 42
    assert metadata["feature_order"] == FEATURE_COLUMNS
    versions = {"python": platform.python_version(), **{
        name: version(name) for name in ("numpy", "pandas", "scikit-learn")
    }}
    assert versions == metadata["dependency_versions"]
    frame = pd.read_csv(development_path, dtype={ID_COLUMN: "string"}, float_precision="round_trip")
    validate_development(frame)
    assert len(frame) == 80000
    old = pd.read_csv(old_paths[2], dtype={ID_COLUMN: "string"}, float_precision="round_trip")
    assert len(old) == len(frame) and old[ID_COLUMN].is_unique and old[ID_COLUMN].notna().all()
    assert set(old[ID_COLUMN]) == set(frame[ID_COLUMN])
    assert fold_assignment_sha256(old[ID_COLUMN], old["fold"].to_numpy()) == metadata[
        "fold_assignment_sha256"
    ]
    old = old.set_index(ID_COLUMN).loc[frame[ID_COLUMN]].reset_index()
    assert old[ID_COLUMN].equals(frame[ID_COLUMN])
    y = frame[TARGET_COLUMN].to_numpy(dtype=int)
    np.testing.assert_array_equal(old["y_true"], y)
    assignments = old["fold"].to_numpy()
    assert set(assignments) == {1, 2, 3}
    ids = frame[ID_COLUMN].to_numpy(dtype=str)
    # Verify every stored comparator metric before fitting either extended model.
    for name in MODEL_BUILDERS:
        probability = old[f"{name}_probability"].to_numpy()
        pooled = probability_metrics(y, probability, ids)
        for key in METRICS:
            np.testing.assert_allclose(pooled[key], old_metrics["models"][name]["pooled_oof"][key],
                                       rtol=0, atol=1e-12)
        for saved in old_metrics["models"][name]["folds"]:
            mask = assignments == saved["fold"]
            calculated = probability_metrics(y[mask], probability[mask], ids[mask])
            for key in METRICS:
                np.testing.assert_allclose(calculated[key], saved[f"validation_{key}"],
                                           rtol=0, atol=1e-12)
    joined = join_development_features(frame, read_additions(source, frame[ID_COLUMN]))
    assert len(joined) == 80000
    assert not {ID_COLUMN, TARGET_COLUMN} & set(DIAGNOSTIC_FEATURES)
    features = joined.loc[:, DIAGNOSTIC_FEATURES].copy()
    assert all(pd.api.types.is_numeric_dtype(features[col]) for col in ADDED_FEATURES[1:])
    new_oof = old[[ID_COLUMN, "fold", "y_true"]].copy()
    results = {}
    transformed_names = {}
    for name, builder in MODEL_BUILDERS.items():
        predictions = np.full(len(frame), np.nan)
        coverage = np.zeros(len(frame), dtype=int)
        folds = []
        for fold in (1, 2, 3):
            train = np.flatnonzero(assignments != fold)
            valid = np.flatnonzero(assignments == fold)
            assert not np.intersect1d(train, valid).size
            pipeline = Pipeline([
                ("preprocess", build_preprocessor(
                    scale_numeric=False, include_diagnostic_features=True)),
                ("model", builder()),
            ])
            print(f"{name} fold {fold}/3: fitting extended features", flush=True)
            fit_start = time.perf_counter()
            pipeline.fit(features.iloc[train], y[train])
            fit_seconds = time.perf_counter() - fit_start
            names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
            assert not any(col in feature for col in (ID_COLUMN, TARGET_COLUMN) for feature in names)
            transformed_names[f"{name}_fold_{fold}"] = names
            predictions[valid] = class_one_probability(pipeline, features.iloc[valid])
            coverage[valid] += 1
            train_probability = class_one_probability(pipeline, features.iloc[train])
            comparison = metric_comparison(y[valid], old[f"{name}_probability"].to_numpy()[valid],
                                           predictions[valid], ids[valid])
            folds.append({
                "fold": fold, "train_rows": len(train), "validation_rows": len(valid),
                "fit_seconds": fit_seconds,
                "extended_train_roc_auc": float(roc_auc_score(y[train], train_probability)),
                "extended_train_ap": float(average_precision_score(y[train], train_probability)),
                **comparison,
            })
            print(json.dumps({"model": name, "fold": fold, **comparison}), flush=True)
        assert np.all(coverage == 1)
        new_oof[f"{name}_probability"] = predictions
        results[name] = {"folds": folds, "pooled_oof": metric_comparison(
            y, old[f"{name}_probability"].to_numpy(), predictions, ids)}
    pd.testing.assert_frame_equal(new_oof[[ID_COLUMN, "fold", "y_true"]],
                                  old[[ID_COLUMN, "fold", "y_true"]])
    assert development_hash == sha256_file(development_path)
    assert old_hashes == {path.name: sha256_file(path) for path in old_paths}
    metrics = {
        "scope": "paired_development_oof_not_test_metrics", "models": results,
        "reference": old_metrics["reference"], "delta_direction": "extended_minus_baseline",
        "uncertainty": "Not estimated; no unpaired confidence intervals used.",
    }
    run_metadata = {
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "development": {"sha256": development_hash, "rows": len(frame)},
        "source": {"filename": source.name, "sha256": SOURCE_SHA256},
        "baseline_run": str(baseline), "baseline_files_sha256": old_hashes,
        "feature_set_version": "employee-performance-diagnostic-18-v1",
        "feature_order": DIAGNOSTIC_FEATURES, "added_features": ADDED_FEATURES,
        "excluded_for_demonstrated_leakage": [],
        "temporal_availability": {col: "UNKNOWN" for col in DIAGNOSTIC_FEATURES},
        "cross_validation": {**CV_CONFIG, "membership_source": "baseline OOF by Employee_ID"},
        "fold_assignment_sha256": fold_assignment_sha256(frame[ID_COLUMN], assignments),
        "random_seed": RANDOM_STATE, "models": parameters, "dependency_versions": versions,
        "preprocessing": {"fit_inside_each_training_fold": True, "scale_numeric": False,
                          "transformed_feature_names": transformed_names},
        "checks": {"original_features_unchanged": True, "id_target_fold_order_preserved": True,
                   "baseline_unchanged": True, "final_test_accessed": False,
                   "one_prediction_per_record_per_model": True},
        "top_k": "ceil(0.1*N); probability descending, Employee_ID string ascending; mergesort",
    }
    output.mkdir(parents=True, exist_ok=False)
    new_oof.to_csv(output / "oof_predictions.csv", index=False)
    run_metadata["oof_sha256"] = sha256_file(output / "oof_predictions.csv")
    for filename, value in (("metrics.json", metrics), ("metadata.json", run_metadata)):
        (output / filename).write_text(json.dumps(value, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_diagnostic(args.development, args.source, args.baseline_run, args.output_dir)


if __name__ == "__main__":
    main()
