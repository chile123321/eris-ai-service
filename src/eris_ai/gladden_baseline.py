"""Fixed Dummy and logistic baselines for the filtered Gladden temporal cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eris_ai.gladden_cohort import EXPECTED_SHA256, FEATURE_COLUMNS, HORIZON, TWO_WAY

CATEGORICAL = ["role", "shift", "team"]
NUMERIC = [
    "presence_days_28d", "absence_days_28d", "recorded_days_28d",
    "observable_calendar_days_28d", "no_record_calendar_days_28d",
]
SEED = 42


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preprocessor():
    return ColumnTransformer([
        ("category", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ]), NUMERIC),
    ], remainder="drop")


def probability_metrics(y, p):
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    return {"ap": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def weekly_priorities(validation, probabilities, *, random_expectation=False):
    """Rank within each eligible week; constant baseline gets analytic random expectations."""
    frame = validation[["sub_ID", "snapshot_date", "target"]].reset_index(drop=True).copy()
    frame["probability"] = probabilities
    selected = np.zeros(len(frame), dtype=bool)
    selection_chance = np.zeros(len(frame))
    rows = []
    for week, group in frame.groupby("snapshot_date", sort=True):
        n, positives = len(group), int(group.target.sum())
        k = math.ceil(0.10 * n)
        prevalence = positives / n
        selection_chance[group.index] = k / n
        if random_expectation:
            if group.probability.nunique() != 1:
                raise ValueError("Random expectation is only valid for constant scores")
            true_alerts = k * prevalence
        else:
            ranking = group.sort_values(["probability", "sub_ID"],
                                        ascending=[False, True], kind="mergesort")
            chosen = ranking.index[:k]
            selected[chosen] = True
            true_alerts = int(frame.loc[chosen, "target"].sum())
        precision = true_alerts / k
        rows.append({
            "snapshot_date": str(week.date()), "eligible_people": n,
            "positive_snapshots": positives, "prevalence": prevalence, "alert_count": k,
            "true_positive_alerts": true_alerts, "precision_at_10": precision,
            "recall_at_10": true_alerts / positives if positives else None,
            "lift_at_10": precision / prevalence if positives else None,
        })
    weekly = pd.DataFrame(rows)
    positive_ids = frame.loc[frame.target.eq(1), "sub_ID"].nunique()
    if random_expectation:
        frame["selection_chance"] = selection_chance
        # Independent uniform top-k draws across weeks; no arbitrary tied ID selection.
        unique_alerted = sum(
            1 - np.prod(1 - group.selection_chance.to_numpy())
            for _, group in frame.loc[frame.target.eq(1)].groupby("sub_ID")
        )
    else:
        unique_alerted = int(frame.loc[selected & frame.target.eq(1), "sub_ID"].nunique())
    summary = {
        "interpretation": "analytic random expectation" if random_expectation else "observed ranking",
        "weekly": rows, "weeks": len(rows), "alert_count": int(weekly.alert_count.sum()),
        "true_positive_alerts": float(weekly.true_positive_alerts.sum()),
        "macro_weekly": {key: float(weekly[key].mean()) if weekly[key].notna().any() else None
                         for key in ["precision_at_10", "recall_at_10", "lift_at_10"]},
        "weeks_with_positive_cases": int(weekly.positive_snapshots.gt(0).sum()),
        "unique_resigning_people": int(positive_ids),
        "unique_resigning_people_alerted": float(unique_alerted) if random_expectation else unique_alerted,
        "unique_case_coverage": float(unique_alerted / positive_ids) if positive_ids else None,
        "unique_count_note": "Dummy is expected unique people under independent weekly random "
            "selection; logistic is actual distinct people alerted on a positive 90-day snapshot.",
    }
    return summary, selected, selection_chance


def load_verified(cohort_dir):
    metadata_path = cohort_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["data_version"] == EXPECTED_SHA256
    assert metadata["feature_allowlist"] == FEATURE_COLUMNS == CATEGORICAL + NUMERIC
    expected_windows = {name: list(window) for name, window in TWO_WAY.items()}
    assert metadata["two_way_exploratory"]["windows"] == expected_windows
    hashes = {"metadata.json": sha256(metadata_path)}
    for name, expected in metadata["output_sha256"].items():
        hashes[name] = sha256(cohort_dir / name)
        if hashes[name] != expected:
            raise ValueError(f"Cohort hash mismatch: {name}")
    dates = ["snapshot_date", "horizon_end", "exit_date", "followup_end", "last_feature_record_date"]
    frame = pd.read_csv(cohort_dir / "cohort_binary.csv",
                        dtype={"sub_ID": "string", "shift": "string", "team": "string"},
                        parse_dates=dates)
    splits = pd.read_csv(cohort_dir / "split_two_way_exploratory.csv", dtype={"sub_ID": "string"},
                         parse_dates=["snapshot_date", "horizon_end"])
    key = ["sub_ID", "snapshot_date"]
    assert not frame.duplicated(key).any() and not splits.duplicated(key).any()
    pd.testing.assert_frame_equal(frame[key + ["horizon_end", "target"]],
                                  splits[key + ["horizon_end", "target"]])
    frame["split"] = splits.split
    assert set(frame.label_status) == {"POSITIVE", "NEGATIVE"}
    assert (frame.target == frame.label_status.eq("POSITIVE").astype(int)).all()
    assert frame.horizon_end.eq(frame.snapshot_date + HORIZON).all()
    assert frame.horizon_end.le(pd.Timestamp("2022-06-30")).all()
    assert frame.last_feature_record_date.le(frame.snapshot_date).all()
    positive = frame.target.eq(1)
    assert frame.loc[positive, "exit_type"].eq("Resignation").all()
    assert frame.loc[positive, "exit_date"].gt(frame.loc[positive, "snapshot_date"]).all()
    assert frame.loc[positive, "exit_date"].le(frame.loc[positive, "horizon_end"]).all()
    negative = frame.loc[~positive]
    assert negative.followup_end.ge(negative.horizon_end).all()
    assert (negative.exit_date.isna() | negative.exit_date.gt(negative.horizon_end)).all()
    assert frame[FEATURE_COLUMNS].notna().all().all()
    subsets = {}
    for name, count, cases in [("train", 17160, 44), ("validation", 8788, 27)]:
        start, end = map(pd.Timestamp, TWO_WAY[name])
        selected = frame.loc[frame.split.eq(name)].reset_index(drop=True)
        assert frame.split.eq(name).equals(frame.snapshot_date.between(start, end))
        assert len(selected) == count
        assert selected.loc[selected.target.eq(1), "sub_ID"].nunique() == cases
        assert selected.snapshot_date.min() == start and selected.snapshot_date.max() == end
        saved = metadata["two_way_exploratory"]["sets"][name]
        assert selected.sub_ID.nunique() == saved["people"]
        assert int(selected.target.sum()) == saved["statuses"]["POSITIVE"]
        subsets[name] = selected
    train, validation = subsets["train"], subsets["validation"]
    assert train.horizon_end.max() < validation.snapshot_date.min()
    assert len(set(train.sub_ID) & set(validation.sub_ID)) == 468
    return train, validation, metadata, hashes


def run(cohort_dir: Path, output: Path):
    if output.exists():
        raise ValueError("Run exists; refusing to overwrite or retrain")
    train, validation, cohort_metadata, input_hashes = load_verified(cohort_dir)
    models = {
        "dummy_prior": DummyClassifier(strategy="prior", random_state=SEED),
        "logistic_regression": LogisticRegression(C=1.0, max_iter=2000, class_weight=None,
                                                   random_state=SEED, solver="lbfgs"),
    }
    predictions = validation[["sub_ID", "snapshot_date", "horizon_end", "exit_date", "target"]].copy()
    results, names, fit_times = {}, {}, {}
    x_train, x_validation = train.loc[:, FEATURE_COLUMNS], validation.loc[:, FEATURE_COLUMNS]
    started = datetime.now(timezone.utc).isoformat()
    for name, model in models.items():
        pipeline = Pipeline([("preprocess", preprocessor()), ("model", model)])
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(x_train, train.target)
        fit_times[name] = time.perf_counter() - start
        names[name] = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
        class_one = list(model.classes_).index(1)
        # Exactly one validation prediction pass for each fixed trained pipeline.
        p = pipeline.predict_proba(x_validation)[:, class_one]
        is_dummy = name == "dummy_prior"
        if is_dummy:
            np.testing.assert_allclose(p, train.target.mean(), rtol=0, atol=1e-15)
        priorities, selected, chances = weekly_priorities(validation, p, random_expectation=is_dummy)
        results[name] = {"probability": probability_metrics(validation.target, p),
                         "priorities": priorities,
                         "solver_iterations": model.n_iter_.tolist() if not is_dummy else None}
        predictions[f"{name}_probability"] = p
        predictions["dummy_weekly_selection_probability" if is_dummy else "logistic_alert_top10"] = (
            chances if is_dummy else selected
        )
    results["delta_logistic_minus_dummy"] = {
        metric: results["logistic_regression"]["probability"][metric]
        - results["dummy_prior"]["probability"][metric]
        for metric in results["dummy_prior"]["probability"]
    }
    metadata = {
        "scope": "fixed_two_model_simulation_temporal_validation_only",
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_version": cohort_metadata["data_version"],
        "feature_set_version": cohort_metadata["feature_set_version"],
        "input_directory": str(cohort_dir), "input_sha256": input_hashes,
        "feature_allowlist": FEATURE_COLUMNS, "transformed_features": names,
        "preprocessing": "train-only Pipeline: categorical most-frequent imputation + one-hot "
            "ignore unknown; numeric median imputation + StandardScaler; no weighting/resampling",
        "models": {name: model.get_params(deep=False) for name, model in models.items()},
        "seed": SEED, "fit_seconds": fit_times,
        "split": cohort_metadata["two_way_exploratory"],
        "train_label_end": str(train.horizon_end.max().date()),
        "validation_first_snapshot": str(validation.snapshot_date.min().date()),
        "train_prevalence": float(train.target.mean()),
        "validation_prevalence": float(validation.target.mean()),
        "weekly_top_k": "ceil(0.1 * eligible employees that week); logistic ties by string sub_ID "
            "ascending. Dummy evaluated by random expectation, not an arbitrary tied ranking.",
        "uncertainty": "Not estimated; no independent-row intervals or significance claims. "
            "Future paired resampling must cluster by sub_ID, retaining all their snapshots.",
        "limitations": [
            "Binary cohort excludes snapshots with Termination in the next 90 days; estimates "
            "apply to this retrospectively filtered simulated cohort, not the whole workforce.",
            "Only 27 distinct validation resignations; repeated positive snapshots are dependent.",
            "468 employees overlap train/validation; evaluation is future weeks for existing "
            "workers, not a test of generalization to new employees.",
            "No independent final evaluation; no tuning, SHAP, API changes or deployment.",
            cohort_metadata["followup_assumption"], cohort_metadata["record_availability_assumption"],
        ],
        "legacy_final_test_accessed": False,
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ["numpy", "pandas", "scikit-learn"]}},
        "script_sha256": sha256(Path(__file__)),
    }
    for filename, digest in input_hashes.items():
        assert sha256(cohort_dir / filename) == digest
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / "validation_predictions.csv", index=False, date_format="%Y-%m-%d")
    metadata["predictions_sha256"] = sha256(output / "validation_predictions.csv")
    for filename, value in [("metadata.json", metadata), ("metrics.json", results)]:
        (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({name: {"probability": result["probability"],
        "priorities": {k: v for k, v in result["priorities"].items() if k != "weekly"}}
        for name, result in results.items() if name in models}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.cohort_dir, args.output_dir)
