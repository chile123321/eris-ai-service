"""Verify saved baseline OOF and bootstrap it without fitting any estimator."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from eris_ai.baseline import (
    CV_CONFIG, MODEL_BUILDERS, fold_assignment_sha256, make_folds,
    probability_metrics, validate_development,
)
from eris_ai.data_preparation import sha256_file
from eris_ai.preprocessing import EXCLUDED_COLUMNS, FEATURE_COLUMNS

METRICS = ["roc_auc", "ap", "brier", "log_loss", "precision_at_10",
           "recall_at_10", "lift_at_10"]


def bootstrap_oof(y, probabilities, ids, repetitions=1000, seed=42):
    """Paired stratified row bootstrap; models remain fixed throughout."""
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y == label) for label in (0, 1)]
    # Integer ranks implement the original lexicographic Employee_ID tie-break.
    ranks = np.argsort(np.argsort(np.asarray(ids, dtype=str), kind="stable"), kind="stable")
    samples = {name: np.empty((repetitions, 3)) for name in probabilities}
    for iteration in range(repetitions):
        indices = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        sampled_y = y[indices]
        k = int(np.ceil(0.10 * len(indices)))
        for name, p in probabilities.items():
            sampled_p = p[indices]
            order = np.lexsort((ranks[indices], -sampled_p))
            lift = sampled_y[order[:k]].mean() / sampled_y.mean()
            samples[name][iteration] = (
                roc_auc_score(sampled_y, sampled_p),
                average_precision_score(sampled_y, sampled_p), lift,
            )
        if (iteration + 1) % 100 == 0:
            print(f"Bootstrap {iteration + 1}/{repetitions}", flush=True)
    return {
        name: {metric: np.percentile(values[:, column], [2.5, 97.5]).tolist()
               for column, metric in enumerate(("roc_auc", "ap", "lift_at_10"))}
        for name, values in samples.items()
    }


def audit(development, run):
    metrics_path, metadata_path = run / "metrics.json", run / "metadata.json"
    oof_path = run / "oof_predictions.csv"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    original_oof_hash = sha256_file(oof_path)
    dev_hash = sha256_file(development)
    assert dev_hash == metadata["development"]["sha256"], "Development hash mismatch"
    dev = pd.read_csv(development, dtype={"Employee_ID": "string"}, float_precision="round_trip")
    oof = pd.read_csv(oof_path, dtype={"Employee_ID": "string"}, float_precision="round_trip")
    validate_development(dev)
    assert len(oof) == len(dev) == 80000
    assert oof.Employee_ID.notna().all() and oof.Employee_ID.is_unique
    assert set(oof.Employee_ID) == set(dev.Employee_ID)
    aligned = oof.set_index("Employee_ID").loc[dev.Employee_ID].reset_index()
    y = dev.Resigned.to_numpy(dtype=int)
    assert np.array_equal(aligned.y_true, y), "Target mismatch"
    assert metadata["feature_order"] == FEATURE_COLUMNS
    assert metadata["cross_validation"] == CV_CONFIG
    assert metadata["random_seed"] == 42
    assert metadata["models"] == {
        name: builder().get_params(deep=False) for name, builder in MODEL_BUILDERS.items()
    }
    prep = metadata["preprocessing"]
    assert prep["fit_inside_each_training_fold"] and not prep["scale_numeric"]
    assert prep["dense_one_hot"] and prep["date_epoch"] == "1970-01-01"
    assert all(not any(c in name for c in ["Employee_ID", "Resigned", *EXCLUDED_COLUMNS])
               for name in prep["transformed_feature_names"])
    old_hash = fold_assignment_sha256(oof.Employee_ID, oof.fold.to_numpy())
    assert old_hash == metadata["fold_assignment_sha256"]
    labels = set(oof.fold)
    assert labels in ({0, 1, 2}, {1, 2, 3})
    offset = 1 if labels == {0, 1, 2} else 0
    _, expected = make_folds(y)
    assert np.array_equal(aligned.fold.to_numpy() + offset, expected + 1)
    ids = aligned.Employee_ID.to_numpy(dtype=str)
    recalculated = {}
    largest_difference = 0.0
    for name in [*MODEL_BUILDERS, "reference"]:
        p = aligned[f"{name}_probability"].to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        saved = metrics["reference"] if name == "reference" else metrics["models"][name]
        pooled = probability_metrics(y, p, ids)
        folds = []
        for fold in (1, 2, 3):
            valid = aligned.fold.to_numpy() + offset == fold
            assert not set(ids[valid]) & set(ids[~valid])
            if name == "reference":
                np.testing.assert_allclose(p[valid], y[~valid].mean(), rtol=0, atol=1e-15)
            calculated = probability_metrics(y[valid], p[valid], ids[valid])
            previous = next(f for f in saved["folds"] if f["fold"] + offset == fold)
            prefix = "" if name == "reference" else "validation_"
            for key in METRICS:
                difference = abs(calculated[key] - previous[prefix + key])
                largest_difference = max(largest_difference, difference)
                assert difference < 1e-12, f"Fold metric mismatch: {name}/{fold}/{key}"
            folds.append({"fold": fold, **{key: calculated[key] for key in METRICS}})
        for key in METRICS:
            difference = abs(pooled[key] - saved["pooled_oof"][key])
            largest_difference = max(largest_difference, difference)
            assert difference < 1e-12, f"OOF metric mismatch: {name}/{key}"
        recalculated[name] = {"pooled_oof": {k: pooled[k] for k in METRICS}, "folds": folds}

    print("OOF, metadata and all pooled/fold metrics verified; starting bootstrap", flush=True)
    intervals = bootstrap_oof(
        y, {n: aligned[f"{n}_probability"].to_numpy() for n in MODEL_BUILDERS}, ids,
    )
    # Only presentation labels change; original probabilities and membership are retained.
    if offset:
        oof["fold"] += 1
        for result in [*metrics["models"].values(), metrics["reference"]]:
            for fold in result["folds"]:
                fold["fold"] += 1
        oof.to_csv(oof_path, index=False)
        metadata["fold_label_migration"] = {
            "mapping": {"0": 1, "1": 2, "2": 3},
            "previous_assignment_sha256": old_hash,
            "previous_oof_sha256": original_oof_hash,
            "membership_and_probabilities_unchanged": True,
        }
    metadata["fold_assignment_sha256"] = fold_assignment_sha256(
        oof.Employee_ID, oof.fold.to_numpy()
    )
    metadata["oof_sha256"] = sha256_file(oof_path)
    metrics["oof_audit"] = recalculated
    metrics["bootstrap"] = {
        "method": "stratified paired row bootstrap of saved OOF; percentile 95%",
        "repetitions": 1000, "seed": 42,
        "top_k": "ceil(0.10*N) recomputed per sample; probability desc, string ID asc",
        "limitation": "Descriptive conditional on existing OOF predictions; excludes full "
                      "training and model-selection uncertainty.",
        "intervals": intervals,
    }
    metadata["audit"] = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(oof), "fold_labels": [1, 2, 3],
        "max_metric_absolute_difference": largest_difference,
        "development_hash_verified": True, "model_configurations_verified": True,
        "id_target_alignment_verified": True, "probability_bounds_verified": True,
        "fold_membership_verified_against_seed": True,
        "retrained": False, "final_test_accessed": False,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"audit": metadata["audit"], "intervals": intervals}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    audit(args.development, args.run)
