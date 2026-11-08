"""Bounded, preregistered Mexwell selection on saved development folds only."""

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from eris_ai.baseline import (
    CV_CONFIG, MODEL_BUILDERS, class_one_probability, fold_assignment_sha256,
    make_folds, probability_metrics, validate_development, validate_transformed_feature_names,
)
from eris_ai.data_preparation import sha256_file
from eris_ai.preprocessing import FEATURE_COLUMNS, build_preprocessor, select_model_features

METRICS = ['ap', 'roc_auc', 'brier', 'log_loss', 'precision_at_10',
           'recall_at_10', 'lift_at_10']
CRITERIA = {
    'minimum_ap_improvement': 0.01, 'minimum_lift': 1.20,
    'brier_no_worse_than_prior': True, 'log_loss_no_worse_than_prior': True,
    'minimum_joint_passing_folds': 2,
    'bootstrap_repetitions': 1000, 'bootstrap_seed': 42,
    'bootstrap_lower_delta_ap_strictly_above': 0,
    'bootstrap_lower_lift_strictly_above': 1,
}


def candidate_configs():
    configs = {}
    for algorithm, depths in [('gradient_boosting', (1, 2)), ('random_forest', (3, 6))]:
        for depth in depths:
            for leaf in (100, 500):
                model = MODEL_BUILDERS[algorithm]().set_params(
                    max_depth=depth, min_samples_leaf=leaf)
                configs[f'{algorithm}_d{depth}_leaf{leaf}'] = {
                    'algorithm': algorithm, 'parameters': model.get_params(deep=False),
                }
    return configs


def screening_checks(pooled, reference, folds, reference_folds, criteria):
    joint = sum(
        f['validation']['ap'] > r['ap'] and f['validation']['lift_at_10'] > 1
        for f, r in zip(folds, reference_folds, strict=True)
    )
    return {
        'ap': pooled['ap'] - reference['ap'] >= criteria['minimum_ap_improvement'],
        'lift': pooled['lift_at_10'] >= criteria['minimum_lift'],
        'brier': pooled['brier'] <= reference['brier'],
        'log_loss': pooled['log_loss'] <= reference['log_loss'],
        'fold_consistency': joint >= criteria['minimum_joint_passing_folds'],
        'joint_passing_folds': int(joint),
    }


def passes(checks):
    return all(checks[k] for k in ('ap', 'lift', 'brier', 'log_loss', 'fold_consistency'))


def paired_intervals(y, p, prior, ids, folds, repetitions=1000, seed=42):
    """Conditional OOF percentile intervals, paired within fold x label strata."""
    strata = [np.flatnonzero((folds == f) & (y == label))
              for f in np.unique(folds) for label in (0, 1)]
    rng = np.random.default_rng(seed)
    ranks = np.argsort(np.argsort(ids.astype(str), kind='stable'), kind='stable')
    samples = []
    for _ in range(repetitions):
        ix = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        sy, sp = y[ix], p[ix]
        order = np.lexsort((ranks[ix], -sp))
        k = int(np.ceil(0.1 * len(ix)))
        samples.append([
            average_precision_score(sy, sp) - average_precision_score(sy, prior[ix]),
            sy[order[:k]].mean() / sy.mean(),
        ])
    intervals = np.percentile(samples, [2.5, 97.5], axis=0)
    return {'delta_ap': intervals[:, 0].tolist(), 'lift_at_10': intervals[:, 1].tolist()}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def run(development, baseline, output):
    if output.exists():
        raise ValueError('Use a new run directory; existing artifacts are immutable')
    old_paths = [baseline / n for n in ('metadata.json', 'metrics.json', 'oof_predictions.csv')]
    old_hashes = {p.name: sha256_file(p) for p in old_paths}
    old_meta = json.loads(old_paths[0].read_text(encoding='utf-8'))
    old_metrics = json.loads(old_paths[1].read_text(encoding='utf-8'))
    dev_hash = sha256_file(development)
    assert dev_hash == old_meta['development']['sha256']
    assert dev_hash == '20e0a090ced201386fbfe30fbb7bec698419e851f561f27373153b3f926f62f8'
    assert old_hashes['oof_predictions.csv'] == old_meta['oof_sha256']
    assert old_meta['feature_order'] == FEATURE_COLUMNS
    assert old_meta['cross_validation'] == CV_CONFIG
    frame = pd.read_csv(development, dtype={'Employee_ID': 'string'},
                        float_precision='round_trip')
    validate_development(frame)
    assert len(frame) == 80000
    old = pd.read_csv(old_paths[2], dtype={'Employee_ID': 'string'},
                      float_precision='round_trip')
    assert len(old) == len(frame) and old.Employee_ID.is_unique
    assert old.Employee_ID.equals(frame.Employee_ID)
    y, ids = frame.Resigned.to_numpy(int), frame.Employee_ID.to_numpy(str)
    np.testing.assert_array_equal(old.y_true, y)
    assignments = old.fold.to_numpy(int)
    _, expected = make_folds(y)
    np.testing.assert_array_equal(assignments, expected + 1)
    fold_hash = fold_assignment_sha256(frame.Employee_ID, assignments)
    assert fold_hash == old_meta['fold_assignment_sha256']
    prior = old.reference_probability.to_numpy(float)
    reference = probability_metrics(y, prior, ids)
    reference_folds = []
    for fold in (1, 2, 3):
        valid = assignments == fold
        np.testing.assert_allclose(prior[valid], y[~valid].mean(), rtol=0, atol=1e-15)
        reference_folds.append(probability_metrics(y[valid], prior[valid], ids[valid]))
    for name in [*MODEL_BUILDERS, 'reference']:
        calculated = probability_metrics(y, old[f'{name}_probability'].to_numpy(), ids)
        saved = old_metrics['reference'] if name == 'reference' else old_metrics['models'][name]
        for key in METRICS:
            assert abs(calculated[key] - saved['pooled_oof'][key]) < 1e-12

    metadata = {
        'registered_before_fit_at_utc': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'Shallower trees and larger leaves reduce baseline overfitting; '
                      'useful signal must satisfy all fixed OOF gates.',
        'development': {'sha256': dev_hash, 'rows': len(frame)},
        'baseline_run': str(baseline), 'baseline_hashes': old_hashes,
        'fold_assignment_sha256': fold_hash, 'cross_validation': CV_CONFIG,
        'feature_set_version': 'employee-performance-v1', 'feature_order': FEATURE_COLUMNS,
        'target_mapping': {'False': 0, 'True': 1}, 'random_seed': 42,
        'preprocessing': {'fit_inside_each_training_fold': True, 'scale_numeric': False,
                          'builder': 'eris_ai.preprocessing.build_preprocessor'},
        'top_k': 'ceil(0.10*N); probability descending, string Employee_ID ascending; stable',
        'criteria': CRITERIA, 'candidate_configs': candidate_configs(),
        'selection': 'Among point-gate passers: highest AP, then highest Lift; '
                     'bootstrap only the leader, no fallback if uncertainty gate fails.',
        'uncertainty': '1000 paired bootstrap draws stratified by fold and label; '
                       'percentile 95%; describes fixed OOF, not full selection uncertainty.',
        'final_test_policy': 'Only after all OOF gates pass: freeze winner, fit development, '
                             'evaluate final once with AP delta>=0.01, Lift>=1.20, '
                             'Brier and Log Loss<=development-prior Dummy. '
                             '1000 paired label-stratified bootstrap draws, seed=42; '
                             'lower delta AP>0 and lower Lift>1. No reselection.',
        'dependency_versions': {'python': platform.python_version(), **{
            k: version(k) for k in ('numpy', 'pandas', 'scikit-learn')}},
        'code_sha256': {str(p): sha256_file(p) for p in [
            Path(__file__), Path(__file__).with_name('baseline.py'),
            Path(__file__).with_name('preprocessing.py')]},
    }
    output.mkdir(parents=True, exist_ok=False)
    registration = output / 'metadata.json'
    save_json(registration, metadata)
    registration_hash = sha256_file(registration)
    # Read the actual frozen registration before the first estimator is fitted.
    metadata = json.loads(registration.read_text(encoding='utf-8'))
    results = {'scope': 'development_OOF_not_test', 'reference': {
        'pooled_oof': reference, 'folds': reference_folds}, 'models': {}}
    oof = old[['Employee_ID', 'fold', 'y_true', 'reference_probability']].copy()
    x = select_model_features(frame)
    for name, config in metadata['candidate_configs'].items():
        p = np.full(len(y), np.nan)
        coverage = np.zeros(len(y), dtype=int)
        records = []
        for fold in (1, 2, 3):
            train, valid = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
            pipeline = Pipeline([
                ('preprocess', build_preprocessor(scale_numeric=False)),
                ('model', MODEL_BUILDERS[config['algorithm']]().set_params(**config['parameters'])),
            ])
            print(f'{name} fold {fold}/3: fitting', flush=True)
            pipeline.fit(x.iloc[train], y[train])
            names = validate_transformed_feature_names(pipeline)
            p[valid] = class_one_probability(pipeline, x.iloc[valid])
            coverage[valid] += 1
            training = probability_metrics(y[train], class_one_probability(
                pipeline, x.iloc[train]), ids[train])
            validation = probability_metrics(y[valid], p[valid], ids[valid])
            records.append({'fold': fold, 'train_rows': len(train), 'validation_rows': len(valid),
                            'train': training, 'validation': validation,
                            'train_minus_validation': {
                                k: training[k] - validation[k] for k in METRICS},
                            'transformed_feature_names': names})
            print(f"AP={validation['ap']:.6f} Lift={validation['lift_at_10']:.6f}", flush=True)
        assert np.all(coverage == 1)
        pooled = probability_metrics(y, p, ids)
        checks = screening_checks(pooled, reference, records, reference_folds, metadata['criteria'])
        results['models'][name] = {
            'pooled_oof': pooled, 'folds': records, 'screening': checks,
            'mean_train_minus_validation': {
                k: float(np.mean([r['train_minus_validation'][k] for r in records])) for k in METRICS},
        }
        oof[f'{name}_probability'] = p
        oof.to_csv(output / 'oof_predictions.csv', index=False)
        save_json(output / 'metrics.json', results)
    eligible = [n for n, r in results['models'].items() if passes(r['screening'])]
    winner = None
    if eligible:
        leader = max(eligible, key=lambda n: (results['models'][n]['pooled_oof']['ap'],
                                              results['models'][n]['pooled_oof']['lift_at_10']))
        intervals = paired_intervals(y, oof[f'{leader}_probability'].to_numpy(), prior,
                                     ids, assignments)
        results['bootstrap'] = {'candidate': leader, 'intervals': intervals}
        if intervals['delta_ap'][0] > 0 and intervals['lift_at_10'][0] > 1:
            winner = leader
    assert sha256_file(registration) == registration_hash
    assert sha256_file(development) == dev_hash
    assert old_hashes == {p.name: sha256_file(p) for p in old_paths}
    results.update({
        'status': 'OOF_QUALIFIED_PENDING_FINAL' if winner else 'NO_QUALIFIED_MEXWELL_MODEL',
        'selected_model_id': winner, 'final_test_accessed': False,
        'registration_sha256': registration_hash,
        'oof_sha256': sha256_file(output / 'oof_predictions.csv'),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    save_json(output / 'metrics.json', results)
    print(json.dumps({'status': results['status'], 'selected_model_id': winner}), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.development, args.baseline_run, args.output_dir)
