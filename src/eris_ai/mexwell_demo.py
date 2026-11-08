"""Explicitly unqualified Mexwell RF demo bundle, single prediction and TreeSHAP."""

import argparse
import json
import platform
from datetime import date, datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from eris_ai.baseline import MODEL_BUILDERS, class_one_probability, validate_development
from eris_ai.data_preparation import EXPECTED_SOURCE_SHA256, sha256_file
from eris_ai.preprocessing import (
    CATEGORICAL_COLUMNS, EDUCATION_ORDER, FEATURE_COLUMNS, NUMERIC_COLUMNS, build_preprocessor,
)

DEVELOPMENT_SHA256 = '20e0a090ced201386fbfe30fbb7bec698419e851f561f27373153b3f926f62f8'
SCHEMA_VERSION = 'mexwell-12-v1'
MODEL_ID = 'mexwell-rf12-demo-v1-20e0a090ced2'
VALIDATION_STATUS = 'NO_QUALIFIED_MEXWELL_MODEL'
ADDITIVITY_ATOL = 1e-8
LIBRARIES = ('numpy', 'pandas', 'scikit-learn', 'shap', 'joblib')
FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class MexwellFeatures(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    Department: str
    Gender: str
    Age: FiniteNumber
    Hire_Date: str
    Years_At_Company: FiniteNumber
    Education_Level: str
    Monthly_Salary: FiniteNumber
    Work_Hours_Per_Week: FiniteNumber
    Overtime_Hours: FiniteNumber
    Sick_Days: FiniteNumber
    Promotions: FiniteNumber
    Employee_Satisfaction_Score: FiniteNumber

    @field_validator('Hire_Date')
    @classmethod
    def iso_date(cls, value):
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError('Hire_Date must be YYYY-MM-DD')
        return value

    @field_validator('Education_Level')
    @classmethod
    def education(cls, value):
        if value not in EDUCATION_ORDER:
            raise ValueError('Unsupported Education_Level')
        return value


def transformed_origins(preprocessor):
    """Map output positions to inputs, without parsing category names or underscores."""
    encoder = preprocessor.named_transformers_['categorical'].named_steps['one_hot']
    origins = [col for col, categories in zip(CATEGORICAL_COLUMNS, encoder.categories_, strict=True)
               for _ in categories]
    origins += [*NUMERIC_COLUMNS, 'Hire_Date', 'Education_Level']
    if len(origins) != len(preprocessor.get_feature_names_out()):
        raise ValueError('Unexpected transformed feature layout')
    return origins


def feature_schema(preprocessor):
    schema = MexwellFeatures.model_json_schema()
    encoder = preprocessor.named_transformers_['categorical'].named_steps['one_hot']
    for col, categories in zip(CATEGORICAL_COLUMNS, encoder.categories_, strict=True):
        schema['properties'][col]['enum'] = categories.tolist()
    schema['properties']['Education_Level']['enum'] = EDUCATION_ORDER
    schema['properties']['Hire_Date']['format'] = 'date'
    return schema


class MexwellDemo:
    def __init__(self, bundle, bundle_sha256=None):
        self.pipeline = bundle['pipeline']
        self.metadata = bundle['metadata']
        self.bundle_sha256 = bundle_sha256
        meta = self.metadata
        if (meta['model_id'] != MODEL_ID or meta['validation_status'] != VALIDATION_STATUS
                or meta['schema_version'] != SCHEMA_VERSION
                or meta['feature_order'] != FEATURE_COLUMNS
                or meta['development']['sha256'] != DEVELOPMENT_SHA256):
            raise ValueError('Not the approved unqualified Mexwell demo bundle')
        if any(meta['library_versions'][k] != version(k) for k in LIBRARIES):
            raise ValueError('Bundle library versions differ from runtime')
        model = self.pipeline.named_steps['model']
        if not isinstance(model, RandomForestClassifier):
            raise ValueError('Demo requires the baseline Random Forest')
        if model.get_params(deep=False) != MODEL_BUILDERS['random_forest']().get_params(deep=False):
            raise ValueError('RF parameters differ from baseline')
        if list(self.pipeline.feature_names_in_) != FEATURE_COLUMNS:
            raise ValueError('Pipeline input schema mismatch')
        if not np.array_equal(model.classes_, [0, 1]):
            raise ValueError('Expected Resigned classes 0 and 1')
        self.class_one = int(np.flatnonzero(model.classes_ == 1)[0])
        pre = self.pipeline.named_steps['preprocess']
        self.origins = transformed_origins(pre)
        if meta['schema'] != feature_schema(pre) or meta['transformed_origins'] != self.origins:
            raise ValueError('Bundle feature mapping mismatch')
        import shap
        # sklearn RF native tree output is class probability, unlike GB log-odds.
        # Training path counts are the background; no held-out records are needed.
        self.explainer = shap.TreeExplainer(
            model, feature_perturbation='tree_path_dependent', model_output='raw')
        if self.explainer.model.tree_output != 'probability':
            raise ValueError('Unexpected TreeSHAP output scale')

    @classmethod
    def load(cls, path):
        # Only load a trusted local bundle generated by this module (joblib uses pickle).
        return cls(joblib.load(path), sha256_file(Path(path)))

    def info(self):
        return {
            'model_id': self.metadata['model_id'], 'schema_version': SCHEMA_VERSION,
            'validation_status': VALIDATION_STATUS, 'demo_mode': True,
            'score_meaning': 'predict_proba for Resigned=1; no forecast horizon',
            'schema': self.metadata['schema'], 'bundle_sha256': self.bundle_sha256,
        }

    def _frame(self, schema_version, employee):
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f'schema_version must be {SCHEMA_VERSION}')
        try:
            values = MexwellFeatures.model_validate(employee.features).model_dump()
        except ValidationError as error:
            fields = sorted({str(e['loc'][0]) for e in error.errors()})
            raise ValueError('Invalid or missing features: ' + ', '.join(fields)) from error
        for col in CATEGORICAL_COLUMNS:
            if values[col] not in self.metadata['schema']['properties'][col]['enum']:
                raise ValueError(f'Unsupported category for {col}')
        return pd.DataFrame([values], columns=FEATURE_COLUMNS)

    def _prediction(self, frame, employee):
        score = float(class_one_probability(self.pipeline, frame)[0])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError('Invalid model output')
        return {
            'employee_id': employee.employee_id, 'score': score, 'target': 'Resigned',
            'positive_class': 1, 'model_id': self.metadata['model_id'],
            'schema_version': SCHEMA_VERSION, 'validation_status': VALIDATION_STATUS,
            'demo_mode': True, 'score_meaning': 'predict_proba for Resigned=1; no forecast horizon',
        }

    def predict(self, schema_version, employee):
        return self._prediction(self._frame(schema_version, employee), employee)

    def explain(self, schema_version, employee):
        frame = self._frame(schema_version, employee)
        result = self._prediction(frame, employee)
        matrix = self.pipeline.named_steps['preprocess'].transform(frame)
        values = np.asarray(self.explainer.shap_values(matrix, check_additivity=True))
        if values.shape != (1, len(self.origins), 2):
            raise RuntimeError('Unexpected TreeSHAP shape')
        contributions = {col: 0.0 for col in FEATURE_COLUMNS}
        for col, value in zip(self.origins, values[0, :, self.class_one], strict=True):
            contributions[col] += float(value)
        base = float(np.asarray(self.explainer.expected_value)[self.class_one])
        reconstructed = base + sum(contributions.values())
        error = abs(reconstructed - result['score'])
        if not np.isfinite(reconstructed) or error > ADDITIVITY_ATOL:
            raise RuntimeError('TreeSHAP additivity check failed')
        return {**result, 'base_value': base, 'contributions': contributions,
                'shap_units': 'probability (0-1), not log-odds; 0.01 = 1 percentage point',
                'explainer': 'TreeSHAP', 'feature_perturbation': 'tree_path_dependent',
                'model_output': 'raw (RandomForest class probability)',
                'background': 'development training path counts stored in the forest',
                'reconstructed_output': reconstructed, 'additivity_error': error,
                'additivity_atol': ADDITIVITY_ATOL,
                'interpretation': 'Explains model prediction behavior, not causes of resignation.'}


def build_bundle(development, baseline_run, output):
    if sha256_file(development) != DEVELOPMENT_SHA256:
        raise ValueError('Development SHA-256 mismatch')
    meta_path = baseline_run / 'metadata.json'
    baseline = json.loads(meta_path.read_text(encoding='utf-8'))
    parameters = MODEL_BUILDERS['random_forest']().get_params(deep=False)
    if (baseline['development']['sha256'] != DEVELOPMENT_SHA256
            or baseline['feature_order'] != FEATURE_COLUMNS
            or baseline['models']['random_forest'] != parameters):
        raise ValueError('Baseline metadata does not match RF12')
    if output.exists():
        existing = MexwellDemo.load(output)
        print(json.dumps({'reused_existing_bundle': True, **existing.info()}), flush=True)
        return
    frame = pd.read_csv(development, dtype={'Employee_ID': 'string'}, float_precision='round_trip')
    validate_development(frame)
    if len(frame) != 80000:
        raise ValueError('Expected 80000 development rows')
    pipeline = Pipeline([('preprocess', build_preprocessor(scale_numeric=False)),
                         ('model', RandomForestClassifier(**parameters))])
    print('Fitting baseline RF once on 80000 development rows for demo packaging', flush=True)
    pipeline.fit(frame.loc[:, FEATURE_COLUMNS], frame.Resigned.to_numpy(int))
    pre = pipeline.named_steps['preprocess']
    baseline_metrics = json.loads((baseline_run / 'metrics.json').read_text(encoding='utf-8'))
    metadata = {
        'model_id': MODEL_ID, 'schema_version': SCHEMA_VERSION, 'demo_only': True,
        'validation_status': VALIDATION_STATUS,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'development': {'sha256': DEVELOPMENT_SHA256, 'rows': len(frame)},
        'source': {'name': 'Mexwell Extended_Employee_Performance_and_Productivity_Data.csv',
                   'sha256': EXPECTED_SOURCE_SHA256},
        'target': 'Resigned', 'target_mapping': {'False': 0, 'True': 1},
        'feature_order': FEATURE_COLUMNS, 'schema': feature_schema(pre),
        'transformed_origins': transformed_origins(pre),
        'model_parameters': parameters, 'random_seed': 42,
        'library_versions': {k: version(k) for k in LIBRARIES},
        'python_version': platform.python_version(),
        'baseline_metadata_sha256': sha256_file(meta_path),
        'baseline_validation': baseline_metrics['models']['random_forest'],
        'evaluation_note': 'Prior development OOF did not qualify; full-development fit '
                           'is demo packaging, not new validation. Final test not accessed.',
        'shap': {'feature_perturbation': 'tree_path_dependent',
                 'output_scale': 'Resigned=1 probability', 'additivity_atol': ADDITIVITY_ATOL},
        'final_test_accessed': False,
    }
    bundle = {'pipeline': pipeline, 'metadata': metadata}
    MexwellDemo(bundle)  # Verify RF/SHAP compatibility before persisting.
    if sha256_file(development) != DEVELOPMENT_SHA256:
        raise ValueError('Development changed during fit')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        joblib.dump(bundle, handle, compress=3)
    print(json.dumps({'model_id': MODEL_ID, 'bundle': str(output),
                      'sha256': sha256_file(output), 'validation_status': VALIDATION_STATUS}),
          flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', type=Path, default=Path('data/processed/development.csv'))
    parser.add_argument('--baseline-run', type=Path,
                        default=Path('artifacts/baseline_gb_rf_20261102_201500'))
    parser.add_argument('--output', type=Path,
                        default=Path('artifacts/mexwell_rf12_demo_v1/bundle.joblib'))
    args = parser.parse_args()
    build_bundle(args.development, args.baseline_run, args.output)
