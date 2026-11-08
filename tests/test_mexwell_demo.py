"""Integration checks use the real local demo bundle; never fit or read held-out data."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from eris_ai.api import create_app
from eris_ai.mexwell_demo import (
    ADDITIVITY_ATOL, MODEL_ID, SCHEMA_VERSION, VALIDATION_STATUS, build_bundle,
)
from eris_ai.preprocessing import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'artifacts/mexwell_rf12_demo_v1/bundle.joblib'
KEY = 'test-service-key-at-least-24-characters'
HEADERS = {'X-ERIS-Service-Key': KEY}


@pytest.fixture(scope='module')
def samples():
    if not BUNDLE.exists():
        pytest.skip('Build the real Mexwell demo bundle to run integration checks')
    development = ROOT / 'data/processed/development.csv'
    frame = pd.read_csv(development, usecols=FEATURE_COLUMNS, nrows=12)
    return [{'schema_version': SCHEMA_VERSION,
             'employee': {'employee_id': f'demo-check-{i}',
                          'features': row[FEATURE_COLUMNS].to_dict()}}
            for i, row in frame.iterrows()]


@pytest.fixture(scope='module')
def app(samples):
    return create_app(service_key=KEY, demo_enabled=True, demo_bundle_path=str(BUNDLE))


@pytest.fixture(scope='module')
def client(app):
    with TestClient(app) as client:
        yield client


def test_predict_explain_actual_bundle_and_grouped_additivity(client, samples):
    for body in samples:
        p = client.post('/v1/predict', json=body, headers=HEADERS)
        e = client.post('/v1/explain', json=body, headers=HEADERS)
        assert p.status_code == e.status_code == 200
        prediction, explanation = p.json(), e.json()
        assert prediction['score'] == pytest.approx(explanation['score'], rel=0, abs=1e-12)
        assert prediction['model_id'] == explanation['model_id'] == MODEL_ID
        assert explanation['validation_status'] == VALIDATION_STATUS
        assert prediction['employee_id'] == body['employee']['employee_id']
        assert set(explanation['contributions']) == set(FEATURE_COLUMNS)
        assert abs(explanation['base_value'] + sum(explanation['contributions'].values())
                   - prediction['score']) <= ADDITIVITY_ATOL
        assert 0 <= prediction['score'] <= 1
        assert explanation['additivity_error'] <= ADDITIVITY_ATOL
        assert 'probability' in explanation['shap_units']


def test_reference_id_never_changes_prediction(client, samples):
    body = copy.deepcopy(samples[0])
    original = client.post('/v1/predict', json=body, headers=HEADERS).json()
    body['employee']['employee_id'] = 'different-reference-only'
    changed = client.post('/v1/predict', json=body, headers=HEADERS).json()
    assert changed['score'] == pytest.approx(original['score'], rel=0, abs=1e-12)
    assert changed['employee_id'] == 'different-reference-only'


@pytest.mark.parametrize('path', ['/v1/predict', '/v1/explain'])
@pytest.mark.parametrize('case', [
    'missing_feature', 'extra_id', 'extra_target', 'numeric_string', 'boolean_number',
    'bad_date', 'unknown_category', 'bad_education', 'null_feature', 'bad_version',
    'missing_reference_id',
])
def test_invalid_input_rejected(client, samples, path, case):
    body = copy.deepcopy(samples[0])
    features = body['employee']['features']
    if case == 'missing_feature':
        features.pop('Hire_Date')
    elif case == 'extra_id':
        features['Employee_ID'] = 'forbidden'
    elif case == 'extra_target':
        features['Resigned'] = 1
    elif case == 'numeric_string':
        features['Age'] = '35'
    elif case == 'boolean_number':
        features['Age'] = True
    elif case == 'bad_date':
        features['Hire_Date'] = '2020-02-30'
    elif case == 'unknown_category':
        features['Department'] = 'not-a-known-category'
    elif case == 'bad_education':
        features['Education_Level'] = 'not-a-known-education'
    elif case == 'null_feature':
        features['Monthly_Salary'] = None
    elif case == 'bad_version':
        body['schema_version'] = 'unconfigured'
    else:
        body['employee'].pop('employee_id')
    assert client.post(path, json=body, headers=HEADERS).status_code == 422


@pytest.mark.parametrize('path', ['/v1/predict', '/v1/explain'])
def test_missing_or_wrong_auth_rejected(client, samples, path):
    assert client.post(path, json=samples[0]).status_code == 401
    assert client.post(path, json=samples[0], headers={
        'X-ERIS-Service-Key': 'wrong-key'}).status_code == 401


@pytest.mark.parametrize('enabled,path_kind', [(False, 'existing'), (True, 'missing'),
                                             (True, 'unset'), (True, 'corrupt')])
def test_unavailable_demo_returns_503(enabled, path_kind, tmp_path):
    corrupt = tmp_path / 'broken.joblib'
    corrupt.write_bytes(b'invalid-bundle')
    path = {'existing': str(BUNDLE), 'missing': str(tmp_path / 'missing.joblib'),
            'unset': '', 'corrupt': str(corrupt)}[path_kind]
    with TestClient(create_app(KEY, demo_enabled=enabled, demo_bundle_path=path)) as client:
        body = {'schema_version': SCHEMA_VERSION,
                'employee': {'employee_id': 'reference-only', 'features': {}}}
        for endpoint in ('/v1/predict', '/v1/explain'):
            assert client.post(endpoint, json=body, headers=HEADERS).status_code == 503
        assert client.get('/health').status_code == 200


def test_environment_configuration_explicit_opt_in(monkeypatch, samples):
    monkeypatch.setenv('ERIS_MEXWELL_DEMO_BUNDLE', str(BUNDLE))
    monkeypatch.delenv('ERIS_MEXWELL_DEMO_ENABLED', raising=False)
    with TestClient(create_app(KEY)) as client:
        assert client.post('/v1/predict', json=samples[0], headers=HEADERS).status_code == 503
    monkeypatch.setenv('ERIS_MEXWELL_DEMO_ENABLED', 'true')
    with TestClient(create_app(KEY)) as client:
        assert client.post('/v1/predict', json=samples[0], headers=HEADERS).status_code == 200


def test_additivity_failure_is_not_returned_as_success(app, client, samples, monkeypatch):
    explainer = app.state.model.explainer
    original = explainer.shap_values

    def broken(*args, **kwargs):
        values = np.array(original(*args, **kwargs), copy=True)
        values[0, 0, 1] += 0.1
        return values

    monkeypatch.setattr(explainer, 'shap_values', broken)
    assert client.post('/v1/explain', json=samples[0], headers=HEADERS).status_code == 503


def test_existing_bundle_reused_without_fitting(samples, monkeypatch, capsys):
    from sklearn.ensemble import RandomForestClassifier

    def forbidden(*args, **kwargs):
        raise AssertionError('Existing bundle must not be refitted')

    monkeypatch.setattr(RandomForestClassifier, 'fit', forbidden)
    build_bundle(ROOT / 'data/processed/development.csv',
                 ROOT / 'artifacts/baseline_gb_rf_20261102_201500', BUNDLE)
    assert '"reused_existing_bundle": true' in capsys.readouterr().out


def test_demo_does_not_enable_batch(client, samples):
    response = client.post('/v1/predict-batch', headers=HEADERS, json={
        'schema_version': SCHEMA_VERSION, 'employees': [samples[0]['employee']]})
    assert response.status_code == 503
