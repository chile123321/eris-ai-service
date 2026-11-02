const { test } = require('node:test');
const assert = require('node:assert/strict');
const { createAiClient } = require('./ai-client.cjs');

const options = { baseUrl: 'http://ai:8000', serviceKey: 'test-only-key' };

test('single prediction request forwards authentication and schema', async () => {
  const client = createAiClient({ ...options, fetchImpl: async (url, request) => {
    assert.equal(url.pathname, '/v1/predict');
    assert.equal(request.headers['X-ERIS-Service-Key'], options.serviceKey);
    assert.equal(JSON.parse(request.body).schema_version, 'unconfigured');
    return { ok: true, json: async () => ({ predictions: [] }) };
  }});
  await client.predictOne('unconfigured', { employee_id: '1', features: {} });
});

test('batch prediction preserves order while chunking requests', async () => {
  let calls = 0;
  const client = createAiClient({ ...options, fetchImpl: async (_, request) => {
    calls++;
    const body = JSON.parse(request.body);
    assert.ok(body.employees.length <= 100);
    return { ok: true, json: async () => ({ model_id: 'one', predictions: body.employees }) };
  }});
  const employees = Array.from({ length: 201 }, (_, index) => ({
    employee_id: String(index), features: {}
  }));
  assert.deepEqual((await client.predictBatch('unconfigured', employees)).predictions, employees);
  assert.equal(calls, 3);
});

test('timeout maps to gateway timeout', async () => {
  const client = createAiClient({ ...options, fetchImpl: async () => {
    throw new DOMException('', 'TimeoutError');
  }});
  await assert.rejects(client.modelInfo(), { statusCode: 504 });
});

test('model change between chunks rejects the whole batch', async () => {
  let call = 0;
  const client = createAiClient({ ...options, fetchImpl: async () => ({
    ok: true,
    json: async () => ({ model_id: String(call++), predictions: [] })
  }) });
  const employees = Array.from({ length: 101 }, (_, index) => ({ employee_id: String(index) }));
  await assert.rejects(client.predictBatch('unconfigured', employees), { statusCode: 503 });
});
