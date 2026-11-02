'use strict';

class AiServiceError extends Error {
  constructor(statusCode, message) {
    super(message);
    this.statusCode = statusCode;
  }
}

function createAiClient({ baseUrl, serviceKey, timeoutMs = 10000, fetchImpl = fetch }) {
  if (!baseUrl || !serviceKey || !Number.isInteger(timeoutMs) || timeoutMs < 1) {
    throw new Error('AI URL, key and positive timeout are required');
  }

  async function request(path, body) {
    let response;
    try {
      response = await fetchImpl(new URL(path, baseUrl), {
        method: body ? 'POST' : 'GET',
        headers: { 'Content-Type': 'application/json', 'X-ERIS-Service-Key': serviceKey },
        body: body ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(timeoutMs)
      });
      if (!response.ok) {
        const status = response.status === 422 ? 422 : response.status === 503 ? 503 : 502;
        throw new AiServiceError(status, 'AI request failed; check model readiness and input contract');
      }
      return await response.json();
    } catch (error) {
      if (error instanceof AiServiceError) throw error;
      const status = ['TimeoutError', 'AbortError'].includes(error.name) ? 504 : 502;
      throw new AiServiceError(status, 'AI service unavailable');
    }
  }

  return {
    modelInfo: () => request('/v1/model-info'),
    predictOne: (schemaVersion, employee) => request('/v1/predict', {
      schema_version: schemaVersion,
      employee
    }),
    predictBatch: async (schemaVersion, employees) => {
      if (!Array.isArray(employees) || !employees.length || employees.length > 1000) {
        throw new AiServiceError(422, 'Provide 1-1000 snapshots per backend call');
      }
      const ids = employees.map((employee) => employee.employee_id);
      if (new Set(ids).size !== ids.length) {
        throw new AiServiceError(422, 'Duplicate employee IDs');
      }
      let combined;
      for (let start = 0; start < employees.length; start += 100) {
        const result = await request('/v1/predict-batch', {
          schema_version: schemaVersion,
          employees: employees.slice(start, start + 100)
        });
        if (combined && result.model_id !== combined.model_id) {
          throw new AiServiceError(503, 'Model changed during batch; retry the whole request');
        }
        if (!combined) combined = { ...result, predictions: [] };
        combined.predictions.push(...result.predictions);
      }
      return combined;
    }
  };
}

module.exports = { createAiClient, AiServiceError };
