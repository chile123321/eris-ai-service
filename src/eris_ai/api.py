import os
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

load_dotenv()


class Employee(BaseModel):
    model_config = ConfigDict(extra="forbid")
    employee_id: str = Field(strict=True, min_length=1, max_length=100)
    features: dict[str, Any] = Field(max_length=100)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = Field(strict=True, min_length=1, max_length=100)
    employee: Employee


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = Field(strict=True, min_length=1, max_length=100)
    employees: list[Employee] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({e.employee_id for e in self.employees}) != len(self.employees):
            raise ValueError("Duplicate employee IDs in batch")
        return self


def create_app(service_key: str | None = None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.key = (
            service_key if service_key is not None else os.getenv("ERIS_SERVICE_KEY", "")
        )
        if len(app.state.key) < 24 or app.state.key.startswith("replace-"):
            raise RuntimeError("Set ERIS_SERVICE_KEY to a random secret of at least 24 characters")
        app.state.model = None
        yield

    app = FastAPI(title="ERIS AI Service", version="0.1.0", lifespan=lifespan)

    def authenticate(x_eris_service_key: Annotated[str | None, Header()] = None):
        if not x_eris_service_key or not secrets.compare_digest(
            x_eris_service_key.encode(), app.state.key.encode()
        ):
            raise HTTPException(401, "Invalid service key")

    def require_model():
        if app.state.model is None:
            raise HTTPException(503, "Model is not configured")
        return app.state.model

    protected = [Depends(authenticate)]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready", dependencies=protected)
    def ready():
        require_model()
        return {"status": "ready"}

    @app.get("/v1/model-info", dependencies=protected)
    def model_info():
        return require_model().info()

    @app.post(
        "/v1/predict",
        dependencies=protected,
        responses={401: {"description": "Unauthorized"}, 503: {"description": "Model unavailable"}},
    )
    def predict(request: PredictRequest):
        return require_model().predict(request.schema_version, [request.employee])

    @app.post(
        "/v1/predict-batch",
        dependencies=protected,
        responses={401: {"description": "Unauthorized"}, 503: {"description": "Model unavailable"}},
    )
    def predict_batch(request: BatchRequest):
        return require_model().predict(request.schema_version, request.employees)

    return app


app = create_app()
