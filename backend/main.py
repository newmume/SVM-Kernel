"""FastAPI service for the Vercel-native SVM Kernel Lab."""

from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

try:
    from .svm_engine import build_model_response
except ImportError:  # Vercel runs with backend/ as the service root.
    from svm_engine import build_model_response


class ModelRequest(BaseModel):
    dataset: Literal["rings", "moons", "xor", "linear"] = "rings"
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf"
    C: float = Field(10.0, ge=0.05, le=200)
    gamma: float = Field(1.0, ge=0.001, le=20)
    degree: int = Field(3, ge=2, le=6)
    coef0: float = Field(0.0, ge=-3, le=3)
    balanced: bool = False
    points: int = Field(160, ge=40, le=400)
    noise: float = Field(0.1, ge=0, le=0.5)
    label_noise: float = Field(0.0, ge=0, le=0.25)
    class_ratio: float = Field(0.45, ge=0.2, le=0.8)
    separation: float = Field(1.25, ge=0.6, le=2.0)
    inner_radius: float = Field(0.9, ge=0.4, le=1.3)
    outer_radius: float = Field(2.05, ge=1.4, le=3.0)
    seed: int = Field(7, ge=0, le=9999)
    resolution: int = Field(80, ge=50, le=120)


app = FastAPI(
    title="SVM Kernel Lab API",
    description="scikit-learn SVC calculations for the interactive Vercel demo",
    version="2.0.0",
)


@app.middleware("http")
async def no_store(request, call_next):
    response: Response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api")
def api_root():
    return {"service": "svm-kernel-lab", "version": "2.0.0", "status": "online"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/model")
def compute_model(params: ModelRequest):
    if params.dataset == "rings" and params.outer_radius <= params.inner_radius + 0.1:
        raise HTTPException(status_code=422, detail="Outer radius must exceed inner radius by at least 0.1.")
    try:
        return build_model_response(params)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
