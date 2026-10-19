"""FastAPI service for PAD screening.

GET  /              intake form
POST /api/predict   clinical features (+ optional accelerometry)
                    -> verdict, ABI risk band, confidence
GET  /api/health    model load status
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.schemas import PatientInput, PredictionResponse
from src.prediction.abi import band_reference, classify_abi
from src.prediction.ensemble import get_ensemble
from src.prediction.fusion import get_fusion

BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load every model at startup so the first request is not the slow one.
    get_ensemble()
    get_fusion()
    yield


app = FastAPI(title="PAD Screening", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse(
        request, "index.html", {"bands": band_reference()})


@app.get("/api/health")
def health():
    ensemble = get_ensemble()
    fusion = get_fusion()
    return {"status": "ok",
            "models": sorted(ensemble.models),
            "threshold": fusion.clinical_threshold,
            "sensor_threshold": fusion.sensor_threshold}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    features = patient.to_features()
    sensor = patient.Accelerometry.model_dump() if patient.Accelerometry else None
    try:
        clinical_probability = get_ensemble().predict(features)["probability"]
        # Every verdict comes from the fusion layer: with m = 0 it combines and
        # recalibrates the two clinical branches on real patients.
        result = get_fusion().predict(clinical_probability, features, sensor)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    branches = {"clinical": clinical_probability, "nhanes": result["cohort_probability"]}
    if sensor is not None:
        branches["sensor"] = result["sensor_probability"]
    return PredictionResponse(
        pad_detected=result["pad_detected"],
        verdict="PAD detected" if result["pad_detected"] else "No PAD detected",
        confidence=result["confidence"],
        probability=result["probability"],
        risk=classify_abi(patient.ABI),
        threshold=result["threshold"],
        modalities=["clinical", "sensor"] if sensor is not None else ["clinical"],
        branch_probabilities=branches,
    )
