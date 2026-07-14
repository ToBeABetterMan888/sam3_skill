#!/usr/bin/env python3
"""
FastAPI web service wrapping Sam3MarkingDetector for agent consumption.

Endpoints:
    POST /detect          – upload an image, get prediction + features
    POST /detect/url      – provide an image URL, get prediction + features
    POST /detect/batch    – upload multiple images, get all results
    GET  /health          – liveness / readiness check
    GET  /info            – model & config metadata
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load .env from project root (parent of src/)
_PROJECT_ROOT = Path(os.environ.get("BOLT_MARKING_ROOT", Path(__file__).resolve().parent.parent))
load_dotenv(_PROJECT_ROOT / ".env")

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration (from .env, then env vars, then defaults)
# ---------------------------------------------------------------------------
PROJECT_ROOT = _PROJECT_ROOT
DEFAULT_SAM3_ROOT = Path(os.environ.get("SAM3_ROOT", "/udat/sam/sam3"))
DEFAULT_CHECKPOINT = os.environ.get("CHECKPOINT", str(PROJECT_ROOT / "models" / "sam3.pt"))
DEFAULT_FEATURE_MODEL = os.environ.get(
    "FEATURE_MODEL", str(PROJECT_ROOT / "models" / "feature_judger" / "random_forest_final.joblib")
)
DEFAULT_DEVICE = os.environ.get("DEVICE", "cuda")
DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("PORT", "8000"))

from sam3_marking_detector import Sam3MarkingDetector, Sam3DetectionResult  # noqa: E402

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class DetectURLRequest(BaseModel):
    url: str = Field(..., description="Public URL of the bolt image")
    image_id: Optional[str] = Field(None, description="Optional image identifier")


class FeatureJudgerRequest(BaseModel):
    use_feature_model: bool = Field(True, description="Apply RandomForest feature judger on top of rule result")


class SingleResult(BaseModel):
    image_id: str
    prediction: str
    confidence: float
    features: Optional[dict] = None
    candidates: Optional[list] = None
    error: Optional[str] = None
    elapsed_ms: float


class BatchResult(BaseModel):
    results: List[SingleResult]
    total_elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


class InfoResponse(BaseModel):
    project_root: str
    checkpoint: str
    feature_model: str
    device: str
    sam_root: str
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# App & detector singleton
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SAM3 Bolt Marking Detector API",
    description=(
        "Detect bolt anti-loosening marking lines using SAM3 segmentation "
        "and classify them as normal / loose / unknown."
    ),
    version="1.0.0",
)

_START_TIME = time.time()
_detector: Optional[Sam3MarkingDetector] = None
_feature_model = None  # loaded lazily


def _get_detector() -> Sam3MarkingDetector:
    global _detector
    if _detector is None:
        device = os.environ.get("DEVICE", DEFAULT_DEVICE)
        checkpoint = os.environ.get("CHECKPOINT", DEFAULT_CHECKPOINT)
        _detector = Sam3MarkingDetector(
            checkpoint_path=checkpoint,
            device=device,
        )
    return _detector


def _load_feature_model():
    global _feature_model
    if _feature_model is not None:
        return _feature_model
    model_path = os.environ.get("FEATURE_MODEL", DEFAULT_FEATURE_MODEL)
    if not Path(model_path).exists():
        return None
    import joblib
    _feature_model = joblib.load(model_path)
    return _feature_model


def _apply_feature_model(result_dict: dict) -> dict:
    """Override rule-based prediction with RandomForest if available."""
    fm = _load_feature_model()
    if fm is None:
        return result_dict
    import pandas as pd

    model = fm["model"]
    numeric_cols = fm["numeric_cols"]
    categorical_cols = fm["categorical_cols"]

    x = pd.DataFrame([result_dict]).reindex(columns=numeric_cols + categorical_cols)
    pred_label = str(model.predict(x)[0])
    result_dict["rule_prediction"] = result_dict["prediction"]
    result_dict["rule_confidence"] = result_dict["confidence"]
    result_dict["prediction"] = pred_label
    result_dict["final_model"] = "random_forest_feature_judger"
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        classes = list(model.named_steps["model"].classes_)
        result_dict["confidence"] = float(max(probs))
        for cls, prob in zip(classes, probs):
            result_dict[f"prob_{cls}"] = float(prob)
    return result_dict


def _read_upload_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image bytes")
    return img


def _process_image(image_bgr: np.ndarray, image_id: str, use_feature_model: bool = True) -> SingleResult:
    t0 = time.time()
    detector = _get_detector()
    det: Sam3DetectionResult = detector.detect(image_bgr, image_id=image_id)

    # Build flat feature dict (same shape as evaluate() output)
    features = det.features or {}
    result_dict = {
        "image_id": image_id,
        "prediction": det.prediction,
        "confidence": det.confidence,
        "error": det.error,
        "candidates": det.candidates,
    }
    for k, v in features.items():
        if k == "components":
            continue  # too large for JSON response
        result_dict[k] = v

    if use_feature_model:
        result_dict = _apply_feature_model(result_dict)

    elapsed = (time.time() - t0) * 1000
    return SingleResult(
        image_id=result_dict.pop("image_id", image_id),
        prediction=result_dict.pop("prediction", det.prediction),
        confidence=result_dict.pop("confidence", det.confidence),
        features={k: v for k, v in result_dict.items() if k not in ("error", "candidates")},
        candidates=result_dict.get("candidates"),
        error=result_dict.get("error"),
        elapsed_ms=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_detector is not None,
        device=os.environ.get("DEVICE", DEFAULT_DEVICE),
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@app.get("/info", response_model=InfoResponse)
def info():
    return InfoResponse(
        project_root=str(PROJECT_ROOT),
        checkpoint=os.environ.get("CHECKPOINT", DEFAULT_CHECKPOINT),
        feature_model=os.environ.get("FEATURE_MODEL", DEFAULT_FEATURE_MODEL),
        device=os.environ.get("DEVICE", DEFAULT_DEVICE),
        sam_root=str(DEFAULT_SAM3_ROOT),
    )


@app.post("/detect", response_model=SingleResult, summary="Detect marking on uploaded image")
async def detect(
    image: UploadFile = File(..., description="Bolt image (jpg/png)"),
    image_id: Optional[str] = Form(None),
    use_feature_model: bool = Form(True),
):
    """Upload a single bolt image and get the loosening detection result."""
    data = await image.read()
    img_bgr = _read_upload_to_bgr(data)
    iid = image_id or image.filename or "upload"
    return _process_image(img_bgr, iid, use_feature_model=use_feature_model)


@app.post("/detect/url", response_model=SingleResult, summary="Detect marking from image URL")
def detect_from_url(body: DetectURLRequest, use_feature_model: bool = True):
    """Provide a publicly accessible image URL for detection."""
    try:
        resp = requests.get(body.url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {exc}")
    img_bgr = _read_upload_to_bgr(resp.content)
    iid = body.image_id or body.url.split("/")[-1] or "url_image"
    return _process_image(img_bgr, iid, use_feature_model=use_feature_model)


@app.post("/detect/batch", response_model=BatchResult, summary="Batch detect on multiple images")
async def detect_batch(
    images: List[UploadFile] = File(..., description="Multiple bolt images"),
    use_feature_model: bool = Form(True),
):
    """Upload multiple bolt images and get all detection results."""
    t0 = time.time()
    results: List[SingleResult] = []
    for idx, image in enumerate(images):
        data = await image.read()
        img_bgr = _read_upload_to_bgr(data)
        iid = image.filename or f"image_{idx}"
        results.append(_process_image(img_bgr, iid, use_feature_model=use_feature_model))
    return BatchResult(
        results=results,
        total_elapsed_ms=round((time.time() - t0) * 1000, 1),
    )


@app.post("/detect/base64", response_model=SingleResult, summary="Detect marking from base64-encoded image")
def detect_from_base64(
    image_base64: str = Form(..., description="Base64-encoded image bytes"),
    image_id: Optional[str] = Form(None),
    use_feature_model: bool = Form(True),
):
    """Submit a base64-encoded image for detection."""
    try:
        data = base64.b64decode(image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data")
    img_bgr = _read_upload_to_bgr(data)
    iid = image_id or "base64_image"
    return _process_image(img_bgr, iid, use_feature_model=use_feature_model)


@app.get("/detect/visualization")
def get_visualization_example():
    """Return a placeholder to document the visualization endpoint pattern."""
    return {
        "hint": "POST an image to /detect, then use the returned features to render overlay locally.",
        "sam_mask_field": "features.merged_sam_area",
        "refined_mask_field": "features.refined_area",
    }


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Bolt Marking Detector Web API")
    parser.add_argument("--host", type=str, default=None, help=f"Override HOST (.env or {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=None, help=f"Override PORT (.env or {DEFAULT_PORT})")
    parser.add_argument("--device", type=str, default=None, help="Override DEVICE env var")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override CHECKPOINT env var")
    parser.add_argument("--feature-model", type=str, default=None, help="Override FEATURE_MODEL env var")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    # CLI args override .env / env vars
    if args.device:
        os.environ["DEVICE"] = args.device
    if args.checkpoint:
        os.environ["CHECKPOINT"] = args.checkpoint
    if args.feature_model:
        os.environ["FEATURE_MODEL"] = args.feature_model

    host = args.host or os.environ.get("HOST", DEFAULT_HOST)
    port = args.port or int(os.environ.get("PORT", str(DEFAULT_PORT)))

    print(f"Starting SAM3 Bolt Marking Detector API on {host}:{port}")
    print(f"  device     = {os.environ.get('DEVICE', DEFAULT_DEVICE)}")
    print(f"  checkpoint = {os.environ.get('CHECKPOINT', DEFAULT_CHECKPOINT)}")
    print(f"  feature    = {os.environ.get('FEATURE_MODEL', DEFAULT_FEATURE_MODEL)}")

    uvicorn.run(
        "web_api:app",
        host=host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
