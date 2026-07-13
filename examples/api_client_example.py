#!/usr/bin/env python3
"""
Example client for the SAM3 Bolt Marking Detector API.

Shows how to call /detect, /detect/url, /detect/base64, /detect/batch,
and /health from another agent or script.

Usage:
    python examples/api_client_example.py                          # uses defaults
    python examples/api_client_example.py --base-url http://gpu:8000
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import List

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_result(label: str, result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  image_id   : {result.get('image_id')}")
    print(f"  prediction : {result.get('prediction')}")
    print(f"  confidence : {result.get('confidence'):.4f}")
    print(f"  elapsed_ms : {result.get('elapsed_ms')}")
    if result.get("error"):
        print(f"  ERROR      : {result['error']}")
    features = result.get("features") or {}
    if features:
        print(f"  best_prompt: {features.get('best_prompt', 'N/A')}")
        print(f"  sam_score  : {features.get('best_sam_score', 'N/A')}")
        print(f"  reason     : {features.get('judgment_reason', 'N/A')}")


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------

def check_health(base_url: str) -> dict:
    """GET /health — verify the server is running."""
    resp = requests.get(f"{base_url}/health", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    print(f"[health] status={data['status']}  model_loaded={data['model_loaded']}  "
          f"device={data['device']}  uptime={data['uptime_seconds']}s")
    return data


# ---------------------------------------------------------------------------
# 2. Single image upload  (POST /detect)
# ---------------------------------------------------------------------------

def detect_from_file(base_url: str, image_path: str, image_id: str | None = None) -> dict:
    """POST /detect — upload a local image file."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with open(path, "rb") as f:
        files = {"image": (path.name, f, "image/jpeg")}
        data = {"image_id": image_id or path.stem, "use_feature_model": "true"}
        resp = requests.post(f"{base_url}/detect", files=files, data=data, timeout=120)

    resp.raise_for_status()
    result = resp.json()
    _print_result("POST /detect (file upload)", result)
    return result


# ---------------------------------------------------------------------------
# 3. Image URL  (POST /detect/url)
# ---------------------------------------------------------------------------

def detect_from_url(base_url: str, image_url: str, image_id: str | None = None) -> dict:
    """POST /detect/url — let the server fetch the image from a URL."""
    payload = {"url": image_url, "image_id": image_id or image_url.split("/")[-1]}
    resp = requests.post(
        f"{base_url}/detect/url",
        json=payload,
        params={"use_feature_model": "true"},
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()
    _print_result("POST /detect/url", result)
    return result


# ---------------------------------------------------------------------------
# 4. Base64 encoded image  (POST /detect/base64)
# ---------------------------------------------------------------------------

def detect_from_base64(base_url: str, image_path: str, image_id: str | None = None) -> dict:
    """POST /detect/base64 — send image as base64 string."""
    path = Path(image_path)
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    data = {
        "image_base64": b64,
        "image_id": image_id or path.stem,
        "use_feature_model": "true",
    }
    resp = requests.post(f"{base_url}/detect/base64", data=data, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    _print_result("POST /detect/base64", result)
    return result


# ---------------------------------------------------------------------------
# 5. Batch processing  (POST /detect/batch)
# ---------------------------------------------------------------------------

def detect_batch(base_url: str, image_paths: List[str]) -> dict:
    """POST /detect/batch — upload multiple images in one request."""
    files = []
    for p in image_paths:
        path = Path(p)
        if not path.exists():
            print(f"  [skip] not found: {path}")
            continue
        files.append(("images", (path.name, open(path, "rb"), "image/jpeg")))

    if not files:
        raise ValueError("No valid images to upload")

    try:
        data = {"use_feature_model": "true"}
        resp = requests.post(f"{base_url}/detect/batch", files=files, data=data, timeout=600)
        resp.raise_for_status()
        batch = resp.json()
    finally:
        for _, (_, fh, _) in files:
            fh.close()

    print(f"\n{'='*60}")
    print(f"  POST /detect/batch  ({len(batch['results'])} images, "
          f"{batch['total_elapsed_ms']:.0f} ms total)")
    print(f"{'='*60}")
    for r in batch["results"]:
        status = "OK" if not r.get("error") else f"ERR: {r['error']}"
        print(f"  [{status:>12}]  {r['image_id']:<40}  "
              f"pred={r['prediction']:<8}  conf={r['confidence']:.3f}  "
              f"{r['elapsed_ms']:.0f}ms")
    return batch


# ---------------------------------------------------------------------------
# 6. Server info  (GET /info)
# ---------------------------------------------------------------------------

def get_info(base_url: str) -> dict:
    """GET /info — retrieve server configuration."""
    resp = requests.get(f"{base_url}/info", timeout=5)
    resp.raise_for_status()
    data = resp.json()
    print(f"\n[info] checkpoint={data['checkpoint']}")
    print(f"       device={data['device']}  sam_root={data['sam_root']}")
    return data


# ---------------------------------------------------------------------------
# Agent integration pattern
# ---------------------------------------------------------------------------

class Sam3Client:
    """Minimal reusable client for programmatic agent use."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_ready(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200 and r.json().get("model_loaded", False)
        except Exception:
            return False

    def detect_file(self, image_path: str, image_id: str | None = None) -> dict:
        path = Path(image_path)
        with open(path, "rb") as f:
            files = {"image": (path.name, f, "image/jpeg")}
            data = {"image_id": image_id or path.stem}
            r = requests.post(f"{self.base_url}/detect", files=files, data=data, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def detect_url(self, image_url: str) -> dict:
        r = requests.post(
            f"{self.base_url}/detect/url",
            json={"url": image_url},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def detect_batch(self, image_paths: List[str]) -> List[dict]:
        files = []
        for p in image_paths:
            path = Path(p)
            files.append(("images", (path.name, open(path, "rb"), "image/jpeg")))
        try:
            r = requests.post(f"{self.base_url}/detect/batch", files=files, timeout=600)
            r.raise_for_status()
            return r.json()["results"]
        finally:
            for _, (_, fh, _) in files:
                fh.close()


# ---------------------------------------------------------------------------
# Demo entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM3 API client examples")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--image", default=None, help="Path to a single bolt image")
    parser.add_argument("--image-url", default=None, help="URL of a bolt image")
    parser.add_argument("--batch-dir", default=None, help="Directory of images for batch demo")
    args = parser.parse_args()

    # Always check health first
    print("1) Health check")
    check_health(args.base_url)

    # GET /info
    print("\n2) Server info")
    get_info(args.base_url)

    # Single file upload
    if args.image:
        print("\n3) Single image detection (file upload)")
        detect_from_file(args.base_url, args.image)

        print("\n4) Single image detection (base64)")
        detect_from_base64(args.base_url, args.image)
    else:
        print("\n[skip] Pass --image to demo POST /detect")

    # URL-based detection
    if args.image_url:
        print("\n5) URL-based detection")
        detect_from_url(args.base_url, args.image_url)
    else:
        print("\n[skip] Pass --image-url to demo POST /detect/url")

    # Batch detection
    if args.batch_dir:
        paths = sorted(Path(args.batch_dir).glob("*.jpg"))[:5]
        if paths:
            print(f"\n6) Batch detection ({len(paths)} images)")
            detect_batch(args.base_url, [str(p) for p in paths])
    else:
        print("\n[skip] Pass --batch-dir to demo POST /detect/batch")

    # Show agent integration pattern
    print(f"""
{'='*60}
  Agent integration example
{'='*60}

    from api_client_example import Sam3Client

    client = Sam3Client("{args.base_url}")
    if client.is_ready():
        result = client.detect_file("bolt_001.jpg")
        print(result["prediction"], result["confidence"])

        # Batch
        results = client.detect_batch(["img1.jpg", "img2.jpg"])
        for r in results:
            print(r["image_id"], r["prediction"])
""")


if __name__ == "__main__":
    main()
