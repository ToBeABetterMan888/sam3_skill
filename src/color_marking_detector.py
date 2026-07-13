#!/usr/bin/env python3
"""
Color-prior bolt marking detector.

This baseline does not use SAM3 to segment the red anti-loosening mark. It
extracts red paint with conservative color rules, then classifies the mark
geometry as normal / loose / unknown.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from sklearn.metrics import classification_report, confusion_matrix


Label = Literal["normal", "loose", "unknown"]


@dataclass
class ColorDetectionResult:
    image_id: str
    prediction: Label
    confidence: float
    red_mask: Optional[np.ndarray] = None
    cleaned_mask: Optional[np.ndarray] = None
    features: Optional[Dict] = None
    error: Optional[str] = None
    true_label: Optional[str] = None


class ColorMarkingDetector:
    """Detect red marking lines using color and simple geometry."""

    def __init__(
        self,
        min_area_ratio: float = 0.00025,
        min_area_px: int = 25,
        min_component_area: int = 10,
    ):
        self.min_area_ratio = min_area_ratio
        self.min_area_px = min_area_px
        self.min_component_area = min_component_area

    def segment_red_mark(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw and cleaned binary red masks."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        b, g, r = cv2.split(image_bgr)

        # Hue wraps around for red. Keep saturation/value thresholds moderate
        # because the paint can be dark, scratched, or overexposed.
        red_low = cv2.inRange(hsv, np.array([0, 35, 25]), np.array([14, 255, 255]))
        red_high = cv2.inRange(hsv, np.array([165, 35, 25]), np.array([179, 255, 255]))
        hsv_mask = cv2.bitwise_or(red_low, red_high)

        # Complement HSV with channel-excess tests. This catches dull red paint
        # whose hue becomes unstable on reflective metal.
        rg_excess = (r.astype(np.int16) - g.astype(np.int16)) > 18
        rb_excess = (r.astype(np.int16) - b.astype(np.int16)) > 12
        red_dominant = (r > 55) & rg_excess & rb_excess

        # LAB a-channel is high for red/magenta. It is useful on low-saturation
        # crops where HSV alone misses thin paint.
        a_channel = lab[:, :, 1]
        lab_red = (a_channel > 145) & (r > 45) & (r.astype(np.int16) > g.astype(np.int16) + 10)

        raw = ((hsv_mask > 0) | red_dominant | lab_red).astype(np.uint8)

        # Remove tiny sensor/color specks, then reconnect small gaps in a thin
        # marking stroke without growing it into the metal texture.
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel_small)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_small)

        # A short directional-agnostic dilation helps broken paint strokes remain
        # measurable as a single geometric object while preserving disconnected
        # loose marks at larger gaps.
        cleaned = cv2.dilate(cleaned, kernel_small, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_small)

        cleaned = self._filter_components(cleaned)
        return raw, cleaned

    def _filter_components(self, mask: np.ndarray) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        out = np.zeros_like(mask, dtype=np.uint8)
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area >= self.min_component_area:
                out[labels == idx] = 1
        return out

    @staticmethod
    def _pca(points_xy: np.ndarray) -> Dict:
        if len(points_xy) < 3:
            return {
                "angle": 0.0,
                "linearity": 0.0,
                "center": points_xy.mean(axis=0).tolist() if len(points_xy) else [0.0, 0.0],
                "major": 0.0,
                "minor": 0.0,
            }
        center = points_xy.mean(axis=0)
        centered = points_xy - center
        cov = np.cov(centered.T)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vec = vecs[:, order[0]]
        angle = math.atan2(float(vec[1]), float(vec[0]))
        denom = float(vals[0] + vals[1] + 1e-8)
        return {
            "angle": angle,
            "linearity": float(vals[0] / denom),
            "center": center.tolist(),
            "major": float(math.sqrt(max(vals[0], 0.0))),
            "minor": float(math.sqrt(max(vals[1], 0.0))),
        }

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs(a - b) % math.pi
        return float(min(diff, math.pi - diff))

    def compute_features(self, mask: np.ndarray, image_shape: Tuple[int, int, int]) -> Dict:
        h, w = image_shape[:2]
        image_area = h * w
        red_area = int(mask.sum())
        min_area = max(self.min_area_px, int(image_area * self.min_area_ratio))

        features: Dict = {
            "image_h": h,
            "image_w": w,
            "red_area": red_area,
            "red_area_ratio": float(red_area / max(image_area, 1)),
            "min_area": min_area,
            "has_mark": red_area >= min_area,
            "component_count": 0,
            "significant_component_count": 0,
            "global_linearity": 0.0,
            "global_angle": 0.0,
            "largest_component_area": 0,
            "perpendicular_offset": 0.0,
            "parallel_gap": 0.0,
            "top2_angle_difference": 0.0,
            "skeleton_length": 0,
            "endpoint_count": 0,
            "max_gap_inside_bbox": 0.0,
        }
        if not features["has_mark"]:
            return features

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        components: List[Dict] = []
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < self.min_component_area:
                continue
            ys, xs = np.where(labels == idx)
            points = np.column_stack([xs, ys]).astype(np.float32)
            pca = self._pca(points)
            components.append({
                "idx": idx,
                "area": area,
                "bbox": [
                    int(stats[idx, cv2.CC_STAT_LEFT]),
                    int(stats[idx, cv2.CC_STAT_TOP]),
                    int(stats[idx, cv2.CC_STAT_WIDTH]),
                    int(stats[idx, cv2.CC_STAT_HEIGHT]),
                ],
                "centroid": [float(centroids[idx][0]), float(centroids[idx][1])],
                **pca,
            })
        components.sort(key=lambda x: x["area"], reverse=True)

        features["component_count"] = len(components)
        features["significant_component_count"] = sum(
            c["area"] >= max(self.min_component_area, red_area * 0.06) for c in components
        )
        features["largest_component_area"] = components[0]["area"] if components else 0
        features["components"] = components[:8]

        ys, xs = np.where(mask > 0)
        points = np.column_stack([xs, ys]).astype(np.float32)
        global_pca = self._pca(points)
        features["global_linearity"] = global_pca["linearity"]
        features["global_angle"] = global_pca["angle"]

        skeleton = skeletonize(mask > 0).astype(np.uint8)
        features["skeleton_length"] = int(skeleton.sum())
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        neighbors = cv2.filter2D(skeleton, -1, kernel)
        features["endpoint_count"] = int((neighbors == 11).sum())

        if components:
            x, y, bw, bh = components[0]["bbox"]
            pad = 3
            y0, y1 = max(0, y - pad), min(h, y + bh + pad)
            x0, x1 = max(0, x - pad), min(w, x + bw + pad)
            bbox_mask = mask[y0:y1, x0:x1]
            if bbox_mask.size:
                dist = distance_transform_edt(bbox_mask == 0)
                features["max_gap_inside_bbox"] = float(dist.max())

        if len(components) >= 2:
            c1, c2 = components[0], components[1]
            angle = global_pca["angle"]
            axis = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            normal = np.array([-axis[1], axis[0]], dtype=np.float32)
            p1 = np.array(c1["centroid"], dtype=np.float32)
            p2 = np.array(c2["centroid"], dtype=np.float32)
            delta = p2 - p1
            features["perpendicular_offset"] = float(abs(np.dot(delta, normal)))
            features["parallel_gap"] = float(abs(np.dot(delta, axis)))
            features["top2_angle_difference"] = self._angle_diff(c1["angle"], c2["angle"])

        return features

    def classify(self, features: Dict) -> Tuple[Label, float]:
        if not features.get("has_mark", False):
            return "unknown", 0.88

        min_dim = min(features["image_h"], features["image_w"])
        red_area = features["red_area"]
        significant = features["significant_component_count"]
        linearity = features["global_linearity"]
        endpoint_count = features["endpoint_count"]
        perp = features["perpendicular_offset"]
        angle_diff = features["top2_angle_difference"]

        # Very tiny red residue is more likely noise than a usable mark.
        if red_area < max(features["min_area"] * 1.8, 45):
            return "unknown", 0.72

        # A single SAM/color-supported stroke with two clean endpoints is a
        # continuous marking candidate. Do not call it loose only because the
        # paint blob is not perfectly straight.
        if significant <= 1 and endpoint_count <= 4:
            return "normal", min(0.86, 0.62 + max(linearity, 0.0) * 0.18)

        # Disconnected or misaligned red strokes are the main loose signal.
        if significant >= 2:
            if perp > max(10.0, 0.035 * min_dim):
                return "loose", min(0.92, 0.62 + perp / max(min_dim, 1))
            if angle_diff > math.radians(28):
                return "loose", 0.78

        if significant >= 3 and linearity < 0.82:
            return "loose", 0.72

        if endpoint_count >= 6 and linearity < 0.86:
            return "loose", 0.66

        # A coherent linear red trace is considered normal, even if the paint is
        # broken into two close aligned components.
        if linearity >= 0.78:
            return "normal", min(0.9, 0.58 + (linearity - 0.78))

        # Low-linearity red blobs are usable marks but suspicious.
        return "loose", 0.58

    def detect(self, image_bgr: np.ndarray, image_id: str = "") -> ColorDetectionResult:
        try:
            raw, cleaned = self.segment_red_mark(image_bgr)
            features = self.compute_features(cleaned, image_bgr.shape)
            prediction, confidence = self.classify(features)
            return ColorDetectionResult(
                image_id=image_id,
                prediction=prediction,
                confidence=confidence,
                red_mask=raw,
                cleaned_mask=cleaned,
                features=features,
            )
        except Exception as exc:
            return ColorDetectionResult(
                image_id=image_id,
                prediction="unknown",
                confidence=0.0,
                features={},
                error=str(exc),
            )


def visualize_result(image_bgr: np.ndarray, result: ColorDetectionResult, save_path: str) -> None:
    vis = image_bgr.copy()
    if result.cleaned_mask is not None and result.cleaned_mask.sum() > 0:
        overlay = vis.copy()
        overlay[result.cleaned_mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

        contours, _ = cv2.findContours(result.cleaned_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)

    label = f"pred={result.prediction} conf={result.confidence:.2f}"
    if result.true_label:
        label = f"true={result.true_label} " + label
    color = {"normal": (0, 220, 0), "loose": (0, 165, 255), "unknown": (180, 180, 180)}[result.prediction]
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 520), 34), (0, 0, 0), -1)
    cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.imwrite(save_path, vis)


def evaluate(
    data_dir: Path,
    labels_csv: Path,
    output_dir: Path,
    max_samples: Optional[int] = None,
    save_vis: bool = False,
    save_errors: bool = True,
) -> pd.DataFrame:
    detector = ColorMarkingDetector()
    df = pd.read_csv(labels_csv)
    if max_samples:
        df = df.head(max_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    err_dir = output_dir / "errors"
    if save_vis:
        vis_dir.mkdir(exist_ok=True)
    if save_errors:
        err_dir.mkdir(exist_ok=True)

    rows = []
    for _, row in df.iterrows():
        image_path = data_dir / row["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            rows.append({
                "id": row["id"],
                "image": row["image"],
                "true_label": row["label"],
                "pred_label": "unknown",
                "confidence": 0.0,
                "error": "failed_to_read_image",
            })
            continue

        result = detector.detect(image, image_id=str(row["id"]))
        result.true_label = row["label"]
        feature_row = result.features or {}
        out = {
            "id": row["id"],
            "image": row["image"],
            "true_label": row["label"],
            "pred_label": result.prediction,
            "confidence": result.confidence,
            "error": result.error,
        }
        for key, value in feature_row.items():
            if key == "components":
                out[key] = json.dumps(value, ensure_ascii=False)
            else:
                out[key] = value
        rows.append(out)

        wrong = row["label"] != result.prediction
        if save_vis or (save_errors and wrong):
            target_dir = vis_dir if save_vis else err_dir
            if wrong:
                target_dir = err_dir / f"{row['label']}_to_{result.prediction}"
                target_dir.mkdir(parents=True, exist_ok=True)
            save_path = target_dir / f"{row['id']}_{Path(row['image']).name}"
            visualize_result(image, result, str(save_path))

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "detailed_results.csv", index=False)

    labels = ["loose", "normal", "unknown"]
    cm = confusion_matrix(results["true_label"], results["pred_label"], labels=labels)
    report = classification_report(
        results["true_label"],
        results["pred_label"],
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float((results["true_label"] == results["pred_label"]).mean()),
        "total_samples": int(len(results)),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "loose_to_normal": int(((results["true_label"] == "loose") & (results["pred_label"] == "normal")).sum()),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    _save_confusion_matrix(cm, labels, output_dir / "confusion_matrix.png")
    return results


def _save_confusion_matrix(cm: np.ndarray, labels: List[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate color-prior bolt marking detector")
    parser.add_argument("--data-dir", type=Path, default=Path("/home/cvailab/dgy/螺栓标记测试"))
    parser.add_argument("--labels", type=Path, default=Path("/home/cvailab/dgy/螺栓标记测试/labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/cvailab/dgy/螺栓标记测试/runs/color_baseline"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--no-save-errors", action="store_true")
    args = parser.parse_args()

    results = evaluate(
        data_dir=args.data_dir,
        labels_csv=args.labels,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        save_vis=args.save_vis,
        save_errors=not args.no_save_errors,
    )
    print(f"Saved {len(results)} predictions to {args.output_dir}")
    print(results[["true_label", "pred_label"]].value_counts().sort_index())


if __name__ == "__main__":
    main()
