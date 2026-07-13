#!/usr/bin/env python3
"""
SAM3-first bolt marking detector.

SAM3 provides the marking-line segmentation proposals. Color and geometry are
only used to choose/refine those SAM3 proposals and to classify the final mask.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(os.environ.get("BOLT_MARKING_ROOT", Path(__file__).resolve().parent.parent))
DEFAULT_SAM3_ROOT = Path(os.environ.get("SAM3_ROOT", "/home/cvailab/zhaoza/sam3"))
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "sam3.pt"
DEFAULT_FEATURE_MODEL = PROJECT_ROOT / "models" / "feature_judger" / "random_forest_final.joblib"
if str(DEFAULT_SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_SAM3_ROOT))
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    build_sam3_image_model = None
    Sam3Processor = None

from color_marking_detector import ColorMarkingDetector
from line_postprocessor import LinePostProcessor
from loosening_judger import LooseningJudger
from sam3_interface_analyzer import Sam3InterfaceAnalyzer


Label = Literal["normal", "loose", "unknown"]


@dataclass
class Sam3Candidate:
    prompt: str
    sam_score: float
    mask: np.ndarray
    area: int
    area_ratio: float
    red_precision: float
    red_recall: float
    line_score: float
    center_score: float
    touches_border: bool
    select_score: float


@dataclass
class Sam3DetectionResult:
    image_id: str
    prediction: Label
    confidence: float
    sam_mask: Optional[np.ndarray] = None
    refined_mask: Optional[np.ndarray] = None
    red_hint_mask: Optional[np.ndarray] = None
    features: Optional[Dict] = None
    candidates: Optional[List[Dict]] = None
    true_label: Optional[str] = None
    error: Optional[str] = None


class Sam3MarkingDetector:
    """Use SAM3 text segmentation as the primary marking-line mask source."""

    def __init__(
        self,
        checkpoint_path: str = str(DEFAULT_CHECKPOINT),
        device: str = "cuda",
        min_sam_score: float = 0.20,
        use_interface_rule: bool = False,
    ):
        self.device = device
        self.min_sam_score = min_sam_score
        self.use_interface_rule = use_interface_rule
        self.color_helper = ColorMarkingDetector(min_area_px=18, min_component_area=8)
        self.line_processor = LinePostProcessor()
        self.loosening_judger = LooseningJudger()

        if build_sam3_image_model is None or Sam3Processor is None:
            raise ImportError(
                "Cannot import SAM3. Set SAM3_ROOT to the SAM3 code directory, "
                "for example: export SAM3_ROOT=/home/cvailab/zhaoza/sam3"
            )
        bpe_path = os.environ.get(
            "SAM3_BPE",
            str(DEFAULT_SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
        )
        print(f"Loading SAM3 model from {checkpoint_path}")
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
            device=device,
            load_from_HF=False,
        )
        self.processor = Sam3Processor(self.model, device=device)
        self.interface_analyzer = Sam3InterfaceAnalyzer(self) if use_interface_rule else None
        print("SAM3 model loaded")

    def _autocast(self):
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)

    @staticmethod
    def _to_2d_mask(mask) -> np.ndarray:
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().float().numpy()
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"Unexpected SAM3 mask shape: {mask.shape}")
        return (mask > 0).astype(np.uint8)

    @staticmethod
    def _mask_line_score(mask: np.ndarray) -> float:
        area = int(mask.sum())
        if area <= 0:
            return 0.0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if n <= 1:
            return 0.0
        scores = []
        for idx in range(1, n):
            comp_area = float(stats[idx, cv2.CC_STAT_AREA])
            if comp_area < 8:
                continue
            w = float(stats[idx, cv2.CC_STAT_WIDTH])
            h = float(stats[idx, cv2.CC_STAT_HEIGHT])
            long_side = max(w, h)
            short_side = max(1.0, min(w, h))
            elongation = min(1.0, long_side / (short_side * 6.0))
            fill = comp_area / max(w * h, 1.0)
            thinness = max(0.0, 1.0 - fill)
            scores.append(elongation * 0.65 + thinness * 0.35)
        return float(max(scores) if scores else 0.0)

    def _candidate_score(
        self,
        sam_score: float,
        mask: np.ndarray,
        red_hint: np.ndarray,
        image_area: int,
    ) -> Tuple[float, Dict]:
        h, w = mask.shape[:2]
        area = int(mask.sum())
        area_ratio = area / max(image_area, 1)
        red_area = int(red_hint.sum())
        overlap = int((mask & red_hint).sum())
        red_precision = overlap / max(area, 1)
        red_recall = overlap / max(red_area, 1)
        line_score = self._mask_line_score(mask)

        ys, xs = np.where(mask > 0)
        if len(xs):
            cx = float(xs.mean())
            cy = float(ys.mean())
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
        else:
            cx = cy = 0.0
            x0 = y0 = x1 = y1 = 0
        center_dist = float(np.hypot((cx - w / 2) / max(w, 1), (cy - h / 2) / max(h, 1)))
        center_score = max(0.0, 1.0 - center_dist / 0.55)

        margin = max(4, int(0.025 * min(h, w)))
        touches_border = x0 <= margin or y0 <= margin or x1 >= w - margin or y1 >= h - margin

        # Prefer SAM masks that are confident, line-like, red-overlapping, and
        # not the whole bolt/background. These are ranking terms, not the
        # segmentation source.
        size_penalty = 0.0
        if area_ratio > 0.18:
            size_penalty = min(0.5, (area_ratio - 0.18) * 2.0)
        if area_ratio < 0.0002:
            size_penalty += 0.25
        edge_penalty = 0.22 if touches_border and center_score < 0.65 else 0.0

        select_score = (
            0.32 * sam_score
            + 0.24 * red_precision
            + 0.16 * min(red_recall, 1.0)
            + 0.16 * line_score
            + 0.12 * center_score
            - size_penalty
            - edge_penalty
        )
        return float(select_score), {
            "area": area,
            "area_ratio": float(area_ratio),
            "red_precision": float(red_precision),
            "red_recall": float(red_recall),
            "line_score": float(line_score),
            "center_score": float(center_score),
            "touches_border": bool(touches_border),
        }

    def _get_sam_candidates(self, image_bgr: np.ndarray, red_hint: np.ndarray) -> List[Sam3Candidate]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        h, w = image_bgr.shape[:2]
        image_area = h * w

        prompts = [
            "complete red anti loosening marking line on the bolt",
            "all red anti loosening paint marks",
            "all red paint marking lines on the bolt and washer",
            "red torque seal mark across the bolt and base",
            "red witness mark line across the fastener",
            "complete red witness paint mark",
            "red paint",
            "red mark",
            "red line",
            "thin red line",
            "painted red line",
            "anti loosening mark",
            "paint mark on bolt",
            "colored marking line",
        ]

        with self._autocast():
            state = self.processor.set_image(pil_image)

        candidates: List[Sam3Candidate] = []
        seen = set()
        for prompt in prompts:
            try:
                with self._autocast():
                    output = self.processor.set_text_prompt(state=state, prompt=prompt)
                masks = output.get("masks", [])
                scores = output.get("scores", [])
                for idx in range(len(masks)):
                    score = float(scores[idx].item()) if torch.is_tensor(scores[idx]) else float(scores[idx])
                    if score < self.min_sam_score:
                        continue
                    mask = self._to_2d_mask(masks[idx])
                    key = cv2.resize(mask, (32, 32), interpolation=cv2.INTER_NEAREST).tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    select_score, stats = self._candidate_score(score, mask, red_hint, image_area)
                    candidates.append(Sam3Candidate(
                        prompt=prompt,
                        sam_score=score,
                        mask=mask,
                        select_score=select_score,
                        **stats,
                    ))
            except Exception as exc:
                print(f"SAM3 prompt failed: {prompt}: {exc}")

        candidates.sort(key=lambda c: c.select_score, reverse=True)
        return candidates

    @staticmethod
    def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = int((mask_a & mask_b).sum())
        union = int((mask_a | mask_b).sum())
        return inter / max(union, 1)

    def _merge_marking_candidates(self, candidates: List[Sam3Candidate]) -> np.ndarray:
        """Merge plausible SAM3 marking instances into one full marking mask."""
        if not candidates:
            raise ValueError("No candidates to merge")

        merged = np.zeros_like(candidates[0].mask, dtype=np.uint8)
        kept = 0
        best_score = candidates[0].select_score
        for candidate in candidates:
            plausible = (
                candidate.sam_score >= 0.45
                and candidate.red_precision >= 0.45
                and candidate.area_ratio <= 0.16
                and candidate.select_score >= max(0.38, best_score - 0.18)
            )
            if candidate.touches_border and candidate.center_score < 0.35:
                plausible = False
            if not plausible:
                continue
            if kept > 0 and self._iou(merged, candidate.mask) > 0.88:
                continue
            merged = (merged | candidate.mask).astype(np.uint8)
            kept += 1
            if kept >= 4:
                break

        if kept == 0:
            merged = candidates[0].mask.astype(np.uint8)
        return merged

    def _refine_sam_mask(
        self,
        sam_mask: np.ndarray,
        red_hint: np.ndarray,
        preserve_topology: bool = False,
    ) -> np.ndarray:
        """Keep red-mark pixels inside/near the selected SAM3 proposal."""
        if sam_mask is None or sam_mask.sum() == 0:
            return np.zeros_like(red_hint)

        near_sam = cv2.dilate(
            sam_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        refined = (red_hint & near_sam).astype(np.uint8)

        # If color is too weak but SAM3 produced a small line-like mask, keep the
        # SAM3 mask. This preserves SAM3 authority on faded/overexposed marks.
        if refined.sum() < 0.12 * max(int(sam_mask.sum()), 1) and sam_mask.sum() < 0.12 * sam_mask.size:
            refined = sam_mask.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        if not preserve_topology:
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        return self.color_helper._filter_components(refined)

    def detect(self, image_bgr: np.ndarray, image_id: str = "") -> Sam3DetectionResult:
        try:
            red_raw, red_clean = self.color_helper.segment_red_mark(image_bgr)
            candidates = self._get_sam_candidates(image_bgr, red_clean)
            if not candidates:
                features = self.color_helper.compute_features(np.zeros_like(red_clean), image_bgr.shape)
                return Sam3DetectionResult(
                    image_id=image_id,
                    prediction="unknown",
                    confidence=0.82,
                    red_hint_mask=red_clean,
                    refined_mask=np.zeros_like(red_clean),
                    features={**features, "sam_candidate_count": 0},
                    candidates=[],
                    error="no_sam_marking_candidate",
                )

            best = candidates[0]
            sam_marking_mask = self._merge_marking_candidates(candidates)
            # Use the raw red support for topology-sensitive judgment so small
            # gaps at the bolt/base interface are not closed by color cleanup.
            refined = self._refine_sam_mask(sam_marking_mask, red_raw, preserve_topology=True)
            features = self.color_helper.compute_features(refined, image_bgr.shape)
            line_features = self.line_processor.analyze(refined, image_bgr.shape)
            features.update({
                "sam_candidate_count": len(candidates),
                "best_prompt": best.prompt,
                "best_sam_score": best.sam_score,
                "best_select_score": best.select_score,
                "best_sam_area": best.area,
                "best_sam_area_ratio": best.area_ratio,
                "best_sam_red_precision": best.red_precision,
                "best_sam_red_recall": best.red_recall,
                "best_sam_line_score": best.line_score,
                "best_sam_center_score": getattr(best, "center_score", 0.0),
                "best_sam_touches_border": getattr(best, "touches_border", False),
                "merged_sam_area": int(sam_marking_mask.sum()),
                "refined_area": int(refined.sum()),
            })
            features.update(line_features)
            if self.interface_analyzer is not None:
                # Interface reasoning needs the SAM3 marking mask. The refined
                # color-supported mask can close or reshape exactly the small
                # interface break/direction change we want to measure.
                interface_features = self.interface_analyzer.analyze(image_bgr, sam_marking_mask)
                features.update(interface_features)
            judgment = self.loosening_judger.judge(features)
            prediction, confidence = judgment.label, judgment.confidence
            features.update({
                "judgment_reason_code": judgment.reason_code,
                "judgment_reason": judgment.reason,
            })

            return Sam3DetectionResult(
                image_id=image_id,
                prediction=prediction,
                confidence=confidence,
                sam_mask=sam_marking_mask,
                refined_mask=refined,
                red_hint_mask=red_clean,
                features=features,
                candidates=[self._candidate_to_dict(c) for c in candidates[:8]],
            )
        except Exception as exc:
            return Sam3DetectionResult(
                image_id=image_id,
                prediction="unknown",
                confidence=0.0,
                features={},
                candidates=[],
                error=str(exc),
            )

    @staticmethod
    def _candidate_to_dict(candidate: Sam3Candidate) -> Dict:
        return {
            "prompt": candidate.prompt,
            "sam_score": candidate.sam_score,
            "area": candidate.area,
            "area_ratio": candidate.area_ratio,
            "red_precision": candidate.red_precision,
            "red_recall": candidate.red_recall,
            "line_score": candidate.line_score,
            "center_score": getattr(candidate, "center_score", 0.0),
            "touches_border": getattr(candidate, "touches_border", False),
            "select_score": candidate.select_score,
        }


def visualize_result(image_bgr: np.ndarray, result: Sam3DetectionResult, save_path: str) -> None:
    original = image_bgr.copy()
    vis = image_bgr.copy()

    if result.sam_mask is not None and result.sam_mask.sum() > 0:
        sam_overlay = vis.copy()
        sam_overlay[result.sam_mask > 0] = (255, 180, 0)
        vis = cv2.addWeighted(sam_overlay, 0.25, vis, 0.75, 0)
        contours, _ = cv2.findContours(result.sam_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 1)

    if result.refined_mask is not None and result.refined_mask.sum() > 0:
        refined_overlay = vis.copy()
        refined_overlay[result.refined_mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(refined_overlay, 0.55, vis, 0.45, 0)
        contours, _ = cv2.findContours(result.refined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)
        if result.features:
            vis = LinePostProcessor.draw_overlay(vis, result.refined_mask.astype(np.uint8), result.features)

    label = f"pred={result.prediction} conf={result.confidence:.2f}"
    if result.true_label:
        label = f"true={result.true_label} " + label
    prompt = ""
    if result.features and result.features.get("best_prompt"):
        status = result.features.get("line_status", "")
        reason = result.features.get("judgment_reason_code", "")
        prompt = f" prompt={result.features['best_prompt']} score={result.features['best_sam_score']:.2f} line={status} reason={reason}"

    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 760), 38), (0, 0, 0), -1)
    cv2.putText(vis, label + prompt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(original, (0, 0), (min(original.shape[1], 220), 34), (0, 0, 0), -1)
    cv2.putText(original, "original", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

    divider = np.full((original.shape[0], 8, 3), 245, dtype=np.uint8)
    paired = np.concatenate([original, divider, vis], axis=1)
    cv2.imwrite(save_path, paired)


def evaluate(
    data_dir: Path,
    labels_csv: Path,
    output_dir: Path,
    checkpoint_path: str,
    device: str,
    max_samples: Optional[int] = None,
    save_vis: bool = False,
    save_errors: bool = True,
    use_interface_rule: bool = False,
    feature_model_path: Optional[Path] = DEFAULT_FEATURE_MODEL,
) -> pd.DataFrame:
    detector = Sam3MarkingDetector(
        checkpoint_path=checkpoint_path,
        device=device,
        use_interface_rule=use_interface_rule,
    )
    feature_model = None
    if feature_model_path and feature_model_path.exists():
        import joblib
        feature_model = joblib.load(feature_model_path)
        print(f"Loaded final feature judger: {feature_model_path}")
    elif feature_model_path:
        print(f"Final feature judger not found, using rule labels only: {feature_model_path}")

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
    for index, row in df.iterrows():
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
        features = result.features or {}
        out = {
            "id": row["id"],
            "image": row["image"],
            "true_label": row["label"],
            "rule_pred_label": result.prediction,
            "rule_confidence": result.confidence,
            "pred_label": result.prediction,
            "confidence": result.confidence,
            "error": result.error,
            "candidates": json.dumps(result.candidates or [], ensure_ascii=False),
        }
        for key, value in features.items():
            if key == "components":
                out[key] = json.dumps(value, ensure_ascii=False)
            else:
                out[key] = value

        if feature_model is not None:
            model = feature_model["model"]
            numeric_cols = feature_model["numeric_cols"]
            categorical_cols = feature_model["categorical_cols"]
            x = pd.DataFrame([out]).reindex(columns=numeric_cols + categorical_cols)
            pred_label = str(model.predict(x)[0])
            out["pred_label"] = pred_label
            out["final_model"] = "random_forest_feature_judger"
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]
                classes = list(model.named_steps["model"].classes_)
                out["confidence"] = float(max(probs))
                for cls, prob in zip(classes, probs):
                    out[f"prob_{cls}"] = float(prob)
            result.prediction = pred_label
            result.confidence = float(out["confidence"])

        rows.append(out)

        wrong = row["label"] != result.prediction
        if save_vis or (save_errors and wrong):
            target_dir = vis_dir if save_vis else err_dir
            if wrong:
                target_dir = err_dir / f"{row['label']}_to_{result.prediction}"
                target_dir.mkdir(parents=True, exist_ok=True)
            save_path = target_dir / f"{row['id']}_{Path(row['image']).name}"
            visualize_result(image, result, str(save_path))

        if (index + 1) % 20 == 0:
            print(f"Processed {index + 1}/{len(df)}")

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
        "final_model": str(feature_model_path) if feature_model is not None else "rule_only",
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
    ax.set_title("SAM3 Marking Detector Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAM3-first bolt marking detector")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "data" / "labels.csv")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "sam3_marking")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--no-save-errors", action="store_true")
    parser.add_argument("--use-interface-rule", action="store_true")
    parser.add_argument("--feature-model", type=Path, default=DEFAULT_FEATURE_MODEL)
    parser.add_argument("--rule-only", action="store_true")
    args = parser.parse_args()

    results = evaluate(
        data_dir=args.data_dir,
        labels_csv=args.labels,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_samples=args.max_samples,
        save_vis=args.save_vis,
        save_errors=not args.no_save_errors,
        use_interface_rule=args.use_interface_rule,
        feature_model_path=None if args.rule_only else args.feature_model,
    )
    print(f"Saved {len(results)} predictions to {args.output_dir}")
    print(pd.crosstab(results["true_label"], results["pred_label"]))


if __name__ == "__main__":
    main()
