#!/usr/bin/env python3
"""SAM3 object-interface analyzer for bolt loosening evidence.

This module is intentionally a second-stage rule. It uses the already-loaded
SAM3 detector to segment bolt/nut/washer-like objects, converts their masks to
smooth outer contours, and compares the red-mark skeleton around the object
interface.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.morphology import skeletonize


@dataclass
class InterfaceMaskCandidate:
    prompt: str
    score: float
    mask: np.ndarray
    area: int
    area_ratio: float


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def pca(points: np.ndarray) -> Optional[Dict]:
    if len(points) < 6:
        return None
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    axis = vecs[:, order[0]].astype(np.float32)
    proj = centered @ axis
    return {
        "center": center,
        "axis": axis,
        "angle": math.atan2(float(axis[1]), float(axis[0])),
        "linearity": float(vals[0] / (vals.sum() + 1e-8)),
        "length": float(proj.max() - proj.min()),
    }


def orient_axis_toward(axis: np.ndarray, center: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    axis = axis.astype(np.float32)
    if float((center - anchor) @ axis) < 0:
        axis = -axis
    return axis


class Sam3InterfaceAnalyzer:
    """Detect loose evidence at the bolt/nut/washer interface."""

    OBJECT_PROMPTS = [
        "hex nut",
        "nut",
        "hex bolt head",
        "bolt head",
        "washer",
        "circular washer",
        "round metal fastener",
        "metal circular fastener",
        "washer around bolt",
    ]

    def __init__(self, detector, min_object_score: float = 0.22):
        self.detector = detector
        self.min_object_score = min_object_score

    def _autocast(self):
        device_type = "cuda" if str(self.detector.device).startswith("cuda") else "cpu"
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)

    @staticmethod
    def _to_mask(mask) -> np.ndarray:
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().float().numpy()
        mask = np.squeeze(mask)
        return (mask > 0).astype(np.uint8)

    @staticmethod
    def boundary(mask: np.ndarray, width: int = 7) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1))
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        return (dilated - eroded).astype(np.uint8)

    @staticmethod
    def fill_object(mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid = np.zeros_like(mask)
        contours = [c for c in contours if cv2.contourArea(c) >= 20]
        if contours:
            cv2.drawContours(solid, contours, -1, 1, -1)
        return solid.astype(np.uint8)

    @staticmethod
    def outer_object_mask(mask: np.ndarray) -> np.ndarray:
        mask = Sam3InterfaceAnalyzer.fill_object(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        outer = np.zeros_like(mask)
        cv2.drawContours(outer, [hull], -1, 1, -1)
        return outer.astype(np.uint8)

    @staticmethod
    def skeleton(mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.sum() == 0:
            return np.zeros_like(mask, dtype=np.uint8)
        clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
        return skeletonize(clean > 0).astype(np.uint8)

    @staticmethod
    def boundary_intersection_center(skel: np.ndarray, band: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        iy, ix = np.where((skel & band) > 0)
        if len(ix) == 0:
            return fallback
        pts = np.column_stack([ix, iy]).astype(np.float32)
        d = np.linalg.norm(pts - fallback[None, :], axis=1)
        return pts[int(np.argmin(d))]

    @staticmethod
    def boundary_frame(boundary: np.ndarray, center: np.ndarray, radius: float = 45.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """Estimate interface tangent and normal from the SAM3 object boundary."""
        ys, xs = np.where(boundary > 0)
        if len(xs) < 6:
            tangent = np.array([1.0, 0.0], dtype=np.float32)
            normal = np.array([0.0, 1.0], dtype=np.float32)
            return tangent, normal, 0.0
        pts = np.column_stack([xs, ys]).astype(np.float32)
        dist = np.linalg.norm(pts - center[None, :], axis=1)
        local = pts[dist <= radius]
        if len(local) < 6:
            local = pts[np.argsort(dist)[: min(24, len(pts))]]
        geo = pca(local)
        if geo is None:
            tangent = np.array([1.0, 0.0], dtype=np.float32)
            normal = np.array([0.0, 1.0], dtype=np.float32)
            return tangent, normal, 0.0
        tangent = geo["axis"] / (np.linalg.norm(geo["axis"]) + 1e-8)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        return tangent.astype(np.float32), normal.astype(np.float32), float(geo["linearity"])

    @staticmethod
    def skeleton_sides_by_graph(skel: np.ndarray, filled: np.ndarray, anchor: np.ndarray, min_steps: int = 3, max_steps: int = 34) -> Tuple[np.ndarray, np.ndarray]:
        ys, xs = np.where(skel > 0)
        empty = np.zeros((0, 2), dtype=np.float32)
        if len(xs) == 0:
            return empty, empty
        pts = np.column_stack([xs, ys]).astype(np.float32)
        start_idx = int(np.argmin(np.linalg.norm(pts - anchor[None, :], axis=1)))
        sx, sy = int(pts[start_idx, 0]), int(pts[start_idx, 1])
        h, w = skel.shape
        seen = np.zeros_like(skel, dtype=np.uint8)
        q = deque([(sx, sy, 0)])
        seen[sy, sx] = 1
        side_a: List[List[float]] = []
        side_b: List[List[float]] = []
        neigh = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        while q:
            x, y, d = q.popleft()
            if min_steps <= d <= max_steps:
                if filled[y, x] > 0:
                    side_a.append([float(x), float(y)])
                else:
                    side_b.append([float(x), float(y)])
            if d >= max_steps:
                continue
            for dx, dy in neigh:
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                if seen[ny, nx] or skel[ny, nx] == 0:
                    continue
                seen[ny, nx] = 1
                q.append((nx, ny, d + 1))
        return np.asarray(side_a, dtype=np.float32), np.asarray(side_b, dtype=np.float32)

    def _segment_object_prompts(self, image_bgr: np.ndarray) -> List[InterfaceMaskCandidate]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image_rgb)
        h, w = image_bgr.shape[:2]
        with self._autocast():
            state = self.detector.processor.set_image(pil)
        out: List[InterfaceMaskCandidate] = []
        seen = set()
        for prompt in self.OBJECT_PROMPTS:
            try:
                with self._autocast():
                    result = self.detector.processor.set_text_prompt(state=state, prompt=prompt)
                masks = result.get("masks", [])
                scores = result.get("scores", [])
                for i in range(len(masks)):
                    score = float(scores[i].item()) if torch.is_tensor(scores[i]) else float(scores[i])
                    if score < self.min_object_score:
                        continue
                    mask = self._to_mask(masks[i])
                    key = cv2.resize(mask, (32, 32), interpolation=cv2.INTER_NEAREST).tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    area = int(mask.sum())
                    out.append(InterfaceMaskCandidate(prompt, score, mask, area, area / max(h * w, 1)))
            except Exception as exc:
                print(f"SAM3 object prompt failed: {prompt}: {exc}")
        out.sort(key=lambda c: c.score, reverse=True)
        return out

    def choose_interface_object(self, image_bgr: np.ndarray, mark: np.ndarray) -> Optional[Dict]:
        candidates = self._segment_object_prompts(image_bgr)
        mark_area = max(int(mark.sum()), 1)
        best = None
        for cand in candidates:
            if cand.area_ratio < 0.015 or cand.area_ratio > 0.85:
                continue
            filled = self.outer_object_mask(cand.mask)
            bnd = self.boundary(filled, width=6)
            band = cv2.dilate(bnd, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1)
            overlap = int((band & mark).sum())
            inside = int((filled & mark).sum())
            outside = int(((1 - filled) & mark).sum())
            both_sides = min(inside, outside) / mark_area
            overlap_ratio = overlap / mark_area
            score = cand.score + 2.8 * overlap_ratio + 1.8 * both_sides
            if overlap < 5:
                score -= 0.5
            item = {
                "candidate": cand,
                "filled_mask": filled,
                "boundary": bnd,
                "band": band,
                "overlap": overlap,
                "overlap_ratio": overlap_ratio,
                "inside_mark": inside,
                "outside_mark": outside,
                "both_sides": both_sides,
                "score": score,
            }
            if best is None or item["score"] > best["score"]:
                best = item
        return best

    def analyze(self, image_bgr: np.ndarray, mark: np.ndarray) -> Dict:
        result = {
            "interface_loose_evidence": False,
            "interface_reason": "no_interface_object",
            "interface_object_prompt": "",
            "interface_object_score": 0.0,
            "interface_overlap": 0,
            "interface_both_sides": 0.0,
            "interface_has_two_sided_skeleton": False,
            "interface_break": False,
            "interface_angle_diff_deg": 0.0,
            "interface_tangent_angle_diff_deg": 0.0,
            "interface_normal_angle_diff_deg": 0.0,
            "interface_boundary_linearity": 0.0,
            "interface_offset_along_boundary_tangent": 0.0,
            "interface_offset_along_boundary_normal": 0.0,
            "interface_red_normal_offset": 0.0,
            "interface_side_a_pixels": 0,
            "interface_side_b_pixels": 0,
        }
        if mark is None or int(mark.sum()) == 0:
            result["interface_reason"] = "missing_mark"
            return result
        obj = self.choose_interface_object(image_bgr, mark.astype(np.uint8))
        if obj is None:
            return result

        cand = obj["candidate"]
        result.update({
            "interface_object_prompt": cand.prompt,
            "interface_object_score": cand.score,
            "interface_overlap": obj["overlap"],
            "interface_both_sides": obj["both_sides"],
        })
        band = obj["band"]
        interface_region = (band & mark).astype(np.uint8)
        iy, ix = np.where(interface_region > 0)
        if len(ix) == 0:
            result["interface_reason"] = "no_mark_boundary_overlap"
            return result

        fallback_center = np.array([ix.mean(), iy.mean()], dtype=np.float32)
        filled = obj["filled_mask"]
        skel = self.skeleton(mark)
        center = self.boundary_intersection_center(skel, band, fallback_center)
        boundary_tangent, boundary_normal, boundary_linearity = self.boundary_frame(obj["boundary"], center)
        result["interface_boundary_linearity"] = boundary_linearity
        side_a, side_b = self.skeleton_sides_by_graph(skel, filled, center)
        result["interface_side_a_pixels"] = int(len(side_a))
        result["interface_side_b_pixels"] = int(len(side_b))

        ga = pca(side_a)
        gb = pca(side_b)
        if ga is not None and gb is not None:
            ga["axis"] = orient_axis_toward(ga["axis"], ga["center"], center)
            gb["axis"] = orient_axis_toward(gb["axis"], gb["center"], center)
            ga["angle"] = math.atan2(float(ga["axis"][1]), float(ga["axis"][0]))
            gb["angle"] = math.atan2(float(gb["axis"][1]), float(gb["axis"][0]))
            angle = math.degrees(angle_diff(ga["angle"], gb["angle"]))
            red_mean_axis = ga["axis"] + gb["axis"]
            if float(np.linalg.norm(red_mean_axis)) < 1e-5:
                red_mean_axis = ga["axis"]
            red_mean_axis = red_mean_axis / (np.linalg.norm(red_mean_axis) + 1e-8)
            red_normal = np.array([-red_mean_axis[1], red_mean_axis[0]], dtype=np.float32)
            delta = ga["center"] - gb["center"]
            result["interface_has_two_sided_skeleton"] = True
            result["interface_angle_diff_deg"] = angle
            # Normal angle equals tangent angle for unoriented lines, but it is
            # emitted explicitly because it is the geometric language we use
            # when reasoning about offset around the interface.
            result["interface_tangent_angle_diff_deg"] = angle
            result["interface_normal_angle_diff_deg"] = angle
            result["interface_offset_along_boundary_tangent"] = abs(float(delta @ boundary_tangent))
            result["interface_offset_along_boundary_normal"] = abs(float(delta @ boundary_normal))
            result["interface_red_normal_offset"] = abs(float(delta @ red_normal))
            # Conservative gate found on the v2 hard-error set: keeps 428 while
            # suppressing most normal continuous-paint false positives.
            if angle >= 30.0 and obj["both_sides"] >= 0.35:
                result["interface_loose_evidence"] = True
                result["interface_reason"] = "skeleton_direction_mismatch"
            else:
                result["interface_reason"] = "skeleton_direction_aligned"
            return result

        one_side_missing = (len(side_a) >= 12 and len(side_b) == 0) or (len(side_b) >= 12 and len(side_a) == 0)
        if one_side_missing and obj["both_sides"] >= 0.35 and obj["overlap"] >= 8:
            result["interface_loose_evidence"] = True
            result["interface_break"] = True
            result["interface_reason"] = "skeleton_interface_break"
        else:
            result["interface_reason"] = "insufficient_two_sided_skeleton"
        return result
