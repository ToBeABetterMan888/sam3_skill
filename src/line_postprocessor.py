#!/usr/bin/env python3
"""
Line-structure post-processing for segmented bolt marking masks.

Input is a binary marking-line mask, normally produced from SAM3 proposals.
Output is a geometry description that can be used for loose/normal reasoning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class SegmentGeometry:
    idx: int
    area: int
    bbox: List[int]
    centroid: List[float]
    angle: float
    linearity: float
    length: float
    width: float
    endpoint_a: List[float]
    endpoint_b: List[float]


class LinePostProcessor:
    """Convert a binary mark mask into line segments and continuity features."""

    @staticmethod
    def skeleton(mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.sum() == 0:
            return np.zeros_like(mask, dtype=np.uint8)
        return skeletonize(mask > 0).astype(np.uint8)

    @staticmethod
    def endpoints(skeleton: np.ndarray) -> List[Tuple[int, int]]:
        if skeleton is None or skeleton.sum() == 0:
            return []
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        pts = np.argwhere(neighbors == 11)
        return [(int(x), int(y)) for y, x in pts]

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = abs(a - b) % math.pi
        return float(min(diff, math.pi - diff))

    @staticmethod
    def _segment_from_points(idx: int, points_xy: np.ndarray, area: int, bbox: List[int]) -> SegmentGeometry:
        centroid = points_xy.mean(axis=0)
        if len(points_xy) < 3:
            return SegmentGeometry(
                idx=idx,
                area=area,
                bbox=bbox,
                centroid=centroid.tolist(),
                angle=0.0,
                linearity=0.0,
                length=0.0,
                width=0.0,
                endpoint_a=centroid.tolist(),
                endpoint_b=centroid.tolist(),
            )

        centered = points_xy - centroid
        cov = np.cov(centered.T)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vec = vecs[:, order[0]]
        normal = vecs[:, order[1]]
        proj = centered @ vec
        side = centered @ normal
        min_i = int(np.argmin(proj))
        max_i = int(np.argmax(proj))
        endpoint_a = points_xy[min_i]
        endpoint_b = points_xy[max_i]
        angle = math.atan2(float(vec[1]), float(vec[0]))
        linearity = float(vals[0] / (vals[0] + vals[1] + 1e-8))
        length = float(proj.max() - proj.min())
        width = float(side.max() - side.min())
        return SegmentGeometry(
            idx=idx,
            area=area,
            bbox=bbox,
            centroid=centroid.tolist(),
            angle=angle,
            linearity=linearity,
            length=length,
            width=width,
            endpoint_a=endpoint_a.tolist(),
            endpoint_b=endpoint_b.tolist(),
        )

    def analyze(self, mask: np.ndarray, image_shape: Tuple[int, int, int]) -> Dict:
        h, w = image_shape[:2]
        binary = (mask > 0).astype(np.uint8) if mask is not None else np.zeros((h, w), dtype=np.uint8)
        skel = self.skeleton(binary)
        endpoint_list = self.endpoints(skel)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        segments: List[SegmentGeometry] = []
        min_area = max(8, int(binary.sum() * 0.04))
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            ys, xs = np.where(labels == idx)
            points = np.column_stack([xs, ys]).astype(np.float32)
            bbox = [
                int(stats[idx, cv2.CC_STAT_LEFT]),
                int(stats[idx, cv2.CC_STAT_TOP]),
                int(stats[idx, cv2.CC_STAT_WIDTH]),
                int(stats[idx, cv2.CC_STAT_HEIGHT]),
            ]
            segments.append(self._segment_from_points(idx, points, area, bbox))
        segments.sort(key=lambda s: s.area, reverse=True)

        features: Dict = {
            "line_component_count": len(segments),
            "line_skeleton_length": int(skel.sum()),
            "line_endpoint_count": len(endpoint_list),
            "line_segments": [s.__dict__ for s in segments[:8]],
            "line_min_endpoint_gap": 0.0,
            "line_parallel_gap": 0.0,
            "line_perpendicular_offset": 0.0,
            "line_angle_difference": 0.0,
            "local_angle_difference": 0.0,
            "local_perpendicular_offset": 0.0,
            "local_bridge_width_ratio": 1.0,
            "local_direction_status": "not_evaluated",
            "line_status": "missing" if binary.sum() == 0 else "single",
        }

        if len(segments) >= 2:
            s1, s2 = segments[0], segments[1]
            endpoints1 = [np.array(s1.endpoint_a), np.array(s1.endpoint_b)]
            endpoints2 = [np.array(s2.endpoint_a), np.array(s2.endpoint_b)]
            best_gap = float("inf")
            best_pair = None
            for p1 in endpoints1:
                for p2 in endpoints2:
                    gap = float(np.linalg.norm(p1 - p2))
                    if gap < best_gap:
                        best_gap = gap
                        best_pair = (p1, p2)

            mean_angle = 0.5 * (s1.angle + s2.angle)
            axis = np.array([math.cos(mean_angle), math.sin(mean_angle)], dtype=np.float32)
            normal = np.array([-axis[1], axis[0]], dtype=np.float32)
            delta = np.array(s2.centroid, dtype=np.float32) - np.array(s1.centroid, dtype=np.float32)
            features["line_min_endpoint_gap"] = best_gap
            features["line_parallel_gap"] = float(abs(np.dot(delta, axis)))
            features["line_perpendicular_offset"] = float(abs(np.dot(delta, normal)))
            features["line_angle_difference"] = self._angle_diff(s1.angle, s2.angle)
            features["line_nearest_endpoint_pair"] = [
                best_pair[0].tolist(),
                best_pair[1].tolist(),
            ] if best_pair else []

            min_dim = min(h, w)
            # If two pieces are nearly collinear and the break is modest, this
            # is usually a normal paint interruption at an edge/shadow rather
            # than a loosened offset.
            if (
                features["line_angle_difference"] < math.radians(8)
                and features["line_min_endpoint_gap"] < max(35.0, 0.08 * min_dim)
                and features["line_perpendicular_offset"] < max(35.0, 0.08 * min_dim)
            ):
                features["line_status"] = "multi_aligned"
            elif features["line_angle_difference"] > math.radians(24):
                features["line_status"] = "misaligned_angle"
            elif features["line_perpendicular_offset"] > max(8.0, 0.025 * min_dim):
                features["line_status"] = "misaligned_offset"
            elif features["line_min_endpoint_gap"] > max(14.0, 0.035 * min_dim):
                features["line_status"] = "broken_gap"
            else:
                features["line_status"] = "multi_aligned"
        elif len(segments) == 1:
            seg = segments[0]
            local = self._analyze_local_halves(binary, seg)
            features.update(local)
            # A single very linear paint stroke can have extra skeleton
            # endpoints from ragged paint boundaries. Treat that as continuous
            # unless the fragmentation is severe.
            if local["local_direction_status"] == "bridge_break":
                features["line_status"] = local["local_direction_status"]
            elif local["local_direction_status"] == "direction_mismatch":
                features["line_status"] = "curved_continuous"
            elif seg.linearity >= 0.78 and len(endpoint_list) <= 10:
                features["line_status"] = "single_continuous"
            elif len(endpoint_list) > 4:
                features["line_status"] = "branched_or_fragmented"
            elif seg.linearity >= 0.72:
                features["line_status"] = "single_continuous"
            else:
                features["line_status"] = "single_irregular"

        return features

    def _analyze_local_halves(self, mask: np.ndarray, segment: SegmentGeometry) -> Dict:
        """Compare local directions on both sides of a connected mark."""
        features = {
            "local_angle_difference": 0.0,
            "local_perpendicular_offset": 0.0,
            "local_bridge_width_ratio": 1.0,
            "local_direction_status": "single_segment",
        }

        ys, xs = np.where(mask > 0)
        if len(xs) < 20:
            features["local_direction_status"] = "too_small"
            return features

        points = np.column_stack([xs, ys]).astype(np.float32)
        angle = float(segment.angle)
        axis = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        normal = np.array([-axis[1], axis[0]], dtype=np.float32)
        center = np.array(segment.centroid, dtype=np.float32)
        rel = points - center
        t = rel @ axis
        n = rel @ normal

        t_min, t_max = float(t.min()), float(t.max())
        length = max(t_max - t_min, 1.0)
        low_cut = t_min + 0.42 * length
        high_cut = t_min + 0.58 * length
        side_a = points[t <= low_cut]
        side_b = points[t >= high_cut]
        bridge_n = n[(t > low_cut) & (t < high_cut)]

        if len(side_a) < 10 or len(side_b) < 10:
            features["local_direction_status"] = "not_enough_halves"
            return features

        geom_a = self._segment_from_points(1, side_a, len(side_a), [0, 0, 0, 0])
        geom_b = self._segment_from_points(2, side_b, len(side_b), [0, 0, 0, 0])
        angle_diff = self._angle_diff(geom_a.angle, geom_b.angle)

        ca = np.array(geom_a.centroid, dtype=np.float32)
        cb = np.array(geom_b.centroid, dtype=np.float32)
        local_offset = float(abs(np.dot(cb - ca, normal)))

        side_width_a = max(float(geom_a.width), 1.0)
        side_width_b = max(float(geom_b.width), 1.0)
        side_width = max(1.0, 0.5 * (side_width_a + side_width_b))
        if len(bridge_n) >= 3:
            bridge_width = float(np.percentile(bridge_n, 95) - np.percentile(bridge_n, 5))
        else:
            bridge_width = 0.0
        bridge_ratio = bridge_width / side_width

        features.update({
            "local_angle_difference": float(angle_diff),
            "local_perpendicular_offset": float(local_offset),
            "local_bridge_width_ratio": float(bridge_ratio),
            "local_upper_angle": float(geom_a.angle),
            "local_lower_angle": float(geom_b.angle),
        })

        if bridge_ratio < 0.32 and (angle_diff > math.radians(10) or local_offset > 10.0):
            features["local_direction_status"] = "bridge_break"
        elif angle_diff > math.radians(18):
            features["local_direction_status"] = "direction_mismatch"
        elif local_offset > max(10.0, 0.06 * max(segment.length, 1.0)) and angle_diff > math.radians(8):
            features["local_direction_status"] = "direction_mismatch"
        else:
            features["local_direction_status"] = "halves_aligned"

        return features

    @staticmethod
    def draw_overlay(image_bgr: np.ndarray, mask: np.ndarray, features: Dict) -> np.ndarray:
        vis = image_bgr.copy()
        if mask is None or mask.sum() == 0:
            return vis

        skel = skeletonize(mask > 0).astype(np.uint8)
        vis[skel > 0] = (0, 255, 0)

        for seg in features.get("line_segments", []):
            pa = tuple(int(round(v)) for v in seg["endpoint_a"])
            pb = tuple(int(round(v)) for v in seg["endpoint_b"])
            cx, cy = (int(round(seg["centroid"][0])), int(round(seg["centroid"][1])))
            cv2.circle(vis, pa, 5, (255, 0, 255), -1)
            cv2.circle(vis, pb, 5, (255, 255, 0), -1)
            cv2.line(vis, pa, pb, (0, 255, 255), 2)
            cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)

        pair = features.get("line_nearest_endpoint_pair") or []
        if len(pair) == 2:
            p1 = tuple(int(round(v)) for v in pair[0])
            p2 = tuple(int(round(v)) for v in pair[1])
            cv2.line(vis, p1, p2, (255, 0, 0), 2)

        return vis
