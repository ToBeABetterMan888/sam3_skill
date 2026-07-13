#!/usr/bin/env python3
"""
Rule-based loosening judgment from SAM3 marking-line geometry.

The detector owns segmentation. The line postprocessor owns geometry. This
module owns the final anti-loosening decision and makes the reason explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal


Label = Literal["normal", "loose", "unknown"]


@dataclass
class LooseningJudgment:
    label: Label
    confidence: float
    reason_code: str
    reason: str


class LooseningJudger:
    """Convert line-geometry features into normal / loose / unknown."""

    LOOSE_STATUSES = {
        "misaligned_angle",
        "misaligned_offset",
        "broken_gap",
        "branched_or_fragmented",
        "bridge_break",
    }

    NORMAL_STATUSES = {
        "single_continuous",
        "multi_aligned",
        "curved_continuous",
    }

    def judge(self, features: Dict) -> LooseningJudgment:
        line_status = str(features.get("line_status", "missing"))
        refined_area = int(features.get("refined_area", features.get("red_area", 0)) or 0)
        min_area = int(features.get("min_area", 20) or 20)
        candidate_count = int(features.get("sam_candidate_count", 0) or 0)
        best_red_precision = float(features.get("best_sam_red_precision", 0.0) or 0.0)
        best_sam_score = float(features.get("best_sam_score", 0.0) or 0.0)

        if candidate_count == 0:
            return LooseningJudgment(
                label="unknown",
                confidence=0.88,
                reason_code="no_sam_marking_candidate",
                reason="SAM3 did not return a usable marking-line candidate.",
            )

        if refined_area < max(20, min_area) and best_red_precision < 0.08:
            return LooseningJudgment(
                label="unknown",
                confidence=0.78,
                reason_code="marking_too_small_or_not_red",
                reason="The SAM3 candidate is too small or has weak red-mark support.",
            )

        if bool(features.get("interface_loose_evidence", False)):
            reason = str(features.get("interface_reason", "sam3_interface_loose_evidence"))
            angle = float(features.get("interface_angle_diff_deg", 0.0) or 0.0)
            confidence = 0.78
            if reason == "skeleton_direction_mismatch":
                confidence += min(0.10, max(angle - 30.0, 0.0) / 100.0)
            elif reason == "skeleton_interface_break":
                confidence += 0.06
            return LooseningJudgment(
                label="loose",
                confidence=min(0.90, confidence),
                reason_code=f"interface_{reason}",
                reason="The red marking skeleton changes direction or breaks at the SAM3 bolt/nut/washer outer interface.",
            )

        if line_status in self.LOOSE_STATUSES:
            confidence = self._loose_confidence(features)
            return LooseningJudgment(
                label="loose",
                confidence=confidence,
                reason_code=f"line_{line_status}",
                reason=self._loose_reason(line_status),
            )

        if line_status in self.NORMAL_STATUSES:
            confidence = 0.66
            if line_status == "single_continuous":
                confidence = 0.72
            if best_sam_score >= 0.75:
                confidence += 0.08
            if best_red_precision >= 0.75:
                confidence += 0.06
            return LooseningJudgment(
                label="normal",
                confidence=min(0.92, confidence),
                reason_code=f"line_{line_status}",
                reason="The SAM3 marking line is continuous/aligned with no geometric loosening signal.",
            )

        if line_status == "single_irregular":
            return LooseningJudgment(
                label="normal",
                confidence=0.60,
                reason_code="line_single_irregular",
                reason="The marking line is present as one connected stroke; irregular paint edges alone are not loosening.",
            )

        return LooseningJudgment(
            label="unknown",
            confidence=0.70,
            reason_code=f"line_{line_status}",
            reason="The marking-line geometry is not sufficient for a reliable judgment.",
        )

    @staticmethod
    def _loose_confidence(features: Dict) -> float:
        line_status = str(features.get("line_status", ""))
        angle = float(features.get("line_angle_difference", 0.0) or 0.0)
        offset = float(features.get("line_perpendicular_offset", 0.0) or 0.0)
        gap = float(features.get("line_min_endpoint_gap", 0.0) or 0.0)
        endpoint_count = int(features.get("line_endpoint_count", 0) or 0)

        confidence = 0.72
        if line_status == "misaligned_angle":
            confidence += min(0.18, angle / 3.14 * 0.35)
        elif line_status == "misaligned_offset":
            confidence += min(0.16, offset / 120.0)
        elif line_status == "broken_gap":
            confidence += min(0.16, gap / 160.0)
        elif line_status == "branched_or_fragmented":
            confidence += min(0.12, max(endpoint_count - 4, 0) * 0.025)
        return min(0.94, confidence)

    @staticmethod
    def _loose_reason(line_status: str) -> str:
        reasons = {
            "misaligned_angle": "Two or more SAM3 marking-line segments have a clear angle mismatch.",
            "misaligned_offset": "Two or more marking-line segments are laterally offset.",
            "broken_gap": "The nearest endpoints of marking-line segments are separated by a large gap.",
            "branched_or_fragmented": "The marking-line skeleton has too many endpoints, suggesting fragmentation.",
            "bridge_break": "The marking line has a narrow/broken bridge at the interface with local direction change.",
        }
        return reasons.get(line_status, "The marking-line geometry indicates loosening.")
