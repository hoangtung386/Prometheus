"""Tissue segmentation and nuclei detection metrics matching the official evaluator."""

from .matching import Match, MatchResult, match_detections
from .nuclei_detection import nuclei_detection_metrics
from .segmentation import SegmentationEvaluator, official_micro_dice

__all__ = [
    "Match",
    "MatchResult",
    "SegmentationEvaluator",
    "match_detections",
    "nuclei_detection_metrics",
    "official_micro_dice",
]
