"""Serialize detections using the official PUMA Multiple Polygons schema."""

from __future__ import annotations

import json
from pathlib import Path

from ..domain import Detection, NucleusClass, Track, nucleus_class_for_track


def _class_name(label: NucleusClass | str, track: Track) -> str:
    if isinstance(label, NucleusClass):
        canonical_name = nucleus_class_for_track(label, track)
    else:
        canonical_name = str(label).removeprefix("nuclei_")
    return f"nuclei_{canonical_name}"


def _path_points(detection: Detection) -> list[list[float]]:
    x_coord, y_coord = detection.centroid
    if detection.box_xyxy is None:
        x_min, y_min = x_coord - 0.5, y_coord - 0.5
        x_max, y_max = x_coord + 0.5, y_coord + 0.5
    else:
        x_min, y_min, x_max, y_max = detection.box_xyxy
    # Do not repeat the first point: the official evaluator computes the
    # arithmetic mean of path_points rather than a geometric polygon centroid.
    return [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]


def write_nuclei_json(
    detections: list[Detection],
    path: str | Path,
    track: Track = Track.TRACK_2,
) -> None:
    polygons = [
        {
            "name": _class_name(detection.label, track),
            "score": float(detection.confidence),
            "path_points": _path_points(detection),
        }
        for detection in detections
    ]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file_obj:
        json.dump({"polygons": polygons}, file_obj)
