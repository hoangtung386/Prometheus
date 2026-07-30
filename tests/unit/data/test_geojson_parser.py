import json
from pathlib import Path

import pytest

from prometheus.data.puma.geojson import parse_nuclei_geojson, parse_tissue_geojson
from prometheus.domain import NucleusClass, TissueClass


def test_parser_preserves_instances_and_official_names(tmp_path: Path) -> None:
    path = tmp_path / "nuclei.geojson"
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"classification": {"name": "nuclei_apoptosis"}},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],
                },
            }
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    instances = parse_nuclei_geojson(path)
    assert len(instances) == 1
    assert instances[0].label is NucleusClass.APOPTOSIS
    assert instances[0].centroid == (2.0, 2.0)


def test_parser_uses_official_vertex_mean_not_area_centroid(tmp_path: Path) -> None:
    path = tmp_path / "nuclei.geojson"
    path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "properties": {"label": "nuclei_tumor"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [8, 0], [2, 2], [0, 0]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    instance = parse_nuclei_geojson(path)[0]
    assert instance.centroid == (10 / 3, 2 / 3)


def test_unknown_label_fails_in_strict_mode(tmp_path: Path) -> None:
    path = tmp_path / "nuclei.geojson"
    path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "properties": {"label": "definitely_unknown"},
                        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 1]]]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown nuclei label"):
        parse_nuclei_geojson(path)


def test_unknown_label_is_skipped_when_not_strict(tmp_path: Path) -> None:
    path = tmp_path / "nuclei.geojson"
    path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "properties": {"label": "definitely_unknown"},
                        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 1]]]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert parse_nuclei_geojson(path, strict=False) == []


def test_interior_rings_are_preserved_for_tissue(tmp_path: Path) -> None:
    # Dropping interior rings fills a nested class with the class that encloses it.
    path = tmp_path / "tissue.geojson"
    path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "properties": {"label": "tissue_tumor"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[0, 0], [40, 0], [40, 40], [0, 40], [0, 0]],
                                [[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]],
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ((label, polygon),) = parse_tissue_geojson(path)

    assert label is TissueClass.TUMOR
    assert len(polygon.holes) == 1
    assert len(polygon.exterior) == 4, "the closing vertex must be dropped"


def test_non_areal_geometry_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "tissue.geojson"
    path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "properties": {"label": "tissue_tumor"},
                        "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert parse_tissue_geojson(path) == []
