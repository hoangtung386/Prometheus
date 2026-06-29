import json

from prometheus.data.puma.geojson import parse_nuclei_geojson
from prometheus.domain import NucleusClass


def test_parser_preserves_instances_and_official_names(tmp_path) -> None:
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


def test_parser_uses_official_vertex_mean_not_area_centroid(tmp_path) -> None:
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


def test_unknown_label_fails_in_strict_mode(tmp_path) -> None:
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
    try:
        parse_nuclei_geojson(path)
    except ValueError as error:
        assert "Unknown nuclei label" in str(error)
    else:
        raise AssertionError("Unknown labels must fail in strict mode")
