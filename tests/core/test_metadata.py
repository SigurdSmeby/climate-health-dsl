"""Tests for the ground-truth metadata sidecar."""
import json

from dsl import __version__
from dsl.core.config.schema import parse_config
from dsl.core.pipeline.metadata import build_metadata, write_metadata


def sample_config():
    return parse_config(
        {
            "period": "monthly",
            "n_total": 24,
            "seed": 7,
            "train_fraction": 0.8,
            "start_period": "2010-01",
            "locations": ["oslo", "bergen"],
            "variables": [
                {"name": "rainfall", "generate": "seasonal_spike"},
                {"name": "mean_temperature", "generate": "seasonal_smooth"},
            ],
            "disease_cases": {
                "population": 1000,
                "depends_on": [
                    {"variable": "rainfall", "lag": 2, "weight": 1.5},
                    {"variable": "mean_temperature", "lag": 1, "weight": 1.0},
                ],
            },
        }
    )


def test_metadata_records_the_ground_truth():
    meta = build_metadata(sample_config())
    # The knobs that define the embedded relationship must all be captured.
    assert meta["seed"] == 7
    assert meta["period"] == "monthly"
    assert meta["n_total"] == 24
    assert meta["start_period"] == "2010-01"
    assert meta["locations"] == ["oslo", "bergen"]
    deps = meta["disease_cases"]["depends_on"]
    assert deps[0] == {"variable": "rainfall", "lag": 2, "weight": 1.5}
    assert meta["disease_cases"]["population"] == 1000


def test_metadata_records_generators():
    meta = build_metadata(sample_config())
    gens = {v["name"]: v["generate"] for v in meta["variables"]}
    assert gens == {
        "rainfall": "seasonal_spike",
        "mean_temperature": "seasonal_smooth",
    }


def test_metadata_records_tool_version():
    meta = build_metadata(sample_config())
    assert meta["dsl_version"] == __version__


def test_metadata_is_json_serializable():
    # The whole point is a portable sidecar file: it must round-trip JSON.
    meta = build_metadata(sample_config())
    assert json.loads(json.dumps(meta)) == meta


def test_write_metadata_creates_file(tmp_path):
    write_metadata(sample_config(), tmp_path)
    path = tmp_path / "metadata.json"
    assert path.is_file()
    loaded = json.loads(path.read_text())
    assert loaded["seed"] == 7


def test_write_metadata_reproduces_config(tmp_path):
    # Feeding the saved scenario back into parse_config must yield the same
    # config — proving the sidecar fully describes the run.
    write_metadata(sample_config(), tmp_path)
    loaded = json.loads((tmp_path / "metadata.json").read_text())
    rebuilt = parse_config(loaded["scenario"])
    assert rebuilt == sample_config()
