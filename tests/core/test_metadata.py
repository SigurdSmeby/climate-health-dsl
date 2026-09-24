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
            "series": [
                {"name": "rainfall", "generate": "seasonal_spike"},
                {"name": "mean_temperature", "generate": "seasonal_smooth"}, {"name": "disease_cases", "counts": {"population": 1000}, "depends_on": [
                    {"series": "rainfall", "lag": 2, "weight": 1.5},
                    {"series": "mean_temperature", "lag": 1, "weight": 1.0},
                ]}],
        }
    )


def test_metadata_records_the_ground_truth():
    scenario = build_metadata(sample_config())["scenario"]
    # The knobs that define the embedded relationship must all be captured.
    assert scenario["seed"] == 7
    assert scenario["period"] == "monthly"
    assert scenario["n_total"] == 24
    assert scenario["start_period"] == "2010-01"
    assert scenario["locations"] == ["oslo", "bergen"]
    disease = next(sp for sp in scenario["series"] if sp.get("counts"))
    deps = disease["depends_on"]
    assert deps[0] == {
        "series": "rainfall", "lag": 2, "weight": 1.5, "transforms": [],
    }
    assert disease["counts"]["population"] == 1000


def test_metadata_records_generators():
    scenario = build_metadata(sample_config())["scenario"]
    gens = {
        spec["name"]: spec["generate"]
        for spec in scenario["series"]
        if spec.get("generate")
    }
    assert gens == {
        "rainfall": "seasonal_spike",
        "mean_temperature": "seasonal_smooth",
    }


def test_metadata_includes_count_distribution():
    config = parse_config(
        {
            "period": "monthly", "n_total": 3, "seed": 0,
            "series": [{"name": "rainfall", "generate": "seasonal_spike"}, {"name": "disease_cases", "counts": {"population": 100, "distribution": "negative_binomial", "overdispersion": 2.5}, "depends_on": [{"series": "rainfall", "lag": 1}]}],
        }
    )
    scenario = build_metadata(config)["scenario"]
    counts = next(sp for sp in scenario["series"] if sp.get("counts"))["counts"]
    assert counts["distribution"] == "negative_binomial"
    assert counts["overdispersion"] == 2.5


def test_metadata_distribution_default():
    scenario = build_metadata(sample_config())["scenario"]
    counts = next(sp for sp in scenario["series"] if sp.get("counts"))["counts"]
    assert counts["distribution"] == "poisson"


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
    assert loaded["scenario"]["seed"] == 7


def test_write_metadata_reproduces_config(tmp_path):
    # Feeding the saved scenario back into parse_config must yield the same
    # config — proving the sidecar fully describes the run.
    write_metadata(sample_config(), tmp_path)
    loaded = json.loads((tmp_path / "metadata.json").read_text())
    rebuilt = parse_config(loaded["scenario"])
    assert rebuilt == sample_config()


def mapping_location_config():
    """A scenario using the per-location population mapping form."""
    return parse_config(
        {
            "period": "monthly",
            "n_total": 24,
            "seed": 7,
            "locations": {
                "oslo": {"population": 700_000},
                "bergen": {"population": 280_000},
            },
            "series": [{"name": "rainfall", "generate": "seasonal_spike"}, {"name": "disease_cases", "counts": {"population": 10_000}, "depends_on": [{"series": "rainfall", "lag": 2}]}],
        }
    )


def test_metadata_preserves_per_location_population():
    # Regression test: location_overrides is excluded from model_dump, so the
    # mapping form's per-location populations must be rebuilt into the
    # scenario block — otherwise the round-trip silently loses them.
    config = mapping_location_config()
    rebuilt = parse_config(build_metadata(config)["scenario"])
    disease = next(sp for sp in rebuilt.series if sp.counts is not None)
    assert rebuilt.population_for("oslo", disease) == 700_000
    assert rebuilt.population_for("bergen", disease) == 280_000
    assert rebuilt == config


def test_write_metadata_reproduces_mapping_form(tmp_path):
    config = mapping_location_config()
    write_metadata(config, tmp_path)
    loaded = json.loads((tmp_path / "metadata.json").read_text())
    assert parse_config(loaded["scenario"]) == config


def test_metadata_round_trips_population_generator():
    # A time-varying (generated) population must survive the round-trip,
    # otherwise reproducing the run would silently revert to a constant.
    config = parse_config(
        {
            "period": "monthly",
            "n_total": 24,
            "series": [
                {"name": "rainfall", "generate": "seasonal_spike"},
                {
                    "name": "disease_cases",
                    "counts": {
                        "population": {
                            "generate": "linear_trend",
                            "params": {"start": 1000, "slope": 10},
                        },
                    },
                    "depends_on": [{"series": "rainfall", "lag": 1}],
                },
            ],
        }
    )
    meta = build_metadata(config)
    # The whole dict must be JSON-serializable: a generated population is a
    # PopulationSpec, which only model_dump(mode="json") flattens safely
    # (regression: a raw spec leaked through, breaking write_metadata).
    json.dumps(meta)
    rebuilt = parse_config(meta["scenario"])
    assert rebuilt == config


def test_write_metadata_with_population_generator(tmp_path):
    # The full write path (json.dumps to disk) must succeed with a generated
    # population — this is what the CLI does.
    config = parse_config(
        {
            "period": "monthly",
            "n_total": 24,
            "series": [
                {"name": "rainfall", "generate": "seasonal_spike"},
                {
                    "name": "disease_cases",
                    "counts": {
                        "population": {
                            "generate": "linear_trend",
                            "params": {"start": 1000, "slope": 10},
                        },
                    },
                    "depends_on": [{"series": "rainfall", "lag": 1}],
                },
            ],
        }
    )
    write_metadata(config, tmp_path)
    loaded = json.loads((tmp_path / "metadata.json").read_text())
    disease = next(
        sp for sp in loaded["scenario"]["series"] if sp.get("counts")
    )
    assert disease["counts"]["population"] == {
        "generate": "linear_trend",
        "params": {"start": 1000, "slope": 10},
    }
