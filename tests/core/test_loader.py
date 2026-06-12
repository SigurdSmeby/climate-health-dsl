"""Tests for the YAML loader: file → plain dict, with clear errors."""
import pytest

from dsl.core.config.loader import load_yaml


def test_loads_valid_yaml(tmp_path):
    path = tmp_path / "scenario.yaml"
    path.write_text("period: weekly\nn_total: 78\n")
    data = load_yaml(path)
    assert data == {"period": "weekly", "n_total": 78}


def test_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_yaml(missing)


def test_malformed_yaml_raises_clear_error(tmp_path):
    path = tmp_path / "broken.yaml"
    # An unclosed bracket is invalid YAML syntax.
    path.write_text("variables: [unclosed\n")
    with pytest.raises(ValueError, match="broken.yaml"):
        load_yaml(path)


def test_non_mapping_yaml_raises(tmp_path):
    # A YAML file containing just a list is valid YAML but not a scenario.
    path = tmp_path / "list.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(path)
