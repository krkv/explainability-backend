"""Unit tests for the feature value adapter."""

import pytest

from src.core.feature_value_adapter import FeatureValueAdapter


@pytest.fixture
def sample_metadata():
    return {
        "age": {
            "display_name": "Age",
            "kind": "continuous",
            "aliases": ["patient age"],
            "transform": {"type": "identity"},
        },
        "sex": {
            "display_name": "Sex",
            "kind": "categorical",
            "aliases": ["gender"],
            "categories": {
                "0": {"label": "Female"},
                "1": {"label": "Male"},
            },
            "transform": {"type": "identity"},
        },
    }


def test_identity_numeric_passthrough(sample_metadata):
    adapter = FeatureValueAdapter(sample_metadata)
    assert adapter.to_display("age", 0.5) == 0.5
    assert adapter.to_model("age", 0.5) == 0.5


def test_categorical_label_decoding(sample_metadata):
    adapter = FeatureValueAdapter(sample_metadata)
    assert adapter.to_display("sex", 1) == "Male"
    assert adapter.to_model("sex", "Female") == 0


def test_delta_to_model_identity_behavior(sample_metadata):
    adapter = FeatureValueAdapter(sample_metadata)
    assert adapter.delta_to_model("age", 10.0) == 10.0


def test_delta_to_model_rejects_categorical(sample_metadata):
    adapter = FeatureValueAdapter(sample_metadata)
    with pytest.raises(ValueError):
        adapter.delta_to_model("sex", 1.0)
