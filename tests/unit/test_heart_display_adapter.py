"""Unit tests for heart display-space adapter wiring."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.core.feature_value_adapter import FeatureValueAdapter
from src.usecases.heart.heart_config import HeartConfig
from src.usecases.heart.heart_functions import HeartFunctions
from src.usecases.heart.heart_usecase import HeartUseCase


class DummyHeartModel:
    """Simple deterministic classifier for tests."""

    def predict(self, dataframe):
        return (dataframe["age"].astype(float) >= 0.6).astype(int).to_numpy()

    def predict_proba(self, dataframe):
        probabilities = []
        for _, row in dataframe.iterrows():
            positive = min(max(float(row["age"]), 0.0), 1.0)
            probabilities.append([1 - positive, positive])
        return np.asarray(probabilities)


class DummyDiceExplainer:
    """Simple DiCE stub returning one counterfactual row."""

    def generate_counterfactuals(self, query_instances, total_CFs, desired_class):
        final_cfs_df = query_instances.copy()
        final_cfs_df["age"] = 0.6

        class Example:
            pass

        example = Example()
        example.final_cfs_df = final_cfs_df

        class Result:
            pass

        result = Result()
        result.cf_examples_list = [example]
        return result


@pytest.fixture
def heart_metadata():
    return {
        "age": {
            "display_name": "Age",
            "description": "Age of the patient in years.",
            "unit": "years",
            "kind": "continuous",
            "aliases": ["patient age"],
            "categories": {},
            "transform": {
                "type": "min_max",
                "raw_min": 20,
                "raw_max": 80,
                "model_min": 0.0,
                "model_max": 1.0,
            },
        },
        "sex": {
            "display_name": "Sex",
            "description": "Sex of the patient.",
            "unit": None,
            "kind": "categorical",
            "aliases": ["gender"],
            "categories": {
                "0": {"label": "Female"},
                "1": {"label": "Male"},
            },
            "transform": {"type": "identity"},
        },
        "num": {
            "display_name": "Heart Disease",
            "description": "Presence of heart disease.",
            "unit": None,
            "kind": "categorical",
            "aliases": ["target", "heart disease"],
            "categories": {
                "0": {"label": "No heart disease"},
                "1": {"label": "Heart disease present"},
            },
            "transform": {"type": "identity"},
        },
    }


@pytest.fixture
def heart_frames():
    dataset = pd.DataFrame(
        {
            "age": [0.0, 1.0],
            "sex": [0, 1],
        },
        index=[10, 11],
    )
    dataset_full = dataset.copy()
    dataset_full["num"] = [0, 1]
    return dataset, dataset_full


@pytest.fixture
def heart_functions(heart_metadata, heart_frames):
    dataset, dataset_full = heart_frames
    adapter = FeatureValueAdapter(heart_metadata)
    display_dataset = adapter.to_display_frame(dataset, features=dataset.columns)
    display_dataset_full = adapter.to_display_frame(dataset_full, features=dataset.columns)
    return HeartFunctions(
        model=DummyHeartModel(),
        dataset=dataset,
        dataset_full=dataset_full,
        display_dataset=display_dataset,
        display_dataset_full=display_dataset_full,
        y_values=dataset_full["num"],
        explainer=Mock(),
        dice_exp=DummyDiceExplainer(),
        dice_dataset=dataset.copy(),
        feature_adapter=adapter,
        model_metadata={"description": "Test model", "parameters": {}},
        feature_metadata=heart_metadata,
        alias_lookup={"patient age": "age", "gender": "sex", "heart disease": "num"},
        global_feature_importances={"age": 0.7, "sex": 0.2},
        target_variable="num",
        class_names=["NEGATIVE", "POSITIVE"],
        feature_names=["age", "sex"],
        shap_cache_path=None,
        cf_cache_path=None,
    )


def test_show_one_returns_display_labels_and_values(heart_functions):
    response = heart_functions.show_one(10)
    patient_data = response["data"]["patient_data"]
    assert "Age" in patient_data
    assert patient_data["Age"] == 20.0
    assert patient_data["Sex"] == "Female"


def test_dataset_summary_uses_display_space_averages(heart_functions):
    response = heart_functions.dataset_summary()
    summary = response["data"]["comparison"]["all_patients_average"]
    assert summary["Age"] == 50.0
    assert summary["Sex"] in {"Female", "Male"}


def test_what_if_converts_display_delta_to_model_delta(heart_functions):
    response = heart_functions.what_if(patient_id=10, feature="patient age", value_change=36.0)
    data = response["data"]
    assert data["original_value"] == 20.0
    assert data["new_value"] == 56.0
    assert data["new_prediction"] == 1


def test_counterfactual_text_uses_display_labels_and_values(heart_functions):
    response = heart_functions.counterfactual(10)
    assert "Age" in response["text"]
    assert "56.0" in response["text"]
    assert response["data"]["counterfactuals"][0]["Age"] == 56.0


def test_heart_alias_lookup_uses_enriched_metadata():
    usecase = HeartUseCase(model_loader=Mock(), data_loader=Mock(), explainer_loader=Mock())
    assert usecase.alias_lookup["cholesterol"] == "chol"
    assert usecase.alias_lookup["gender"] == "sex"


def test_heart_prompt_uses_display_dataset_and_feature_catalog(tmp_path, heart_metadata):
    dataset = pd.DataFrame(
        {
            "age": [0.0, 1.0],
            "sex": [0, 1],
            "num": [0, 1],
        }
    )

    metadata_path = tmp_path / "feature_metadata.json"
    metadata_path.write_text(json.dumps(heart_metadata), encoding="utf-8")

    functions_path = tmp_path / "functions.json"
    functions_path.write_text(json.dumps([{"type": "function", "function": {"name": "what_if", "parameters": {"type": "object", "properties": {}}}}]), encoding="utf-8")

    config = HeartConfig(
        dataset_path=Path("unused.csv"),
        feature_metadata_path=metadata_path,
        functions_json_path=functions_path,
        shap_cache_path=tmp_path / "shap_cache.pkl",
        cf_cache_path=tmp_path / "cf_cache.pkl",
        global_fi_cache_path=tmp_path / "global_fi_cache.pkl",
    )

    data_loader = Mock()
    data_loader.load_dataset.return_value = dataset

    usecase = HeartUseCase(
        model_loader=Mock(),
        data_loader=data_loader,
        explainer_loader=Mock(),
        config=config,
    )

    prompt = usecase.get_system_prompt([])
    display_json = usecase.display_dataset.describe(include="all").fillna("").to_json()
    model_json = usecase.dataset.describe(include="all").fillna("").to_json()

    assert display_json in prompt
    assert model_json not in prompt
    assert "Age" in prompt
    assert "years" in prompt
