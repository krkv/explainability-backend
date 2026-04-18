"""Tests for user-facing heart response formatting."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.usecases.heart.heart_functions import HeartFunctions
from src.usecases.heart.heart_config import HeartConfig
from src.usecases.heart.heart_usecase import HeartUseCase


class DummyHeartModel:
    def predict(self, dataframe):
        return np.zeros(len(dataframe), dtype=int)

    def predict_proba(self, dataframe):
        return np.tile(np.array([[0.75, 0.25]]), (len(dataframe), 1))


def build_feature_metadata():
    return {
        "age": {
            "display_name": "Age",
            "kind": "continuous",
            "aliases": ["patient age"],
            "categories": {},
        },
        "trestbps": {
            "display_name": "Resting Blood Pressure",
            "kind": "continuous",
            "aliases": ["blood pressure"],
            "categories": {},
        },
        "sex": {
            "display_name": "Sex",
            "kind": "categorical",
            "aliases": ["gender"],
            "categories": {
                "0": {"label": "Female"},
                "1": {"label": "Male"},
            },
        },
        "num": {
            "display_name": "Heart Disease",
            "kind": "categorical",
            "aliases": ["heart disease"],
            "categories": {
                "0": {"label": "No heart disease"},
                "1": {"label": "Heart disease present"},
            },
        },
    }


def build_heart_functions():
    dataset = pd.DataFrame(
        {
            "age": [54],
            "trestbps": [140],
            "sex": [1],
        },
        index=[10],
    )
    dataset_full = dataset.copy()
    dataset_full["num"] = [0]
    metadata = build_feature_metadata()

    return HeartFunctions(
        model=DummyHeartModel(),
        dataset=dataset,
        dataset_full=dataset_full,
        y_values=dataset_full["num"],
        explainer=Mock(),
        dice_exp=Mock(),
        dice_dataset=dataset.copy(),
        model_metadata={"description": "Test model", "parameters": {}},
        feature_metadata=metadata,
        alias_lookup={
            "age": "age",
            "patient age": "age",
            "trestbps": "trestbps",
            "blood pressure": "trestbps",
            "resting blood pressure": "trestbps",
            "sex": "sex",
            "gender": "sex",
            "heart disease": "num",
        },
        global_feature_importances={"age": 0.7, "trestbps": 0.2, "sex": 0.1},
        target_variable="num",
        class_names=["NEGATIVE", "POSITIVE"],
        feature_names=["age", "trestbps", "sex"],
    )


def test_show_one_uses_display_names_and_feature_value_rows():
    heart_functions = build_heart_functions()

    response = heart_functions.show_one(10)

    assert response["data"]["patient_data"] == {
        "Age": 54,
        "Resting Blood Pressure": 140,
        "Sex": "Male",
    }
    assert "<th>Feature</th>" in response["text"]
    assert "<th>Value</th>" in response["text"]
    assert "Resting Blood Pressure" in response["text"]
    assert "trestbps" not in response["text"]


def test_dataset_summary_uses_display_names():
    heart_functions = build_heart_functions()

    response = heart_functions.dataset_summary()

    summary = response["data"]["comparison"]["all_patients_average"]
    assert "Age" in summary
    assert "Resting Blood Pressure" in summary
    assert "Sex" in summary
    assert "trestbps" not in response["text"]


def test_feature_importance_patient_returns_labeled_data_even_from_legacy_cache():
    heart_functions = build_heart_functions()
    heart_functions._shap_cache[10] = (
        [0.4, 0.2, 0.1],
        "<p>legacy cached text with raw feature names trestbps</p>",
    )

    response = heart_functions.feature_importance_patient(10)

    assert response["data"]["feature_importance"] == {
        "Age": 0.4,
        "Resting Blood Pressure": 0.2,
        "Sex": 0.1,
    }
    assert "<th>Feature</th>" in response["text"]
    assert "<th>Importance</th>" in response["text"]
    assert "Resting Blood Pressure" in response["text"]
    assert "trestbps" not in response["text"]


def test_feature_importance_global_returns_labeled_data():
    heart_functions = build_heart_functions()

    response = heart_functions.feature_importance_global()

    assert response["data"]["global_feature_importance"] == {
        "Age": 0.7,
        "Resting Blood Pressure": 0.2,
        "Sex": 0.1,
    }
    assert "<th>Feature</th>" in response["text"]
    assert "<th>Importance</th>" in response["text"]
    assert "trestbps" not in response["text"]


def test_heart_usecase_alias_lookup_includes_display_name(tmp_path):
    metadata_path = tmp_path / "feature_metadata.json"
    metadata_path.write_text(json.dumps(build_feature_metadata()), encoding="utf-8")

    functions_path = tmp_path / "functions.json"
    functions_path.write_text("[]", encoding="utf-8")

    config = HeartConfig(
        dataset_path=Path("unused.csv"),
        feature_metadata_path=metadata_path,
        functions_json_path=functions_path,
        shap_cache_path=tmp_path / "shap_cache.pkl",
        cf_cache_path=tmp_path / "cf_cache.pkl",
        global_fi_cache_path=tmp_path / "global_fi_cache.pkl",
    )

    usecase = HeartUseCase(
        model_loader=Mock(),
        data_loader=Mock(),
        explainer_loader=Mock(),
        config=config,
    )

    assert usecase.alias_lookup["resting blood pressure"] == "trestbps"
    assert usecase.alias_lookup["sex"] == "sex"
