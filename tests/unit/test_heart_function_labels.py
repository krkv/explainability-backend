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
            "description": "Age of the patient in years.",
            "unit": "years",
            "kind": "continuous",
            "aliases": ["patient age"],
            "categories": {},
        },
        "trestbps": {
            "display_name": "Resting Blood Pressure",
            "description": "Resting blood pressure on admission to the hospital.",
            "unit": "mm Hg",
            "kind": "continuous",
            "aliases": ["blood pressure"],
            "categories": {},
        },
        "cp": {
            "display_name": "Chest Pain Type",
            "description": "Chest pain type from the published Cleveland processed dataset.",
            "unit": None,
            "kind": "categorical",
            "aliases": ["chest pain"],
            "categories": {
                "1": {"label": "Typical angina"},
                "2": {"label": "Atypical angina"},
                "3": {"label": "Non-anginal pain"},
                "4": {"label": "Asymptomatic"},
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
        },
        "num": {
            "display_name": "Heart Disease",
            "description": "Binary heart disease target derived from the original UCI diagnosis label.",
            "unit": None,
            "kind": "categorical",
            "aliases": ["heart disease"],
            "categories": {
                "0": {"label": "No heart disease"},
                "1": {"label": "Heart disease present"},
            },
        },
        "ca": {
            "display_name": "Major Vessels",
            "description": "Number of major vessels colored by fluoroscopy.",
            "unit": "count",
            "kind": "categorical",
            "aliases": [],
            "categories": {
                "0": {"label": "0 vessels"},
                "1": {"label": "1 vessel"},
                "2": {"label": "2 vessels"},
                "3": {"label": "3 vessels"},
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
            "cp": "cp",
            "chest pain": "cp",
            "chest pain type": "cp",
            "sex": "sex",
            "gender": "sex",
            "major vessels": "ca",
            "heart disease": "num",
        },
        global_feature_importances={"age": 0.7, "trestbps": 0.2, "sex": 0.1},
        target_variable="num",
        class_names=["NEGATIVE", "POSITIVE"],
        feature_names=["age", "trestbps", "sex"],
    )


def test_misclassified_cases_summarizes_groups_with_display_labels():
    dataset = pd.DataFrame(
        {
            "age": [54, 62, 49],
            "trestbps": [140, 150, 130],
            "sex": [1, 0, 1],
        },
        index=[10, 11, 12],
    )
    dataset_full = dataset.copy()
    dataset_full["num"] = [0, 1, 1]
    metadata = build_feature_metadata()

    class MixedPredictionModel:
        def predict(self, dataframe):
            return np.array([0, 0, 1], dtype=int)

        def predict_proba(self, dataframe):
            return np.tile(np.array([[0.75, 0.25]]), (len(dataframe), 1))

    heart_functions = HeartFunctions(
        model=MixedPredictionModel(),
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
        },
        global_feature_importances={"age": 0.7, "trestbps": 0.2, "sex": 0.1},
        target_variable="num",
        class_names=["NEGATIVE", "POSITIVE"],
        feature_names=["age", "trestbps", "sex"],
    )

    response = heart_functions.misclassified_cases()

    assert response["data"]["false_positives"] == 0
    assert response["data"]["false_negatives"] == 1
    assert response["data"]["feature_distribution"]["misclassified_cases"] == {
        "Age": 62.0,
        "Resting Blood Pressure": 150.0,
        "Sex": "Female",
    }
    assert response["data"]["feature_distribution"]["correctly_classified_cases"] == {
        "Age": 51.5,
        "Resting Blood Pressure": 135.0,
        "Sex": "Male",
    }
    assert "Resting Blood Pressure" in response["text"]
    assert "_summarize_group" not in response["text"]


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

    summary_rows = response["data"]["dataset_statistics"]
    labels = {row["Feature"] for row in summary_rows}
    assert "Age" in labels
    assert "Resting Blood Pressure" in labels
    assert "Sex" in labels
    assert "heart_disease_average" not in response["data"]
    assert "Mean / Mode" in response["text"]
    assert "trestbps" not in response["text"]


def test_define_feature_returns_feature_definition_for_alias():
    heart_functions = build_heart_functions()

    response = heart_functions.define_feature("blood pressure")

    assert response["data"]["feature"] == "trestbps"
    assert response["data"]["display_name"] == "Resting Blood Pressure"
    assert response["data"]["description"] == "Resting blood pressure on admission to the hospital."
    assert response["data"]["unit"] == "mm Hg"
    assert "Resting Blood Pressure" in response["text"]
    assert "mm Hg" in response["text"]


def test_define_feature_returns_categories_for_categorical_feature():
    heart_functions = build_heart_functions()

    response = heart_functions.define_feature("chest pain")

    assert response["data"]["feature"] == "cp"
    assert response["data"]["categories"] == {
        "1": "Typical angina",
        "2": "Atypical angina",
        "3": "Non-anginal pain",
        "4": "Asymptomatic",
    }
    assert "Typical angina" in response["text"]
    assert "Asymptomatic" in response["text"]


def test_define_feature_returns_definition_for_display_name():
    heart_functions = build_heart_functions()

    response = heart_functions.define_feature("Major Vessels")

    assert response["data"]["feature"] == "ca"
    assert response["data"]["display_name"] == "Major Vessels"
    assert response["data"]["categories"]["3"] == "3 vessels"
    assert "Major Vessels" in response["text"]
    assert "0 vessels" in response["text"]


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
