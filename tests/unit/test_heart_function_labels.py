"""Tests for user-facing heart response formatting."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.domain.interfaces.llm_provider import AgentRole
from src.usecases.energy.energy_config import EnergyConfig
from src.usecases.energy.energy_usecase import EnergyUseCase
from src.usecases.heart.heart_functions import HeartFunctions
from src.usecases.heart.heart_config import HeartConfig
from src.usecases.heart.heart_usecase import HeartUseCase


class DummyHeartModel:
    def predict(self, dataframe):
        return np.zeros(len(dataframe), dtype=int)

    def predict_proba(self, dataframe):
        return np.tile(np.array([[0.75, 0.25]]), (len(dataframe), 1))


class CategoricalWhatIfModel:
    def predict(self, dataframe):
        positive = (dataframe["cp"].to_numpy() >= 3).astype(int)
        return positive

    def predict_proba(self, dataframe):
        positive_prob = np.where(dataframe["cp"].to_numpy() >= 3, 0.8, 0.2)
        negative_prob = 1 - positive_prob
        return np.column_stack([negative_prob, positive_prob])


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
        functions_catalog=[
            {
                "type": "function",
                "function": {
                    "name": "available_functions",
                    "description": "Get a formatted list of all available heart use case functions.",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "count_patients",
                    "description": "Count patients in the heart dataset.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "count_type": {
                                "type": "string",
                                "description": "Which patient count to return.",
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "predict",
                    "description": "Predict the class and probability scores for a specific patient using their ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "integer",
                                "description": "The ID of the patient to predict.",
                            }
                        },
                        "required": ["patient_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dataset_summary",
                    "description": "Return one table with overall dataset statistics for each feature.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "integer",
                                "description": "Optional patient ID to include.",
                            }
                        },
                        "required": [],
                    },
                },
            },
        ],
    )


def build_heart_functions_with_cp():
    dataset = pd.DataFrame(
        {
            "age": [54],
            "trestbps": [140],
            "cp": [4],
            "sex": [1],
        },
        index=[10],
    )
    dataset_full = dataset.copy()
    dataset_full["num"] = [1]
    metadata = build_feature_metadata()

    return HeartFunctions(
        model=CategoricalWhatIfModel(),
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
        global_feature_importances={"age": 0.7, "trestbps": 0.2, "cp": 0.15, "sex": 0.1},
        target_variable="num",
        class_names=["NEGATIVE", "POSITIVE"],
        feature_names=["age", "trestbps", "cp", "sex"],
        functions_catalog=[],
    )


def test_available_functions_returns_formatted_catalog():
    heart_functions = build_heart_functions()

    response = heart_functions.available_functions()

    available = response["data"]["available_functions"]
    assert [item["name"] for item in available] == [
        "available_functions",
        "count_patients",
        "predict",
        "dataset_summary",
    ]
    assert available[1]["signature"] == "count_patients(count_type=optional)"
    assert available[2]["signature"] == "predict(patient_id=...)"
    assert available[3]["signature"] == "dataset_summary(patient_id=optional)"
    assert "<code>count_patients(count_type=optional)</code>" in response["text"]
    assert "Return one table with overall dataset statistics" in response["text"]


def test_count_patients_returns_total_and_predicted_class_counts():
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
            return np.array([0, 1, 1], dtype=int)

        def predict_proba(self, dataframe):
            return np.tile(np.array([[0.25, 0.75]]), (len(dataframe), 1))

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
            "heart disease": "num",
        },
        global_feature_importances={"age": 0.7, "trestbps": 0.2, "sex": 0.1},
        target_variable="num",
        class_names=["NEGATIVE", "POSITIVE"],
        feature_names=["age", "trestbps", "sex"],
    )

    total_response = heart_functions.count_patients()
    positive_response = heart_functions.count_patients("positive_predicted")
    negative_response = heart_functions.count_patients("negative predicted")

    assert total_response["data"] == {"count": 3, "count_type": "total"}
    assert "3" in total_response["text"]

    assert positive_response["data"] == {
        "count": 2,
        "count_type": "positive_predicted",
        "prediction": 1,
    }
    assert "predicts heart disease for <var>2</var> patients" in positive_response["text"]

    assert negative_response["data"] == {
        "count": 1,
        "count_type": "negative_predicted",
        "prediction": 0,
    }
    assert "predicts no heart disease for <var>1</var> patients" in negative_response["text"]


def test_count_patients_rejects_unknown_count_type():
    heart_functions = build_heart_functions()

    response = heart_functions.count_patients("actual_positive")

    assert "error" in response
    assert "not supported" in response["error"]
    assert "positive_predicted" in response["text"]


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


def test_age_group_performance_handles_non_contiguous_patient_ids():
    dataset = pd.DataFrame(
        {
            "age": [35, 50, 70],
            "trestbps": [120, 130, 140],
            "sex": [1, 0, 1],
        },
        index=[10, 11, 12],
    )
    dataset_full = dataset.copy()
    dataset_full["num"] = [0, 1, 1]
    metadata = build_feature_metadata()

    class MixedPredictionModel:
        def predict(self, dataframe):
            return np.array([0, 1, 0], dtype=int)

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

    response = heart_functions.age_group_performance()

    assert set(response["data"].keys()) == {"<40", "40-60", ">60"}
    assert response["data"]["<40"]["accuracy"] == 1.0
    assert response["data"]["40-60"]["accuracy"] == 1.0
    assert response["data"][">60"]["accuracy"] == 0.0


def test_what_if_rejects_non_string_feature_gracefully():
    heart_functions = build_heart_functions()

    response = heart_functions.what_if(10, 123, 1.0)

    assert "error" in response
    assert "not found in patient data" in response["error"]
    assert "<p>Feature <code>123</code> not found in patient data.</p>" == response["text"]


def test_what_if_accepts_categorical_label_shorthand():
    heart_functions = build_heart_functions_with_cp()

    response = heart_functions.what_if(10, "chest pain", "typical")

    assert response["data"]["feature_modified"] == "Chest Pain Type"
    assert response["data"]["value_change"] == "typical"
    assert response["data"]["original_value"] == "Asymptomatic"
    assert response["data"]["new_value"] == "Typical angina"
    assert response["data"]["original_prediction"] == 1
    assert response["data"]["new_prediction"] == 0
    assert "changing <code>Chest Pain Type</code> from <var>Asymptomatic</var> to <var>Typical angina</var>" in response["text"]


def test_what_if_returns_available_categories_for_unknown_categorical_label():
    heart_functions = build_heart_functions_with_cp()

    response = heart_functions.what_if(10, "chest pain", "mystery")

    assert "error" in response
    assert "not recognized" in response["error"]
    assert "Typical angina" in response["text"]
    assert "Asymptomatic" in response["text"]


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


def test_heart_usecase_registers_available_functions(tmp_path):
    metadata_path = tmp_path / "feature_metadata.json"
    metadata_path.write_text(json.dumps(build_feature_metadata()), encoding="utf-8")

    functions_path = tmp_path / "functions.json"
    functions_path.write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "available_functions",
                        "description": "Get a formatted list of all available heart use case functions.",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    config = HeartConfig(
        dataset_path=Path("unused.csv"),
        feature_metadata_path=metadata_path,
        functions_json_path=functions_path,
        shap_cache_path=tmp_path / "shap_cache.pkl",
        cf_cache_path=tmp_path / "cf_cache.pkl",
        global_fi_cache_path=tmp_path / "global_fi_cache.pkl",
    )

    model_loader = Mock()
    data_loader = Mock()
    explainer_loader = Mock()
    dataset = pd.DataFrame({"age": [54], "sex": [1], "trestbps": [140], "num": [0]}, index=[10])
    model_loader.load_model.return_value = DummyHeartModel()
    data_loader.load_dataset.return_value = dataset

    usecase = HeartUseCase(
        model_loader=model_loader,
        data_loader=data_loader,
        explainer_loader=explainer_loader,
        config=config,
    )
    usecase._explainer = Mock()
    usecase._dice_exp = Mock()
    usecase._dice_dataset = dataset.drop(columns=["num"]).copy()
    usecase._global_feature_importances = {"age": 0.7, "trestbps": 0.2, "sex": 0.1}

    functions = usecase.get_functions()

    assert "available_functions" in functions
    assert "count_patients" in functions


def test_heart_usecase_prompt_guides_categorical_label_mapping(tmp_path):
    metadata_path = tmp_path / "feature_metadata.json"
    metadata_path.write_text(json.dumps(build_feature_metadata()), encoding="utf-8")

    functions_path = tmp_path / "functions.json"
    functions_path.write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "what_if",
                        "description": "Simulate how changing a specific feature value would affect the prediction for a patient.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "patient_id": {"type": "integer"},
                                "feature": {"type": "string"},
                                "value_change": {
                                    "anyOf": [{"type": "number"}, {"type": "string"}]
                                },
                            },
                            "required": ["patient_id", "feature", "value_change"],
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    config = HeartConfig(
        dataset_path=Path("unused.csv"),
        feature_metadata_path=metadata_path,
        functions_json_path=functions_path,
        shap_cache_path=tmp_path / "shap_cache.pkl",
        cf_cache_path=tmp_path / "cf_cache.pkl",
        global_fi_cache_path=tmp_path / "global_fi_cache.pkl",
    )

    model_loader = Mock()
    data_loader = Mock()
    explainer_loader = Mock()
    dataset = pd.DataFrame({"age": [54], "sex": [1], "cp": [4], "trestbps": [140], "num": [1]}, index=[10])
    model_loader.load_model.return_value = DummyHeartModel()
    data_loader.load_dataset.return_value = dataset

    usecase = HeartUseCase(
        model_loader=model_loader,
        data_loader=data_loader,
        explainer_loader=explainer_loader,
        config=config,
    )

    prompt = usecase.get_system_prompt([{"role": "user", "content": "What if chest pain was typical?"}])

    assert "feature metadata with display names, aliases, descriptions, and categorical options" in prompt
    assert 'value_change="Typical angina"' in prompt
    assert "Convert those into the appropriate function arguments instead of refusing" in prompt


def test_heart_usecase_prompt_guides_patient_count_mapping(tmp_path):
    metadata_path = tmp_path / "feature_metadata.json"
    metadata_path.write_text(json.dumps(build_feature_metadata()), encoding="utf-8")

    functions_path = tmp_path / "functions.json"
    functions_path.write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "count_patients",
                        "description": "Count patients in the heart dataset.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "count_type": {
                                    "type": "string",
                                    "enum": ["total", "positive_predicted", "negative_predicted"],
                                }
                            },
                            "required": [],
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    config = HeartConfig(
        dataset_path=Path("unused.csv"),
        feature_metadata_path=metadata_path,
        functions_json_path=functions_path,
        shap_cache_path=tmp_path / "shap_cache.pkl",
        cf_cache_path=tmp_path / "cf_cache.pkl",
        global_fi_cache_path=tmp_path / "global_fi_cache.pkl",
    )

    model_loader = Mock()
    data_loader = Mock()
    explainer_loader = Mock()
    dataset = pd.DataFrame({"age": [54], "sex": [1], "trestbps": [140], "num": [1]}, index=[10])
    model_loader.load_model.return_value = DummyHeartModel()
    data_loader.load_dataset.return_value = dataset

    usecase = HeartUseCase(
        model_loader=model_loader,
        data_loader=data_loader,
        explainer_loader=explainer_loader,
        config=config,
    )

    prompt = usecase.get_system_prompt([{"role": "user", "content": "How many patients have heart disease?"}])

    assert "how many patients have heart disease" in prompt
    assert "positive predicted count option" in prompt
    assert "negative predicted count option" in prompt


def test_heart_usecase_suggester_generation_config_includes_latest_assistant_context(tmp_path):
    metadata_path = tmp_path / "feature_metadata.json"
    metadata_path.write_text(json.dumps(build_feature_metadata()), encoding="utf-8")

    functions_path = tmp_path / "functions.json"
    functions_path.write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "predict",
                        "description": "Predict the class and probability scores for a specific patient using their ID.",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "counterfactual",
                        "description": "Generate a counterfactual explanation for a patient.",
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    config = HeartConfig(
        dataset_path=Path("unused.csv"),
        feature_metadata_path=metadata_path,
        functions_json_path=functions_path,
        shap_cache_path=tmp_path / "shap_cache.pkl",
        cf_cache_path=tmp_path / "cf_cache.pkl",
        global_fi_cache_path=tmp_path / "global_fi_cache.pkl",
    )

    model_loader = Mock()
    data_loader = Mock()
    explainer_loader = Mock()
    dataset = pd.DataFrame({"age": [54], "sex": [1], "trestbps": [140], "num": [1]}, index=[10])
    model_loader.load_model.return_value = DummyHeartModel()
    data_loader.load_dataset.return_value = dataset

    usecase = HeartUseCase(
        model_loader=model_loader,
        data_loader=data_loader,
        explainer_loader=explainer_loader,
        config=config,
    )

    generation_config = usecase.get_generation_config(
        conversation=[{"role": "user", "content": "Show patient 10"}],
        agent_role=AgentRole.SUGGESTER,
        context={"latest_assistant_response": "Patient 10 has elevated predicted risk."},
    )

    assert generation_config.response_schema["required"] == ["suggestions"]
    assert generation_config.response_schema["properties"]["suggestions"]["minItems"] == 3
    assert generation_config.response_schema["properties"]["suggestions"]["maxItems"] == 5
    assert "medical professional" in generation_config.system_prompt
    assert "Patient 10 has elevated predicted risk." in generation_config.system_prompt
    assert "Do not mention function names" in generation_config.system_prompt
    assert '"name": "counterfactual"' in generation_config.system_prompt


def test_energy_usecase_rejects_suggester_generation_config(tmp_path):
    functions_path = tmp_path / "functions.json"
    functions_path.write_text("[]", encoding="utf-8")

    usecase = EnergyUseCase(
        model_loader=Mock(),
        data_loader=Mock(),
        explainer_loader=Mock(),
        config=EnergyConfig(
            dataset_path=Path("unused.csv"),
            functions_json_path=functions_path,
        ),
    )

    with pytest.raises(ValueError, match="only implemented for the heart use case"):
        usecase.get_generation_config(
            conversation=[{"role": "user", "content": "What should I ask next?"}],
            agent_role=AgentRole.SUGGESTER,
        )
