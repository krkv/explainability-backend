"""Tests for two-decimal numeric formatting in HTML responses."""

from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.usecases.energy.energy_functions import EnergyFunctions
from src.usecases.heart.heart_functions import HeartFunctions


class DummyEnergyModel:
    def predict(self, dataframe):
        base = np.array([12.3456], dtype=float)
        return np.repeat(base, len(dataframe))


class DummyEnergyExplainer:
    def shap_values(self, dataframe, nsamples=None, silent=None):
        return np.array([[0.1234, -0.9876, 1.5]])


class DummyHeartModel:
    def predict(self, dataframe):
        return np.zeros(len(dataframe), dtype=int)

    def predict_proba(self, dataframe):
        return np.tile(np.array([[0.7543, 0.2457]]), (len(dataframe), 1))


def build_energy_functions():
    dataset = pd.DataFrame(
        {
            "indoor_temperature": [21.345],
            "outdoor_temperature": [5.4321],
            "past_electricity": [9.8765],
        },
        index=[10],
    )
    y_values = pd.Series([10.111], index=[10])

    return EnergyFunctions(
        model=DummyEnergyModel(),
        dataset=dataset,
        y_values=y_values,
        explainer=DummyEnergyExplainer(),
        dice_exp=Mock(),
        dice_dataset=dataset.copy(),
    )


def build_heart_functions():
    dataset = pd.DataFrame(
        {
            "age": [54.678],
            "trestbps": [140.432],
            "sex": [1],
        },
        index=[10],
    )
    dataset_full = dataset.copy()
    dataset_full["num"] = [0]
    feature_metadata = {
        "age": {"display_name": "Age", "kind": "continuous", "categories": {}},
        "trestbps": {"display_name": "Resting Blood Pressure", "kind": "continuous", "categories": {}},
        "sex": {
            "display_name": "Sex",
            "kind": "categorical",
            "categories": {"0": {"label": "Female"}, "1": {"label": "Male"}},
        },
    }

    return HeartFunctions(
        model=DummyHeartModel(),
        dataset=dataset,
        dataset_full=dataset_full,
        y_values=dataset_full["num"],
        explainer=Mock(),
        dice_exp=Mock(),
        dice_dataset=dataset.copy(),
        model_metadata={"description": "Test model", "parameters": {}},
        dataset_metadata={},
        feature_metadata=feature_metadata,
        alias_lookup={
            "age": "age",
            "trestbps": "trestbps",
            "resting blood pressure": "trestbps",
            "sex": "sex",
        },
        global_feature_importances={"age": 0.7345, "trestbps": 0.205, "sex": 0.1},
        target_variable="num",
        class_names=["NEGATIVE", "POSITIVE"],
        feature_names=["age", "trestbps", "sex"],
    )


def test_energy_html_rounds_inline_and_table_values_to_two_decimals():
    energy_functions = build_energy_functions()

    show_one_response = energy_functions.show_one(10)
    predict_new_response = energy_functions.predict_new(21.345, 5.4321, 9.8765)

    assert "21.35" in show_one_response
    assert "5.43" in show_one_response
    assert "9.88" in show_one_response
    assert "<var>21.35</var>" in predict_new_response
    assert "<var>5.43</var>" in predict_new_response
    assert "<var>9.88</var>" in predict_new_response
    assert "<samp>12.35</samp>" in predict_new_response


def test_heart_html_rounds_table_and_probability_values_to_two_decimals():
    heart_functions = build_heart_functions()

    importance_response = heart_functions.feature_importance_global()
    predict_response = heart_functions.predict(10)

    assert "0.73" in importance_response["text"]
    assert "0.21" in importance_response["text"]
    assert "0.10" in importance_response["text"]
    assert "24.57%" in predict_response["text"]
    assert "75.43%" in predict_response["text"]
