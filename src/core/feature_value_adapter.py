"""Utilities for translating between model-space and display-space feature values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import pandas as pd


@dataclass(frozen=True)
class FeatureCategory:
    """Human-readable metadata for a categorical feature value."""

    label: str


class FeatureValueAdapter:
    """Translate feature values between model-facing and user-facing representations."""

    def __init__(self, feature_metadata: Dict[str, Dict[str, Any]]):
        self.feature_metadata = feature_metadata

    def get_feature_metadata(self, feature: str) -> Dict[str, Any]:
        return self.feature_metadata.get(feature, {})

    def get_display_name(self, feature: str) -> str:
        metadata = self.get_feature_metadata(feature)
        return metadata.get("display_name", feature)

    def get_unit(self, feature: str) -> str | None:
        metadata = self.get_feature_metadata(feature)
        return metadata.get("unit")

    def get_kind(self, feature: str) -> str:
        metadata = self.get_feature_metadata(feature)
        return metadata.get("kind", "continuous")

    def to_display(self, feature: str, model_value: Any) -> Any:
        if pd.isna(model_value):
            return None

        metadata = self.get_feature_metadata(feature)
        transformed = self._transform_to_display(model_value, metadata.get("transform", {}))
        categories = metadata.get("categories", {})
        category_metadata = self._get_category_metadata(categories, transformed)
        if category_metadata is not None:
            return category_metadata.label
        return transformed

    def to_model(self, feature: str, display_value: Any) -> Any:
        if pd.isna(display_value):
            return None

        metadata = self.get_feature_metadata(feature)
        encoded_value = self._encode_category_value(display_value, metadata.get("categories", {}))
        return self._transform_to_model(encoded_value, metadata.get("transform", {}))

    def delta_to_model(self, feature: str, display_delta: float) -> float:
        metadata = self.get_feature_metadata(feature)
        kind = metadata.get("kind", "continuous")
        if kind != "continuous":
            raise ValueError(f"Feature '{feature}' does not support delta-based changes.")

        transform = metadata.get("transform", {})
        transform_type = transform.get("type", "identity")
        if transform_type == "identity":
            return float(display_delta)
        if transform_type == "min_max":
            raw_min = transform.get("raw_min")
            raw_max = transform.get("raw_max")
            model_min = transform.get("model_min", 0.0)
            model_max = transform.get("model_max", 1.0)
            if raw_min is None or raw_max is None:
                raise ValueError(f"Feature '{feature}' is missing min-max transform parameters.")
            raw_span = raw_max - raw_min
            if raw_span == 0:
                raise ValueError(f"Feature '{feature}' has zero raw span for min-max transform.")
            model_span = model_max - model_min
            return float(display_delta) * float(model_span) / float(raw_span)
        if transform_type == "standard":
            std = transform.get("std")
            if std in (None, 0):
                raise ValueError(f"Feature '{feature}' is missing standard transform parameters.")
            return float(display_delta) / float(std)
        raise ValueError(f"Unsupported transform type '{transform_type}' for feature '{feature}'.")

    def to_display_frame(self, dataframe: pd.DataFrame, features: Iterable[str] | None = None) -> pd.DataFrame:
        display_frame = dataframe.copy()
        feature_set = set(features) if features is not None else set(display_frame.columns)
        for feature in display_frame.columns:
            if feature in feature_set:
                display_frame[feature] = display_frame[feature].apply(lambda value: self.to_display(feature, value))
        return display_frame

    def rename_columns_for_display(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.rename(columns={column: self.get_display_name(column) for column in dataframe.columns})

    def build_feature_catalog(self, features: Iterable[str]) -> list[Dict[str, Any]]:
        catalog = []
        for feature in features:
            metadata = self.get_feature_metadata(feature)
            categories = metadata.get("categories", {})
            category_labels = {
                value: category.get("label", value)
                for value, category in categories.items()
            }
            catalog.append(
                {
                    "feature": feature,
                    "display_name": self.get_display_name(feature),
                    "description": metadata.get("description", ""),
                    "kind": metadata.get("kind", "continuous"),
                    "unit": metadata.get("unit"),
                    "aliases": metadata.get("aliases", []),
                    "categories": category_labels,
                }
            )
        return catalog

    def _transform_to_display(self, model_value: Any, transform: Dict[str, Any]) -> Any:
        transform_type = transform.get("type", "identity")
        if transform_type == "identity":
            return model_value
        if transform_type == "min_max":
            raw_min = transform.get("raw_min")
            raw_max = transform.get("raw_max")
            model_min = transform.get("model_min", 0.0)
            model_max = transform.get("model_max", 1.0)
            if raw_min is None or raw_max is None:
                raise ValueError("Missing min-max transform parameters.")
            model_span = model_max - model_min
            if model_span == 0:
                raise ValueError("Model span is zero for min-max transform.")
            raw_span = raw_max - raw_min
            return raw_min + ((float(model_value) - float(model_min)) * raw_span / float(model_span))
        if transform_type == "standard":
            mean = transform.get("mean")
            std = transform.get("std")
            if mean is None or std is None:
                raise ValueError("Missing standard transform parameters.")
            return (float(model_value) * float(std)) + float(mean)
        raise ValueError(f"Unsupported transform type '{transform_type}'.")

    def _transform_to_model(self, display_value: Any, transform: Dict[str, Any]) -> Any:
        transform_type = transform.get("type", "identity")
        if transform_type == "identity":
            return display_value
        if transform_type == "min_max":
            raw_min = transform.get("raw_min")
            raw_max = transform.get("raw_max")
            model_min = transform.get("model_min", 0.0)
            model_max = transform.get("model_max", 1.0)
            if raw_min is None or raw_max is None:
                raise ValueError("Missing min-max transform parameters.")
            raw_span = raw_max - raw_min
            if raw_span == 0:
                raise ValueError("Raw span is zero for min-max transform.")
            model_span = model_max - model_min
            return model_min + ((float(display_value) - float(raw_min)) * model_span / float(raw_span))
        if transform_type == "standard":
            mean = transform.get("mean")
            std = transform.get("std")
            if mean is None or std is None:
                raise ValueError("Missing standard transform parameters.")
            if std == 0:
                raise ValueError("Standard deviation cannot be zero for standard transform.")
            return (float(display_value) - float(mean)) / float(std)
        raise ValueError(f"Unsupported transform type '{transform_type}'.")

    def _get_category_metadata(self, categories: Dict[str, Dict[str, Any]], value: Any) -> FeatureCategory | None:
        if not categories:
            return None

        possible_keys = [str(value)]
        if isinstance(value, float) and value.is_integer():
            possible_keys.insert(0, str(int(value)))
        for key in possible_keys:
            if key in categories:
                return FeatureCategory(label=categories[key].get("label", key))
        return None

    def _encode_category_value(self, display_value: Any, categories: Dict[str, Dict[str, Any]]) -> Any:
        if not categories:
            return display_value

        for raw_value, category in categories.items():
            if display_value == category.get("label"):
                if self._looks_like_number(raw_value):
                    numeric_value = float(raw_value)
                    return int(numeric_value) if numeric_value.is_integer() else numeric_value
                return raw_value
        return display_value

    @staticmethod
    def _looks_like_number(value: str) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
