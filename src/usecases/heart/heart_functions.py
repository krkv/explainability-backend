"""Heart use case functions refactored from instances/heart/executive.py."""

import copy
import json
import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tabulate import tabulate
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class HeartFunctions:
    """Heart use case functions with dependencies injected via constructor."""
    
    def __init__(
        self,
        model,
        dataset: pd.DataFrame,
        dataset_full: pd.DataFrame,
        y_values: pd.Series,
        explainer,
        dice_exp,
        dice_dataset: pd.DataFrame,
        model_metadata: Dict[str, Any],
        feature_metadata: Dict[str, Any],
        alias_lookup: Dict[str, str],
        global_feature_importances: Dict[str, float],
        target_variable: str,
        class_names: List[str],
        feature_names: List[str],
        functions_catalog: Optional[List[Dict[str, Any]]] = None,
        shap_cache_path: str = None,
        cf_cache_path: str = None,
    ):
        """
        Initialize heart functions with required dependencies.
        
        Args:
            model: Trained model for heart disease prediction
            dataset: Dataset DataFrame (without target column)
            dataset_full: Full dataset DataFrame (with target column)
            y_values: Target variable Series
            explainer: SHAP explainer instance
            dice_exp: DiCE explainer instance
            dice_dataset: Dataset for DiCE (without prediction column)
            model_metadata: Model metadata dictionary
            feature_metadata: Feature metadata dictionary
            alias_lookup: Dictionary mapping feature aliases to feature names
            global_feature_importances: Dictionary of global feature importances
            target_variable: Name of target variable
            class_names: List of class names
            feature_names: List of feature names
            shap_cache_path: Path to SHAP cache pickle
            cf_cache_path: Path to Counterfactuals cache pickle
        """
        self.model = model
        self.dataset = dataset
        self.dataset_full = dataset_full
        self.y_values = y_values
        self.explainer = explainer
        self.dice_exp = dice_exp
        self.dice_dataset = dice_dataset
        self.model_metadata = model_metadata
        self.feature_metadata = feature_metadata
        self.alias_lookup = alias_lookup
        self.global_feature_importances = global_feature_importances
        self.target_variable = target_variable
        self.class_names = class_names
        self.feature_names = feature_names
        self.functions_catalog = functions_catalog or []
        self.shap_cache_path = shap_cache_path
        self.cf_cache_path = cf_cache_path
        
        self._shap_cache = self._load_pickle(self.shap_cache_path)
        self._cf_cache = self._load_pickle(self.cf_cache_path)

    def available_functions(self) -> Dict[str, Any]:
        """Return a formatted list of the available heart use case functions."""
        function_items = []

        for entry in self.functions_catalog:
            function = entry.get("function", {})
            name = function.get("name")
            description = function.get("description", "")
            if not name:
                continue

            properties = function.get("parameters", {}).get("properties", {})
            required = set(function.get("parameters", {}).get("required", []))
            signature_parts = []
            parameter_details = []

            for param_name, param_schema in properties.items():
                signature_parts.append(
                    f"{param_name}=..."
                    if param_name in required else f"{param_name}=optional"
                )
                parameter_details.append(
                    {
                        "name": param_name,
                        "required": param_name in required,
                        "description": param_schema.get("description", ""),
                    }
                )

            signature = f"{name}({', '.join(signature_parts)})" if signature_parts else f"{name}()"
            function_items.append(
                {
                    "name": name,
                    "signature": signature,
                    "description": description,
                    "parameters": parameter_details,
                }
            )

        text = "<p>Here are the available functions:</p><ul>"
        for item in function_items:
            text += f"<li><code>{item['signature']}</code>: {item['description']}</li>"
        text += "</ul><p>Let me know which one you want to use.</p>"

        return {
            "data": {"available_functions": function_items},
            "text": text,
        }

    def _load_pickle(self, path):
        if path:
            import os, pickle
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache from {path}: {e}")
        return {}

    def _save_pickle(self, path, data):
        if path:
            import os, pickle
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to save cache to {path}: {e}")
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Returns the exact training hyperparameters of the model."""
        if "parameters" in self.model_metadata:
            headers = ["Parameter", "Value"]
            table = [
                [param, str(value)]
                for param, value in self.model_metadata["parameters"].items()
            ]
            return {
                "data": self.model_metadata["parameters"],
                "text": "<p>Model training hyperparameters are:</p>" + tabulate(table, headers, tablefmt='html')
            }
        return {"error": "Model parameters not found in metadata."}
    
    def get_model_description(self) -> Dict[str, Any]:
        """Returns a general description of the model architecture and its purpose."""
        if "description" in self.model_metadata:
            return {
                "data": self.model_metadata["description"],
                "text": f"<p>Model description is: {self.model_metadata['description']}</p>"
            }
        return {"error": "Model description not found in metadata."}
    
    def predict(self, patient_id: int) -> Dict[str, Any]:
        """Predict heart disease risk for a specific patient by ID."""
        if patient_id not in self.dataset.index:
            return {
                "error": f"Patient ID {patient_id} not found in the dataset.",
                "text": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset."
            }
        
        patient_row = self.dataset.loc[patient_id].to_frame().T
        prediction = self.model.predict(patient_row)[0]
        probabilities = self.model.predict_proba(patient_row)[0].tolist()
        
        prob_neg = probabilities[0]
        prob_pos = probabilities[1]
        
        text = f"<p>Patient <code>ID</code> <var>{patient_id}</var> has a predicted risk of heart disease: <var>{self.class_names[prediction]}</var>.</p> "
        text += f"<p>The prediction class probabilities are: {self.class_names[1].lower()} {int(round(prob_pos*100))}%, {self.class_names[0].lower()} {int(round(prob_neg*100))}%.</p>"
        
        return {
            "data": {
                "patient_id": patient_id,
                "prediction": int(prediction),
                "probabilities": probabilities
            },
            "text": text
        }
    
    def feature_importance_patient(self, patient_id: int) -> Dict[str, Any]:
        """Returns SHAP-based feature importance scores (patient-specific)."""
        if patient_id not in self.dataset.index:
            return {
                "error": f"Patient ID {patient_id} not found in the dataset.",
                "text": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset."
            }
        
        if patient_id in self._shap_cache:
            cache_entry = self._shap_cache[patient_id]
            influences = cache_entry[0] if isinstance(cache_entry, tuple) else cache_entry
        else:
            patient_row = self.dataset.loc[patient_id].to_frame().T
            shap_values = self.explainer.shap_values(patient_row, nsamples=10_000, silent=True)
            influences = shap_values.squeeze().tolist()
            
            self._shap_cache[patient_id] = influences
            self._save_pickle(self.shap_cache_path, self._shap_cache)

        labeled_influences = self._label_feature_scores(influences)
        result = pd.DataFrame(labeled_influences.items(), columns=['Feature', 'Importance']).sort_values(
            by='Importance', ascending=False
        )
        text = (
            f"<p>For the patient with <code>ID</code> <var>{patient_id}</var> the feature importances are:</p>"
            + self._table_html(result)
        )
        
        return {
            "data": {"patient_id": patient_id, "feature_importance": labeled_influences},
            "text": text
        }
    
    def feature_importance_global(self) -> Dict[str, Any]:
        """Returns SHAP-based feature importance scores (global)."""
        labeled_importances = {
            self._feature_label(feature): importance
            for feature, importance in self.global_feature_importances.items()
        }
        result_html = self._table_html(pd.DataFrame(labeled_importances.items(), columns=['Feature', 'Importance']))
        text = "<p>Global feature importances based on SHAP values are:</p>" + result_html
        
        return {
            "data": {"global_feature_importance": labeled_importances},
            "text": text
        }
    
    def dataset_summary(self, patient_id: Optional[int] = None) -> Dict[str, Any]:
        """Return a single table with overall dataset statistics."""
        stats_rows = self._build_dataset_statistics()
        stats_columns = [
            "Feature",
            "Type",
            "Count",
            "Mean / Mode",
            "Std",
            "Min",
            "25%",
            "50%",
            "75%",
            "Max",
            "Categories",
        ]
        stats_table = pd.DataFrame(stats_rows, columns=stats_columns)
        result = {"dataset_statistics": stats_rows}

        text = "<p>Here are the overall dataset statistics for each feature:</p>"
        text += self._table_html(stats_table)

        if patient_id is not None:
            try:
                patient_features = self._display_record(self.dataset.loc[patient_id].to_dict())
                result["patient"] = {
                    "patient_id": patient_id,
                    "features": patient_features,
                }
                text += f"<p>Patient <code>ID</code> <var>{patient_id}</var> features:</p>" + self._dict_table_html(patient_features)
            except KeyError:
                return {
                    "error": f"Patient <code>ID</code> <var>{patient_id}</var> is out of range. Dataset has <var>{len(self.dataset)}</var> patients.",
                    "text": f"<p>Patient <code>ID</code> <var>{patient_id}</var> is out of range. Dataset has <var>{len(self.dataset)}</var> patients.</p>",
                }

        return {"data": result, "text": text}

    def define_feature(self, feature: str) -> Dict[str, Any]:
        """Return the metadata-backed definition for a feature."""
        canonical_feature = self._resolve_feature_name(feature)
        if canonical_feature is None:
            available_features = ", ".join(
                f"<code>{self._feature_label(name)}</code>" for name in self.dataset.columns
            )
            return {
                "error": f"Feature '{feature}' not found in metadata.",
                "text": (
                    f"<p>I could not find a feature matching <code>{feature}</code>.</p>"
                    f"<p>Available features are: {available_features}.</p>"
                ),
            }

        metadata = self.feature_metadata.get(canonical_feature, {})
        display_name = metadata.get("display_name", canonical_feature)
        description = metadata.get("description", "No description available.")
        unit = metadata.get("unit")
        kind = metadata.get("kind", "continuous")
        aliases = metadata.get("aliases", [])
        categories = metadata.get("categories", {})

        category_rows = [
            {"Code": code, "Label": details.get("label", code)}
            for code, details in self._sorted_categories(categories)
        ]

        data = {
            "feature": canonical_feature,
            "display_name": display_name,
            "description": description,
            "unit": unit,
            "kind": kind,
            "aliases": aliases,
            "categories": {
                code: details.get("label", code)
                for code, details in self._sorted_categories(categories)
            },
        }

        text = f"<p><code>{display_name}</code>: {description}</p>"
        if unit:
            text += f"<p>Unit: <var>{unit}</var>.</p>"
        if aliases:
            text += "<p>Also referred to as: " + ", ".join(
                f"<code>{alias}</code>" for alias in aliases
            ) + ".</p>"
        if category_rows:
            text += "<p>Available categories are:</p>"
            text += self._table_html(pd.DataFrame(category_rows, columns=["Code", "Label"]))

        return {"data": data, "text": text}
    
    def performance_metrics(self, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Computes and returns selected performance metrics, including AUC-ROC."""
        y_pred = self.model.predict(self.dataset)
        
        try:
            y_prob = self.model.predict_proba(self.dataset)
        except AttributeError:
            y_prob = None
        
        n_classes = len(np.unique(self.y_values))
        
        try:
            if y_prob is not None:
                if n_classes == 2:
                    y_score = y_prob[:, 1]
                    auc = roc_auc_score(self.y_values, y_score)
                else:
                    auc = roc_auc_score(self.y_values, y_prob, multi_class="ovr")
            else:
                auc = 0.0
        except Exception:
            auc = 0.0
        
        all_metrics = {
            "accuracy": float(accuracy_score(self.y_values, y_pred)),
            "precision": float(precision_score(self.y_values, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(self.y_values, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(self.y_values, y_pred, average="weighted", zero_division=0)),
            "auc_roc": float(auc)
        }
        
        metric_mapping = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1_score",
            "f1_score": "f1_score",
            "f1-score": "f1_score",
            "auc": "auc_roc",
            "auc_roc": "auc_roc",
            "auc-roc": "auc_roc"
        }
        
        if metrics:
            normalized_metrics = []
            for m in metrics:
                key = m.strip().lower().replace("-", "_")
                if key in metric_mapping:
                    normalized_metrics.append(metric_mapping[key])
                else:
                    return {
                        "error": f"Metric <code>{m}</code> is not recognized. Valid metrics are: <code>{', '.join(metric_mapping.keys())}</code>"
                    }
            data = {key: all_metrics[key] for key in normalized_metrics}
            text = "<p>Selected performance metrics are:</p>" + self._dict_table_html(data, key_header="Metric")
            return {"text": text, "data": data}
        else:
            data = all_metrics
            text = "<p>All performance metrics are:</p>" + self._dict_table_html(data, key_header="Metric")
            return {"text": text, "data": data}
    
    def confusion_matrix_stats(self) -> Dict[str, Any]:
        """Returns the confusion matrix with counts of TN, FP, FN, TP."""
        y_pred = self.model.predict(self.dataset)
        cm = confusion_matrix(self.y_values, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            data = {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            }
            text = f"<p>Confusion matrix statistics:</p>" + self._dict_table_html(data, key_header="Statistic", value_header="Count")
            return {"text": text, "data": data}
        else:
            data = {
                "confusion_matrix": cm.tolist(),
                "classes": self.class_names
            }
            text = f"<p>Confusion matrix for multi-class classification:</p>" + tabulate(cm, headers=self.class_names, tablefmt='html', numalign="left")
            return {"text": text, "data": data}
    
    def what_if(self, patient_id: int, feature: str, value_change: Any) -> Dict[str, Any]:
        """Simulates how changing a single feature affects predictions."""
        feature_key = self._resolve_feature_name(feature)
        
        if patient_id not in self.dataset.index:
            return {
                "error": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset.",
                "text": f"<p>Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset.</p>",
            }
        
        patient_row = self.dataset.loc[patient_id].to_frame().T
        
        if feature_key is None or feature_key not in patient_row.columns:
            return {
                "error": f"Feature <code>{feature}</code> not found in patient data.",
                "text": f"<p>Feature <code>{feature}</code> not found in patient data.</p>",
            }
        
        modified_row = patient_row.copy()
        current_value = patient_row[feature_key].iloc[0]
        feature_label = self._feature_label(feature_key)
        feature_kind = self.feature_metadata.get(feature_key, {}).get("kind", "continuous")

        if feature_kind == "categorical" and not isinstance(value_change, (int, float, np.integer, np.floating)):
            resolved_value = self._resolve_category_value(feature_key, value_change)
            if resolved_value is None:
                available_categories = ", ".join(
                    f"<code>{details.get('label', code)}</code>"
                    for code, details in self._sorted_categories(
                        self.feature_metadata.get(feature_key, {}).get("categories", {})
                    )
                )
                return {
                    "error": (
                        f"Category <code>{value_change}</code> is not recognized for feature "
                        f"<code>{feature_label}</code>."
                    ),
                    "text": (
                        f"<p>I could not match <code>{value_change}</code> to a valid category for "
                        f"<code>{feature_label}</code>.</p>"
                        f"<p>Available categories are: {available_categories}.</p>"
                    ),
                }
            modified_row[feature_key] = resolved_value
        else:
            if not isinstance(value_change, (int, float, np.integer, np.floating)):
                return {
                    "error": (
                        f"Feature <code>{feature_label}</code> requires a numeric change, "
                        f"but received <code>{value_change}</code>."
                    ),
                    "text": (
                        f"<p>Feature <code>{feature_label}</code> requires a numeric change, "
                        f"but received <code>{value_change}</code>.</p>"
                    ),
                }
            modified_row[feature_key] += value_change
        
        original_prediction = int(self.model.predict(patient_row)[0])
        new_prediction = int(self.model.predict(modified_row)[0])
        
        original_prob = self.model.predict_proba(patient_row)[0].tolist()
        new_prob = self.model.predict_proba(modified_row)[0].tolist()

        original_value = self._display_value(feature_key, current_value)
        new_value = self._display_value(feature_key, modified_row[feature_key].iloc[0])
        
        data = {
            "patient_id": patient_id,
            "feature_modified": feature_label,
            "value_change": value_change,
            "original_value": original_value,
            "new_value": new_value,
            "original_prediction": original_prediction,
            "new_prediction": new_prediction,
            "probability_change": {
                "original": original_prob,
                "new": new_prob
            }
        }
        if feature_kind == "categorical" and not isinstance(value_change, (int, float, np.integer, np.floating)):
            text = (
                f"<p>For patient <code>ID</code> <var>{patient_id}</var>, changing "
                f"<code>{feature_label}</code> from <var>{original_value}</var> to "
                f"<var>{new_value}</var> results in:</p>"
            )
        else:
            text = (
                f"<p>For patient <code>ID</code> <var>{patient_id}</var>, modifying feature "
                f"<code>{feature_label}</code> by <var>{value_change}</var> results in:</p>"
            )
        text += f"<p>Original prediction: <var>{self.class_names[original_prediction]}</var> with probabilities: <var>{[round(prob, 2) for prob in original_prob]}</var></p>"
        text += f"<p>New prediction: <var>{self.class_names[new_prediction]}</var> with probabilities: <var>{[round(prob, 2) for prob in new_prob]}</var></p>"
        
        return {"data": data, "text": text}
    
    def counterfactual(self, patient_id: int) -> Dict[str, Any]:
        """Generates counterfactual explanations using DiCE."""
        if patient_id not in self.dataset.index:
            return {"error": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset."}
        
        if patient_id in self._cf_cache:
            cache_entry = self._cf_cache[patient_id]
            if isinstance(cache_entry, tuple):
                cache_entry = cache_entry[0]
            counterfactual_records = cache_entry.get("counterfactuals_model", cache_entry.get("counterfactuals", []))
            new_predictions = [int(p) for p in cache_entry["new_predictions"]]
            data = {
                "patient_id": patient_id,
                "original_prediction": int(cache_entry["original_prediction"]),
                "counterfactuals": [self._display_record(record) for record in counterfactual_records],
                "new_predictions": new_predictions,
            }
            output_string = self._build_counterfactual_text(
                patient_id=patient_id,
                original_prediction=int(cache_entry["original_prediction"]),
                counterfactual_records=counterfactual_records,
                new_predictions=new_predictions,
            )
            return {"data": data, "text": output_string}

        original_prediction = self.model.predict(self.dice_dataset.loc[[patient_id]])[0]
        
        cfe = self.dice_exp.generate_counterfactuals(
            query_instances=self.dice_dataset.loc[[patient_id]],
            total_CFs=10,
            desired_class="opposite"
        )
        
        final_cfes = cfe.cf_examples_list[0].final_cfs_df
        final_cfe_ids = list(final_cfes.index)
        if 'prediction' in final_cfes.columns:
            final_cfes.pop('prediction')
        
        new_predictions = self.model.predict(final_cfes)
        original_instance = self.dice_dataset.loc[[patient_id]]
        
        counterfactual_records = final_cfes.to_dict(orient='records')
        output_string = self._build_counterfactual_text(
            patient_id=patient_id,
            original_prediction=int(original_prediction),
            counterfactual_records=counterfactual_records,
            new_predictions=[int(p) for p in new_predictions.tolist()],
        )

        display_counterfactuals = [
            self._display_record(record)
            for record in counterfactual_records
        ]
        
        data = {
            "patient_id": patient_id,
            "original_prediction": int(original_prediction),
            "counterfactuals": display_counterfactuals,
            "new_predictions": [int(p) for p in new_predictions.tolist()]
        }
        
        self._cf_cache[patient_id] = {
            "patient_id": patient_id,
            "original_prediction": int(original_prediction),
            "counterfactuals_model": counterfactual_records,
            "new_predictions": [int(p) for p in new_predictions.tolist()],
        }
        self._save_pickle(self.cf_cache_path, self._cf_cache)
        
        return {"data": data, "text": output_string}
    
    def misclassified_cases(self) -> Dict[str, Any]:
        """Identifies frequent misclassifications and extracts common feature patterns."""
        y_pred = self.model.predict(self.dataset)
        
        df_copy = self.dataset.copy()
        df_copy["predicted"] = y_pred
        df_copy["misclassified"] = df_copy["predicted"] != self.y_values
        
        misclassified_df = df_copy[df_copy["misclassified"]]
        correctly_classified_df = df_copy[~df_copy["misclassified"]]
        
        misclassified_summary = self._summarize_group(
            misclassified_df.drop(columns=["predicted", "misclassified"], errors="ignore")
        ) if not misclassified_df.empty else {}
        correctly_classified_summary = self._summarize_group(
            correctly_classified_df.drop(columns=["predicted", "misclassified"], errors="ignore")
        ) if not correctly_classified_df.empty else {}

        data = {
            "false_positives": int(((y_pred == 1) & (self.y_values == 0)).sum()),
            "false_negatives": int(((y_pred == 0) & (self.y_values == 1)).sum()),
            "feature_distribution": {
                "misclassified_cases": misclassified_summary,
                "correctly_classified_cases": correctly_classified_summary,
            }
        }
        
        text = "<p>Misclassified cases statistics:</p>"
        text += f"<p>False Positives: <var>{data['false_positives']}</var></p>"
        text += f"<p>False Negatives: <var>{data['false_negatives']}</var></p>"
        text += "<p>Feature distribution for misclassified cases:</p>" + self._dict_table_html(data["feature_distribution"]["misclassified_cases"])
        text += "<p>Feature distribution for correctly classified cases:</p>" + self._dict_table_html(data["feature_distribution"]["correctly_classified_cases"])
        
        return {"data": data, "text": text}
    
    def age_group_performance(self) -> Dict[str, Any]:
        """Computes model performance across different age groups."""
        if "age" not in self.dataset.columns:
            return {"error": f"Required column <code>age</code> is missing from dataset."}
        
        y_pred = pd.Series(self.model.predict(self.dataset), index=self.dataset.index)
        
        age_groups = {
            "<40": self.dataset[self.dataset["age"] < 40],
            "40-60": self.dataset[(self.dataset["age"] >= 40) & (self.dataset["age"] <= 60)],
            ">60": self.dataset[self.dataset["age"] > 60]
        }
        
        results = {}
        for group, subset in age_groups.items():
            if not subset.empty:
                y_true_group = self.y_values[subset.index]
                y_pred_group = y_pred.loc[subset.index]
                
                results[group] = {
                    "accuracy": float(accuracy_score(y_true_group, y_pred_group)),
                    "precision": float(precision_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
                    "recall": float(recall_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
                    "f1_score": float(f1_score(y_true_group, y_pred_group, average="weighted", zero_division=0))
                }
        
        text = "<p>Model performance across age groups:</p>"
        headers = ["Age Group", "Accuracy", "Precision", "Recall", "F1 Score"]
        table = [
            [group, results[group]["accuracy"], results[group]["precision"], results[group]["recall"], results[group]["f1_score"]]
            for group in results
        ]
        text += tabulate(table, headers=headers, tablefmt='html', numalign="left")
        
        return {"data": results, "text": text}
    
    def feature_interactions(self) -> Dict[str, Any]:
        """Computes feature interactions based on correlation analysis."""
        df = self.dataset.select_dtypes(include=[np.number])
        
        correlation_matrix = df.corr()
        feature_names = correlation_matrix.columns.tolist()
        
        interaction_dict = {
            f"{self._feature_label(feature_names[i])} & {self._feature_label(feature_names[j])}": float(correlation_matrix.iloc[i, j])
            for i in range(len(feature_names)) for j in range(i + 1, len(feature_names))
        }
        
        top_interactions = dict(sorted(interaction_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        
        data = {"top_feature_interactions": top_interactions}
        text = "<p>Top feature interactions based on correlation analysis:</p>" + self._dict_table_html(top_interactions, key_header="Feature", value_header="Correlation")
        return {"data": data, "text": text}
    
    def count_all(self) -> Dict[str, Any]:
        """Count all instances in the dataset."""
        count = len(self.dataset)
        return {
            "data": {"count": count},
            "text": f"<p>There are <var>{count}</var> instances in the dataset.</p>"
        }

    def show_ids(self) -> Dict[str, Any]:
        """Show all available IDs."""
        ids = self.dataset.index.get_level_values(0).unique().sort_values().tolist()
        ids_str = ', '.join([f'<var>{id}</var>' for id in ids])
        return {
            "data": {"ids": ids},
            "text": f"<p>Available <code>ID</code> values are: {ids_str}.</p>"
        }
    
    def show_one(self, patient_id: int) -> Dict[str, Any]:
        """Show data for a specific ID."""
        if patient_id not in self.dataset.index:
            return {
                "error": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset.",
                "text": f"<p>There is no data for <code>ID</code> <var>{patient_id}</var>.</p>"
            }
        intro = f"<p>Here is the data for <code>ID</code> <var>{patient_id}</var>:</p>"
        patient_data = self._display_record(self.dataset.loc[patient_id].to_dict())
        framed = pd.DataFrame(patient_data.items(), columns=["Feature", "Value"])
        table = self._table_html(framed)
        return {
            "data": {"patient_data": patient_data},
            "text": intro + table
        }
    
    # Helper methods

    def _feature_label(self, feature: str) -> str:
        metadata = self.feature_metadata.get(feature, {})
        return metadata.get("display_name", feature)

    def _display_value(self, feature: str, value: Any) -> Any:
        if pd.isna(value):
            return value

        categories = self.feature_metadata.get(feature, {}).get("categories", {})
        possible_keys = [str(value)]
        if isinstance(value, (int, np.integer)):
            possible_keys.insert(0, str(int(value)))
        elif isinstance(value, (float, np.floating)):
            if float(value).is_integer():
                possible_keys.insert(0, str(int(value)))
            value = round(float(value), 3)

        for key in possible_keys:
            if key in categories:
                return categories[key].get("label", key)
        return value

    def _display_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            self._feature_label(feature): self._display_value(feature, value)
            for feature, value in record.items()
        }

    def _resolve_feature_name(self, feature: str) -> Optional[str]:
        if not isinstance(feature, str):
            return None
        normalized = feature.strip().lower()
        if not normalized:
            return None
        if normalized in self.alias_lookup:
            return self.alias_lookup[normalized]
        if normalized in self.feature_metadata:
            return normalized
        return None

    def _resolve_category_value(self, feature: str, value: Any) -> Optional[Any]:
        categories = self.feature_metadata.get(feature, {}).get("categories", {})
        normalized_value = self._normalize_lookup_value(value)
        if not normalized_value:
            return None
        value_tokens = normalized_value.split()

        exact_matches: List[str] = []
        partial_matches: List[str] = []

        for code, details in self._sorted_categories(categories):
            label = details.get("label", code)
            normalized_code = self._normalize_lookup_value(code)
            normalized_label = self._normalize_lookup_value(label)

            if normalized_value in {normalized_code, normalized_label}:
                exact_matches.append(code)
                continue

            label_tokens = normalized_label.split()
            if value_tokens and all(token in label_tokens for token in value_tokens):
                partial_matches.append(code)

        if len(exact_matches) == 1:
            return self._coerce_category_code(feature, exact_matches[0])
        if len(partial_matches) == 1:
            return self._coerce_category_code(feature, partial_matches[0])
        return None

    def _coerce_category_code(self, feature: str, code: str) -> Any:
        series = self.dataset[feature]
        if pd.api.types.is_integer_dtype(series.dtype):
            try:
                return int(float(code))
            except (TypeError, ValueError):
                return code
        if pd.api.types.is_float_dtype(series.dtype):
            try:
                return float(code)
            except (TypeError, ValueError):
                return code
        return code

    def _normalize_lookup_value(self, value: Any) -> str:
        normalized = str(value).strip().lower()
        normalized = normalized.replace("_", " ").replace("-", " ")
        return re.sub(r"\s+", " ", normalized)

    def _dict_table_html(self, record: Dict[str, Any], key_header: str = "Feature", value_header: str = "Value") -> str:
        frame = pd.DataFrame(record.items(), columns=[key_header, value_header])
        return self._table_html(frame)

    def _table_html(self, frame: pd.DataFrame) -> str:
        return f"<p>{frame.to_html(index=False)}</p>"

    def _build_dataset_statistics(self) -> List[Dict[str, Any]]:
        stats_rows = []
        for feature in self.dataset.columns:
            metadata = self.feature_metadata.get(feature, {})
            kind = metadata.get("kind", "continuous")
            series = self.dataset[feature].dropna()
            row = {
                "Feature": self._feature_label(feature),
                "Type": kind,
                "Count": int(series.count()),
                "Mean / Mode": "",
                "Std": "",
                "Min": "",
                "25%": "",
                "50%": "",
                "75%": "",
                "Max": "",
                "Categories": "",
            }

            if kind == "continuous":
                row.update(
                    {
                        "Mean / Mode": round(float(series.mean()), 3),
                        "Std": round(float(series.std()), 3) if len(series) > 1 else 0.0,
                        "Min": round(float(series.min()), 3),
                        "25%": round(float(series.quantile(0.25)), 3),
                        "50%": round(float(series.quantile(0.5)), 3),
                        "75%": round(float(series.quantile(0.75)), 3),
                        "Max": round(float(series.max()), 3),
                    }
                )
            else:
                modes = series.mode(dropna=True)
                categories = metadata.get("categories", {})
                row["Mean / Mode"] = self._display_value(
                    feature,
                    modes.iloc[0] if not modes.empty else None,
                )
                row["Categories"] = ", ".join(
                    details.get("label", code)
                    for code, details in self._sorted_categories(categories)
                )

            stats_rows.append(row)
        return stats_rows

    def _summarize_group(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """Summarize a cohort with readable feature-level aggregates."""
        summary: Dict[str, Any] = {}

        for feature in dataframe.columns:
            metadata = self.feature_metadata.get(feature, {})
            kind = metadata.get("kind", "continuous")
            series = dataframe[feature].dropna()

            if series.empty:
                continue

            if kind == "continuous":
                value: Any = round(float(series.mean()), 3)
            else:
                modes = series.mode(dropna=True)
                value = modes.iloc[0] if not modes.empty else None
                value = self._display_value(feature, value)

            summary[self._feature_label(feature)] = value

        return summary

    def _sorted_categories(self, categories: Dict[str, Any]) -> List[Any]:
        return sorted(categories.items(), key=lambda item: self._category_sort_key(item[0]))

    def _category_sort_key(self, value: str) -> Any:
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))

    def _label_feature_scores(self, influences: List[float]) -> Dict[str, float]:
        return {
            self._feature_label(feature): float(influences[index])
            for index, feature in enumerate(self.dataset.columns)
        }

    def _build_counterfactual_text(
        self,
        patient_id: int,
        original_prediction: int,
        counterfactual_records: List[Dict[str, Any]],
        new_predictions: List[int],
    ) -> str:
        output_string = f"<p>The original prediction for the data sample with <code>ID</code> <var>{patient_id}</var> is <samp>{self.class_names[original_prediction]}</samp>.</p>"
        output_string += "<p>Here are some options to change the prediction of this instance.</p><ul>"
        transition_words = ["Further,", "Also,", "In addition,", "Furthermore,"]
        original_instance = self.dice_dataset.loc[[patient_id]]

        for i, counterfactual_record in enumerate(counterfactual_records[:3]):
            prefix = "First, if you" if i == 0 else f"{random.choice(transition_words)} if you"
            counterfactual_frame = pd.DataFrame([counterfactual_record])
            output_string += f"<li>{prefix}{self._get_change_string(counterfactual_frame, original_instance)}, the model will predict <samp>{self.class_names[new_predictions[i]]}</samp>.</li>"
        output_string += "</ul>"
        return output_string
    
    def _get_change_string(self, cfe: pd.DataFrame, original_instance: pd.DataFrame) -> str:
        """Build a string describing changes between counterfactual and original instance."""
        cfe_features = list(cfe.columns)
        original_features = list(original_instance.columns)
        assert set(cfe_features) == set(original_features), "CFE features and Original Instance features are different!"
        
        change_string = ""
        for feature in cfe_features:
            orig_f = original_instance[feature].values[0]
            cfe_f = cfe[feature].values[0]
            
            if isinstance(cfe_f, str):
                cfe_f = float(cfe_f)
            
            if orig_f != cfe_f:
                if cfe_f > orig_f:
                    inc_dec = " increase"
                else:
                    inc_dec = " decrease"
                display_value = self._display_value(feature, cfe_f)
                change_string += f"{inc_dec} <code>{self._feature_label(feature)}</code> to <var>{display_value}</var>"
                change_string += " and "
        
        if change_string.endswith(" and "):
            change_string = change_string[:-5]
        return change_string
