"""Heart use case functions refactored from instances/heart/executive.py."""

import random
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tabulate import tabulate

from src.core.feature_value_adapter import FeatureValueAdapter
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class HeartFunctions:
    """Heart use case functions with dependencies injected via constructor."""

    def __init__(
        self,
        model,
        dataset: pd.DataFrame,
        dataset_full: pd.DataFrame,
        display_dataset: pd.DataFrame,
        display_dataset_full: pd.DataFrame,
        y_values: pd.Series,
        explainer,
        dice_exp,
        dice_dataset: pd.DataFrame,
        feature_adapter: FeatureValueAdapter,
        model_metadata: Dict[str, Any],
        feature_metadata: Dict[str, Any],
        alias_lookup: Dict[str, str],
        global_feature_importances: Dict[str, float],
        target_variable: str,
        class_names: List[str],
        feature_names: List[str],
        shap_cache_path: str = None,
        cf_cache_path: str = None,
    ):
        self.model = model
        self.dataset = dataset
        self.dataset_full = dataset_full
        self.display_dataset = display_dataset
        self.display_dataset_full = display_dataset_full
        self.y_values = y_values
        self.explainer = explainer
        self.dice_exp = dice_exp
        self.dice_dataset = dice_dataset
        self.feature_adapter = feature_adapter
        self.model_metadata = model_metadata
        self.feature_metadata = feature_metadata
        self.alias_lookup = alias_lookup
        self.global_feature_importances = global_feature_importances
        self.target_variable = target_variable
        self.class_names = class_names
        self.feature_names = feature_names
        self.shap_cache_path = shap_cache_path
        self.cf_cache_path = cf_cache_path

        self._shap_cache = self._load_pickle(self.shap_cache_path)
        self._cf_cache = self._load_pickle(self.cf_cache_path)

    def _load_pickle(self, path):
        if path:
            import os
            import pickle

            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache from {path}: {e}")
        return {}

    def _save_pickle(self, path, data):
        if path:
            import os
            import pickle

            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
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
                "text": "<p>Model training hyperparameters are:</p>" + tabulate(table, headers, tablefmt="html"),
            }
        return {"error": "Model parameters not found in metadata."}

    def get_model_description(self) -> Dict[str, Any]:
        """Returns a general description of the model architecture and its purpose."""
        if "description" in self.model_metadata:
            return {
                "data": self.model_metadata["description"],
                "text": f"<p>Model description is: {self.model_metadata['description']}</p>",
            }
        return {"error": "Model description not found in metadata."}

    def predict(self, patient_id: int) -> Dict[str, Any]:
        """Predict heart disease risk for a specific patient by ID."""
        if patient_id not in self.dataset.index:
            return {
                "error": f"Patient ID {patient_id} not found in the dataset.",
                "text": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset.",
            }

        patient_row = self.dataset.loc[patient_id].to_frame().T
        prediction = self.model.predict(patient_row)[0]
        probabilities = self.model.predict_proba(patient_row)[0].tolist()

        prob_neg = probabilities[0]
        prob_pos = probabilities[1]

        text = f"<p>Patient <code>ID</code> <var>{patient_id}</var> has a predicted risk of heart disease: <var>{self.class_names[prediction]}</var>.</p> "
        text += f"<p>The prediction class probabilities are: {self.class_names[1].lower()} {int(round(prob_pos * 100))}%, {self.class_names[0].lower()} {int(round(prob_neg * 100))}%.</p>"

        return {
            "data": {
                "patient_id": patient_id,
                "prediction": int(prediction),
                "probabilities": probabilities,
            },
            "text": text,
        }

    def feature_importance_patient(self, patient_id: int) -> Dict[str, Any]:
        """Returns SHAP-based feature importance scores (patient-specific)."""
        if patient_id not in self.dataset.index:
            return {
                "error": f"Patient ID {patient_id} not found in the dataset.",
                "text": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset.",
            }

        influences = self._get_cached_influences(patient_id)
        if influences is None:
            patient_row = self.dataset.loc[patient_id].to_frame().T
            shap_values = self.explainer.shap_values(patient_row, nsamples=10_000, silent=True)
            influences = np.asarray(shap_values).squeeze().tolist()
            self._shap_cache[patient_id] = influences
            self._save_pickle(self.shap_cache_path, self._shap_cache)

        labeled_importance = self._label_feature_scores(influences)
        text = f"<p>For the patient with <code>ID</code> <var>{patient_id}</var> the feature importances are:</p>"
        text += f"<p>{pd.DataFrame(labeled_importance.items(), columns=['Feature', 'Influence']).to_html(index=False)}</p>"

        return {
            "data": {
                "patient_id": patient_id,
                "feature_importance": influences,
                "feature_importance_by_feature": labeled_importance,
            },
            "text": text,
        }

    def feature_importance_global(self) -> Dict[str, Any]:
        """Returns SHAP-based feature importance scores (global)."""
        labeled_importance = {
            self._feature_label(feature): float(score)
            for feature, score in self.global_feature_importances.items()
        }
        result_html = pd.DataFrame(labeled_importance.items(), columns=["Feature", "Importance"]).to_html(index=False)
        text = "<p>Global feature importances based on SHAP values are:</p>" + f"{result_html}"

        return {
            "data": {
                "global_feature_importance": self.global_feature_importances,
                "global_feature_importance_by_feature": labeled_importance,
            },
            "text": text,
        }

    def dataset_summary(self, patient_id: Optional[int] = None) -> Dict[str, Any]:
        """Compare a patient's features to group-level summaries."""
        heart_disease_group = self.display_dataset_full[self.dataset_full[self.target_variable] == 1]
        non_heart_disease_group = self.display_dataset_full[self.dataset_full[self.target_variable] == 0]

        avg_hd = self._summarize_display_group(heart_disease_group)
        avg_nohd = self._summarize_display_group(non_heart_disease_group)
        avg_all = self._summarize_display_group(self.display_dataset_full)

        result = {
            "comparison": {
                "heart_disease_average": avg_hd,
                "non_heart_disease_average": avg_nohd,
                "all_patients_average": avg_all,
            }
        }

        text = "<p>Average or most common feature values for patients with heart disease:</p>"
        text += tabulate(avg_hd.items(), headers=["Feature", "Value"], tablefmt="html", numalign="left")
        text += "<p>Average or most common feature values for patients without heart disease:</p>"
        text += tabulate(avg_nohd.items(), headers=["Feature", "Value"], tablefmt="html", numalign="left")
        text += "<p>Average or most common feature values for all patients:</p>"
        text += tabulate(avg_all.items(), headers=["Feature", "Value"], tablefmt="html", numalign="left")

        if patient_id is not None:
            if patient_id not in self.display_dataset.index:
                return {
                    "error": f"Patient <code>ID</code> <var>{patient_id}</var> is out of range. Dataset has <var>{len(self.display_dataset)}</var> patients."
                }

            patient_features = self._frame_to_display_record(self.display_dataset.loc[[patient_id]])
            result["patient_id"] = patient_id
            result["comparison"]["patient"] = patient_features
            text += f"<p>Patient <code>ID</code> <var>{patient_id}</var> features:</p>"
            text += tabulate(patient_features.items(), headers=["Feature", "Value"], tablefmt="html", numalign="left")

        return {"data": result, "text": text}

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
            "auc_roc": float(auc),
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
            "auc-roc": "auc_roc",
        }

        if metrics:
            normalized_metrics = []
            for metric in metrics:
                key = metric.strip().lower().replace("-", "_")
                if key in metric_mapping:
                    normalized_metrics.append(metric_mapping[key])
                else:
                    return {
                        "error": f"Metric <code>{metric}</code> is not recognized. Valid metrics are: <code>{', '.join(metric_mapping.keys())}</code>"
                    }
            data = {key: all_metrics[key] for key in normalized_metrics}
            text = "<p>Selected performance metrics are:</p>" + tabulate(data.items(), headers=["Metric", "Value"], tablefmt="html", numalign="left")
            return {"text": text, "data": data}

        text = "<p>All performance metrics are:</p>" + tabulate(all_metrics.items(), headers=["Metric", "Value"], tablefmt="html", numalign="left")
        return {"text": text, "data": all_metrics}

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
                "true_positives": int(tp),
            }
            text = "<p>Confusion matrix statistics:</p>" + tabulate(data.items(), headers=["Statistic", "Count"], tablefmt="html", numalign="left")
            return {"text": text, "data": data}

        data = {
            "confusion_matrix": cm.tolist(),
            "classes": self.class_names,
        }
        text = "<p>Confusion matrix for multi-class classification:</p>" + tabulate(cm, headers=self.class_names, tablefmt="html", numalign="left")
        return {"text": text, "data": data}

    def what_if(self, patient_id: int, feature: str, value_change: float) -> Dict[str, Any]:
        """Simulates how changing a single feature affects predictions."""
        feature_key = self.alias_lookup.get(feature.lower(), feature)

        if patient_id not in self.dataset.index:
            return {"error": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset."}

        patient_row = self.dataset.loc[patient_id].to_frame().T
        if feature_key not in patient_row.columns:
            return {"error": f"Feature <code>{feature}</code> not found in patient data."}

        if self.feature_adapter.get_kind(feature_key) != "continuous":
            return {"error": f"Feature <code>{feature}</code> does not support delta-based what-if analysis."}

        modified_row = patient_row.copy()
        value_change_model = self.feature_adapter.delta_to_model(feature_key, value_change)
        modified_row[feature_key] += value_change_model

        original_prediction = int(self.model.predict(patient_row)[0])
        new_prediction = int(self.model.predict(modified_row)[0])

        original_prob = self.model.predict_proba(patient_row)[0].tolist()
        new_prob = self.model.predict_proba(modified_row)[0].tolist()

        original_display_value = self._display_value(feature_key, patient_row.iloc[0][feature_key])
        new_display_value = self._display_value(feature_key, modified_row.iloc[0][feature_key])
        display_feature = self._feature_label(feature_key)
        value_change_display = self._round_value(value_change)

        data = {
            "patient_id": patient_id,
            "feature_key": feature_key,
            "feature_modified": display_feature,
            "value_change": value_change_display,
            "original_value": original_display_value,
            "new_value": new_display_value,
            "original_prediction": original_prediction,
            "new_prediction": new_prediction,
            "probability_change": {
                "original": original_prob,
                "new": new_prob,
            },
        }
        text = f"<p>For patient <code>ID</code> <var>{patient_id}</var>, modifying <code>{display_feature}</code> by <var>{value_change_display}</var> changes it from <var>{original_display_value}</var> to <var>{new_display_value}</var>.</p>"
        text += f"<p>Original prediction: <var>{self.class_names[original_prediction]}</var> with probabilities: <var>{[round(prob, 2) for prob in original_prob]}</var></p>"
        text += f"<p>New prediction: <var>{self.class_names[new_prediction]}</var> with probabilities: <var>{[round(prob, 2) for prob in new_prob]}</var></p>"

        return {"data": data, "text": text}

    def counterfactual(self, patient_id: int) -> Dict[str, Any]:
        """Generates counterfactual explanations using DiCE."""
        if patient_id not in self.dataset.index:
            return {"error": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset."}

        cached_data = self._get_cached_counterfactual_data(patient_id)
        if cached_data is None:
            original_prediction = int(self.model.predict(self.dice_dataset.loc[[patient_id]])[0])
            cfe = self.dice_exp.generate_counterfactuals(
                query_instances=self.dice_dataset.loc[[patient_id]],
                total_CFs=10,
                desired_class="opposite",
            )

            final_cfes = cfe.cf_examples_list[0].final_cfs_df.copy()
            if "prediction" in final_cfes.columns:
                final_cfes.pop("prediction")

            new_predictions = [int(prediction) for prediction in self.model.predict(final_cfes).tolist()]
            cached_data = {
                "patient_id": patient_id,
                "original_prediction": original_prediction,
                "counterfactuals_model": final_cfes.to_dict(orient="records"),
                "new_predictions": new_predictions,
            }
            self._cf_cache[patient_id] = cached_data
            self._save_pickle(self.cf_cache_path, self._cf_cache)

        display_counterfactuals = self._display_records_from_model_records(cached_data["counterfactuals_model"])
        output_string = self._build_counterfactual_text(
            patient_id=patient_id,
            original_prediction=cached_data["original_prediction"],
            counterfactual_records=cached_data["counterfactuals_model"],
            new_predictions=cached_data["new_predictions"],
        )

        return {
            "data": {
                "patient_id": patient_id,
                "original_prediction": cached_data["original_prediction"],
                "counterfactuals": display_counterfactuals,
                "new_predictions": cached_data["new_predictions"],
            },
            "text": output_string,
        }

    def misclassified_cases(self) -> Dict[str, Any]:
        """Identifies frequent misclassifications and extracts common feature patterns."""
        y_pred = self.model.predict(self.dataset)

        df_copy = self.dataset.copy()
        df_copy["predicted"] = y_pred
        df_copy["misclassified"] = df_copy["predicted"] != self.y_values

        misclassified_df = df_copy[df_copy["misclassified"]]
        correctly_classified_df = df_copy[~df_copy["misclassified"]]

        misclassified_summary = self._summarize_display_group(self.display_dataset.loc[misclassified_df.index])
        correctly_classified_summary = self._summarize_display_group(self.display_dataset.loc[correctly_classified_df.index])

        data = {
            "false_positives": int(((y_pred == 1) & (self.y_values == 0)).sum()),
            "false_negatives": int(((y_pred == 0) & (self.y_values == 1)).sum()),
            "feature_distribution": {
                "misclassified_cases": misclassified_summary,
                "correctly_classified_cases": correctly_classified_summary,
            },
        }

        text = "<p>Misclassified cases statistics:</p>"
        text += f"<p>False Positives: <var>{data['false_positives']}</var></p>"
        text += f"<p>False Negatives: <var>{data['false_negatives']}</var></p>"
        text += "<p>Feature distribution for misclassified cases:</p>" + tabulate(misclassified_summary.items(), headers=["Feature", "Value"], tablefmt="html", numalign="left")
        text += "<p>Feature distribution for correctly classified cases:</p>" + tabulate(correctly_classified_summary.items(), headers=["Feature", "Value"], tablefmt="html", numalign="left")

        return {"data": data, "text": text}

    def age_group_performance(self) -> Dict[str, Any]:
        """Computes model performance across age groups."""
        if "age" not in self.dataset.columns:
            return {"error": "Required column <code>age</code> is missing from dataset."}

        y_pred = self.model.predict(self.dataset)
        age_thresholds = [0.4, 0.6]
        age_groups = {
            self._format_threshold_label("age", None, age_thresholds[0]): self.dataset[self.dataset["age"] < age_thresholds[0]],
            self._format_threshold_label("age", age_thresholds[0], age_thresholds[1]): self.dataset[(self.dataset["age"] >= age_thresholds[0]) & (self.dataset["age"] <= age_thresholds[1])],
            self._format_threshold_label("age", age_thresholds[1], None): self.dataset[self.dataset["age"] > age_thresholds[1]],
        }

        results = {}
        for group, subset in age_groups.items():
            if not subset.empty:
                y_true_group = self.y_values[subset.index]
                y_pred_group = y_pred[subset.index]
                results[group] = {
                    "accuracy": float(accuracy_score(y_true_group, y_pred_group)),
                    "precision": float(precision_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
                    "recall": float(recall_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
                    "f1_score": float(f1_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
                }

        headers = ["Age Group", "Accuracy", "Precision", "Recall", "F1 Score"]
        table = [
            [group, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"]]
            for group, metrics in results.items()
        ]
        text = "<p>Model performance across age groups:</p>" + tabulate(table, headers=headers, tablefmt="html", numalign="left")
        return {"data": results, "text": text}

    def feature_interactions(self) -> Dict[str, Any]:
        """Computes feature interactions based on correlation analysis."""
        df = self.dataset.select_dtypes(include=[np.number])
        correlation_matrix = df.corr()
        feature_names = correlation_matrix.columns.tolist()

        interaction_dict = {
            f"{self._feature_label(feature_names[i])} & {self._feature_label(feature_names[j])}": float(correlation_matrix.iloc[i, j])
            for i in range(len(feature_names))
            for j in range(i + 1, len(feature_names))
        }

        top_interactions = dict(sorted(interaction_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
        text = "<p>Top feature interactions based on correlation analysis:</p>" + tabulate(top_interactions.items(), headers=["Feature Interaction", "Correlation"], tablefmt="html", numalign="left")
        return {"data": {"top_feature_interactions": top_interactions}, "text": text}

    def count_all(self) -> Dict[str, Any]:
        """Count all instances in the dataset."""
        count = len(self.dataset)
        return {
            "data": {"count": count},
            "text": f"<p>There are <var>{count}</var> instances in the dataset.</p>",
        }

    def show_ids(self) -> Dict[str, Any]:
        """Show all available IDs."""
        ids = self.dataset.index.get_level_values(0).unique().sort_values().tolist()
        ids_str = ", ".join([f"<var>{patient_id}</var>" for patient_id in ids])
        return {
            "data": {"ids": ids},
            "text": f"<p>Available <code>ID</code> values are: {ids_str}.</p>",
        }

    def show_one(self, patient_id: int) -> Dict[str, Any]:
        """Show data for a specific ID."""
        if patient_id not in self.dataset.index:
            return {
                "error": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset.",
                "text": f"<p>There is no data for <code>ID</code> <var>{patient_id}</var>.</p>",
            }

        intro = f"<p>Here is the data for <code>ID</code> <var>{patient_id}</var>:</p>"
        patient_data = self._frame_to_display_record(self.dataset.loc[[patient_id]])
        framed = pd.DataFrame(patient_data.items(), columns=["Feature", "Value"])
        table = f"<p>{framed.to_html(index=False)}</p>"
        return {
            "data": {"patient_data": patient_data},
            "text": intro + table,
        }

    def _feature_label(self, feature: str) -> str:
        return self.feature_adapter.get_display_name(feature)

    def _display_value(self, feature: str, value: Any) -> Any:
        return self._round_value(self.feature_adapter.to_display(feature, value))

    def _round_value(self, value: Any) -> Any:
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return round(float(value), 3)
        return value

    def _display_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        display_frame = self.feature_adapter.to_display_frame(frame, features=frame.columns)
        display_frame = display_frame.apply(lambda column: column.map(self._round_value))
        return self.feature_adapter.rename_columns_for_display(display_frame)

    def _frame_to_display_record(self, frame: pd.DataFrame) -> Dict[str, Any]:
        return self._display_frame(frame).to_dict(orient="records")[0]

    def _summarize_display_group(self, frame: pd.DataFrame) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        feature_columns = [feature for feature in self.dataset.columns if feature in frame.columns]
        for feature in feature_columns:
            display_feature = self._feature_label(feature)
            column = frame[feature]
            kind = self.feature_adapter.get_kind(feature)
            if kind == "continuous":
                summary[display_feature] = self._round_value(column.astype(float).mean())
            else:
                modes = column.mode(dropna=True)
                summary[display_feature] = modes.iloc[0] if not modes.empty else None
        return summary

    def _label_feature_scores(self, influences: List[float]) -> Dict[str, float]:
        labeled = {
            self._feature_label(feature): float(influences[index])
            for index, feature in enumerate(self.dataset.columns)
        }
        return dict(sorted(labeled.items(), key=lambda item: item[1], reverse=True))

    def _get_cached_influences(self, patient_id: int) -> Optional[List[float]]:
        if patient_id not in self._shap_cache:
            return None
        cache_entry = self._shap_cache[patient_id]
        if isinstance(cache_entry, tuple):
            return cache_entry[0]
        return cache_entry

    def _get_cached_counterfactual_data(self, patient_id: int) -> Optional[Dict[str, Any]]:
        if patient_id not in self._cf_cache:
            return None
        cache_entry = self._cf_cache[patient_id]
        if isinstance(cache_entry, tuple):
            cache_entry = cache_entry[0]
        if "counterfactuals_model" not in cache_entry and "counterfactuals" in cache_entry:
            cache_entry = {
                "patient_id": cache_entry.get("patient_id", patient_id),
                "original_prediction": cache_entry["original_prediction"],
                "counterfactuals_model": cache_entry["counterfactuals"],
                "new_predictions": cache_entry["new_predictions"],
            }
        return cache_entry

    def _display_records_from_model_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not records:
            return []
        frame = pd.DataFrame(records)
        return self._display_frame(frame).to_dict(orient="records")

    def _build_counterfactual_text(
        self,
        patient_id: int,
        original_prediction: int,
        counterfactual_records: List[Dict[str, Any]],
        new_predictions: List[int],
    ) -> str:
        original_instance = self.dice_dataset.loc[[patient_id]]
        output_string = f"<p>The original prediction for the data sample with <code>ID</code> <var>{patient_id}</var> is <samp>{self.class_names[original_prediction]}</samp>.</p>"
        output_string += "<p>Here are some options to change the prediction of this instance.</p><ul>"
        transition_words = ["Further,", "Also,", "In addition,", "Furthermore,"]

        for index, counterfactual in enumerate(counterfactual_records[:3]):
            prefix = "First, if you" if index == 0 else f"{random.choice(transition_words)} if you"
            counterfactual_frame = pd.DataFrame([counterfactual])
            output_string += f"<li>{prefix}{self._get_change_string(counterfactual_frame, original_instance)}, the model will predict <samp>{self.class_names[new_predictions[index]]}</samp>.</li>"
        output_string += "</ul>"
        return output_string

    def _get_change_string(self, cfe: pd.DataFrame, original_instance: pd.DataFrame) -> str:
        """Build a string describing changes between counterfactual and original instance."""
        cfe_features = list(cfe.columns)
        original_features = list(original_instance.columns)
        assert set(cfe_features) == set(original_features), "CFE features and Original Instance features are different!"

        changes = []
        for feature in cfe_features:
            original_value = original_instance[feature].values[0]
            new_value = cfe[feature].values[0]
            if isinstance(new_value, str):
                new_value = float(new_value)
            if original_value != new_value:
                direction = "increase" if new_value > original_value else "decrease"
                display_value = self._display_value(feature, new_value)
                changes.append(f" {direction} <code>{self._feature_label(feature)}</code> to <var>{display_value}</var>")

        return " and".join(changes)

    def _format_threshold_label(self, feature: str, min_model: Optional[float], max_model: Optional[float]) -> str:
        if min_model is None and max_model is not None:
            return f"<{self._display_value(feature, max_model)}"
        if min_model is not None and max_model is not None:
            return f"{self._display_value(feature, min_model)}-{self._display_value(feature, max_model)}"
        if min_model is not None:
            return f">{self._display_value(feature, min_model)}"
        return self._feature_label(feature)
