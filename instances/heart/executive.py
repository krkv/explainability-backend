import pandas as pd
import pickle
import shap
import copy
import dice_ml
import json
import numpy as np
from sklearn.metrics import explained_variance_score, root_mean_squared_error, mean_absolute_error

INSTANCE_PATH = 'instances/heart/'

dataset = pd.read_csv(INSTANCE_PATH + 'data/test_set.csv')
y_values = dataset.pop('num')

feature_names = dataset.columns.tolist()

explanation_dataset = copy.deepcopy(dataset)
explanation_dataset = explanation_dataset.to_numpy()
explanation_dataset = shap.kmeans(explanation_dataset, 25)

with open(INSTANCE_PATH + 'model/best_model_3_DecisionTreeClassifier.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open(INSTANCE_PATH + 'model/best_model_3_DecisionTreeClassifier_metadata.json', 'r') as file:
    model_metadata = json.load(file)
        
explainer = shap.KernelExplainer(model.predict, explanation_dataset, link="identity")

dice_dataset = copy.deepcopy(dataset)
dice_dataset['prediction'] = model.predict(dice_dataset.to_numpy())

dice_data = dice_ml.Data(dataframe=dice_dataset, 
                         continuous_features=["age", "trestbps", "chol", "thalch", "oldpeak"],
                         categorical_features=["thal", "ca", "slope", "restecg", "cp", "sex", "fbs", "exang"], 
                         outcome_name='prediction')

dice_model = dice_ml.Model(model=model, backend="sklearn")

dice_exp = dice_ml.Dice(dice_data, dice_model, method="random")

dice_dataset.pop('prediction')

# def __init__(self, config_path="heart_disease_inference_settings.json"):
#         """Initialize the class by loading the dataset, model, and necessary configurations."""

#         # Load configuration
#         with open(config_path, "r") as f:
#             self.config = json.load(f)

#         # Validate required fields
#         required_keys = ["model_path", "model_metadata", "dataset_path", "target_variable",
#                          "non_informative_features"]
#         for key in required_keys:
#             if key not in self.config:
#                 raise KeyError(f"Missing required key '{key}' in configuration file.")

#         print("📥 Loading dataset and model...")

#         # Load dataset and model
#         self.df, self.feature_names = self.load_dataset()
#         self.model, self.metadata = self.load_model()

#         self.target_variable = self.config["target_variable"]
#         self.non_informative_features = self.config["non_informative_features"] + [self.target_variable]

#         self.df_cleaned = self.df.drop(columns=self.non_informative_features, errors="ignore")
#         self.feature_names = [f for f in self.feature_names if f not in self.non_informative_features]

#         # Load feature metadata and alias lookup
#         feature_metadata_path = self.config.get("feature_metadata_path")
#         if feature_metadata_path:
#             with open(feature_metadata_path, "r") as f:
#                 self.feature_metadata = json.load(f)
#             self.alias_lookup = {
#                 alias.lower(): feat
#                 for feat, meta in self.feature_metadata.items()
#                 for alias in ([feat] + meta.get("aliases", []))
#             }
#         else:
#             self.feature_metadata = {}
#             self.alias_lookup = {}

#         print("📊 Precomputing feature importance...")
#         self.precomputed_feature_importance = self._compute_feature_importance()
#         print("✅ Data and model successfully loaded. Ready to use!")

def get_model_parameters():
    """Returns the exact training hyperparameters of the model (e.g., max_depth, criterion)."""

    if "parameters" in model_metadata:
        return model_metadata["parameters"]
    return {"error": "Model parameters not found in metadata."}

def get_model_description():
    """Returns a general description of the model architecture and its purpose (e.g., DecisionTreeClassifier trained to predict heart disease)."""

    return {"model_description": model_metadata.get("description", "No description available.")} 

### HERE

def _compute_feature_importance(self):
    """Computes and returns SHAP-based global feature importance scores."""

    # Use SHAP KernelExplainer for model-agnostic explanation if needed
    explainer = shap.Explainer(self.model.predict, self.df_cleaned)
    shap_values = explainer(self.df_cleaned)

    importance = np.abs(shap_values.values).mean(axis=0)

    return {
        "global_feature_importance": dict(sorted(
            {self.feature_names[i]: float(importance[i]) for i in range(len(self.feature_names))}.items(),
            key=lambda x: x[1], reverse=True))
    }

def load_model(self):
    """Loads the trained model (.pkl) and its metadata (.json)."""
    model_path = self.config["model_path"]
    metadata_path = self.config["model_metadata"]

    model = joblib.load(model_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return model, metadata

def predict(self, patient_id: int):
    """Predict heart disease risk for a specific patient by ID."""

    if patient_id not in self.df_cleaned.index:
        return {"error": f"Patient ID {patient_id} not found in the dataset."}

    # Get a single row as a DataFrame
    patient_row = self.df_cleaned.loc[patient_id].to_frame().T

    # Predict
    prediction = self.model.predict(patient_row)[0]
    probabilities = self.model.predict_proba(patient_row)[0].tolist()

    return {
        "patient_id": patient_id,
        "prediction": int(prediction),
        "probabilities": probabilities
    }

def feature_importance(self, patient_id=None):
    """Returns SHAP-based feature importance scores (global or patient-specific)."""
    
    # Return precomputed global SHAP importance
    if patient_id is None:
        return {"global_feature_importance": self.precomputed_feature_importance}

    # Ensure patient exists
    if patient_id not in self.df.index:
        return {"error": f"Patient ID {patient_id} not found in the dataset."}

    # Prepare patient row for SHAP
    patient_row = self.df.loc[patient_id].drop(labels=self.non_informative_features, errors="ignore").to_frame().T

    explainer = shap.Explainer(self.model.predict, self.df_cleaned)
    shap_values = explainer(patient_row)

    # Build per-feature contributions
    patient_importance = {
        self.feature_names[i]: float(abs(shap_values.values[0][i]))
        for i in range(len(self.feature_names))
    }

    # Sort and filter out zero contributions
    sorted_importance = {
        k: v for k, v in sorted(patient_importance.items(), key=lambda item: item[1], reverse=True) if v > 0
    }

    return {"patient_id": patient_id, "feature_importance": sorted_importance}

def dataset_summary(self, patient_id=None):
    """
    If a patient ID is provided, compare the patient's features to average features of patients
    with and without heart disease. If no ID is given, return only the group-level averages.
    """
    # Columns used for comparison
    feature_columns = [f for f in self.df.columns if f not in self.non_informative_features]

    # Compute averages
    avg_hd = self.df[self.df[self.target_variable] == 1][feature_columns].mean().round(3).to_dict()
    avg_nohd = self.df[self.df[self.target_variable] == 0][feature_columns].mean().round(3).to_dict()
    avg_all = self.df[feature_columns].mean().round(3).to_dict()

    result = {
        "comparison": {
            "heart_disease_average": avg_hd,
            "non_heart_disease_average": avg_nohd,
            "all_patients_average": avg_all
        }
    }

    # Optional patient comparison
    if patient_id is not None:
        try:
            patient_row = self.df.iloc[[patient_id]]
            patient_features = patient_row[feature_columns].iloc[0].round(3).to_dict()
            result["patient_id"] = patient_id
            result["comparison"]["patient"] = patient_features
        except IndexError:
            result["warning"] = f"Patient ID {patient_id} is out of range. Dataset has {len(self.df)} patients."

    return result

def performance_metrics(self, metrics: list = None):
    """Computes and returns selected performance metrics, including AUC-ROC."""

    y = self.df[self.target_variable].values.ravel()
    X = self.df_cleaned
    y_pred = self.model.predict(X)

    try:
        y_prob = self.model.predict_proba(X)
    except AttributeError:
        y_prob = None  # Some models don't support this

    n_classes = len(np.unique(y))

    try:
        if y_prob is not None:
            if n_classes == 2:
                y_score = y_prob[:, 1]
                auc = roc_auc_score(y, y_score)
            else:
                auc = roc_auc_score(y, y_prob, multi_class="ovr")
        else:
            auc = 0.0
    except Exception:
        auc = 0.0

    all_metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "auc_roc": float(auc)
    }

    # Normalization mapping for case-insensitive and variant-friendly lookup
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
                raise ValueError(f"Unknown metric: {m}")
        return {key: all_metrics[key] for key in normalized_metrics}
    else:
        return all_metrics

def confusion_matrix_stats(self):
    """Returns the confusion matrix with counts of TN, FP, FN, TP, or the full matrix for multi-class."""

    y = self.df[self.target_variable].values.ravel()
    X = self.df_cleaned
    y_pred = self.model.predict(X)

    cm = confusion_matrix(y, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    else:
        return {
            "confusion_matrix": cm.tolist()
        }

def what_if(self, patient_id: int, feature: str, value_change: float):
    """Simulates how changing a single feature affects predictions for a given patient."""
    feature_key = self.alias_lookup.get(feature.lower(), feature)

    if patient_id not in self.df.index:
        return {"error": f"Patient ID {patient_id} not found in the dataset."}

    patient_row = self.df.loc[patient_id].drop(labels=self.non_informative_features, errors="ignore").to_frame().T

    if feature_key not in patient_row.columns:
        return {"error": f"Feature '{feature}' not found in patient data."}

    modified_row = patient_row.copy()
    modified_row[feature_key] += value_change

    original_prediction = int(self.model.predict(patient_row)[0])
    new_prediction = int(self.model.predict(modified_row)[0])

    original_prob = self.model.predict_proba(patient_row)[0].tolist()
    new_prob = self.model.predict_proba(modified_row)[0].tolist()

    return {
        "patient_id": patient_id,
        "feature_modified": feature_key,
        "value_change": value_change,
        "original_prediction": original_prediction,
        "new_prediction": new_prediction,
        "probability_change": {
            "original": original_prob,
            "new": new_prob
        }
    }

def counterfactual(self, patient_id: int, num_counterfactuals=1, desired_class=0):
    """Generates counterfactual explanations using DiCE for a given patient ID and target class."""

    # Suppress DiCE warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Check if patient exists
    if patient_id not in self.df.index:
        return {"error": f"Patient ID {patient_id} not found in the dataset."}

    # Prepare dataset for DiCE
    df_counterfactual = self.df.drop(columns=["dataset"], errors="ignore").reset_index(drop=True)

    # Extract the original patient row
    original_patient = self.df.loc[[patient_id]].drop(columns=["dataset"], errors="ignore")

    # Drop target variable and reset index for DiCE compatibility
    patient_row = original_patient.drop(columns=[self.target_variable], errors="ignore").reset_index(drop=True)

    # Get feature types from config
    categorical = self.config.get("categorical_features", [])
    continuous = self.config.get("continuous_features", [])

    # Ensure categorical features are int
    df_counterfactual[categorical] = df_counterfactual[categorical].astype(int)
    patient_row[categorical] = patient_row[categorical].astype(int)

    # Create DiCE objects
    data = dice_ml.Data(
        dataframe=df_counterfactual,
        continuous_features=continuous,
        categorical_features=categorical,
        outcome_name=self.target_variable
    )
    model = dice_ml.Model(model=self.model, backend="sklearn")
    exp = dice_ml.Dice(data, model)

    # Generate counterfactuals
    cf = exp.generate_counterfactuals(
        query_instances=patient_row,
        total_CFs=num_counterfactuals,
        desired_class=desired_class
    )
    cf_df = cf.cf_examples_list[0].final_cfs_df
    cf_df[categorical] = cf_df[categorical].astype(int)

    # Compare with original
    counterfactuals_with_changes = []
    for _, cf_row in cf_df.iterrows():
        cf_dict = cf_row.to_dict()
        changes = {"original": {}, "counterfactual": {}}

        for feature in cf_dict:
            if feature != self.target_variable and feature in original_patient.columns:
                original_val = original_patient[feature].values[0]
                cf_val = cf_dict[feature]
                if original_val != cf_val:
                    changes["original"][feature] = int(original_val) if isinstance(original_val,
                                                                                    np.integer) else original_val
                    changes["counterfactual"][feature] = int(cf_val) if isinstance(cf_val, np.integer) else cf_val

        counterfactuals_with_changes.append({
            "counterfactual": {k: int(v) if isinstance(v, np.integer) else v for k, v in cf_dict.items()},
            "changes": changes
        })

    # Restore default warnings
    warnings.simplefilter(action="default", category=FutureWarning)

    return {"patient_id": patient_id, "counterfactuals": counterfactuals_with_changes}

def misclassified_cases(self):
    """Identifies frequent misclassifications and extracts common feature patterns."""

    y = self.df[self.target_variable]
    X = self.df_cleaned
    y_pred = self.model.predict(X)

    df_copy = self.df.copy()
    df_copy["predicted"] = y_pred
    df_copy["misclassified"] = df_copy["predicted"] != y

    misclassified_df = df_copy[df_copy["misclassified"]]
    correctly_classified_df = df_copy[~df_copy["misclassified"]]

    return {
        "false_positives": int(((y_pred == 1) & (y == 0)).sum()),
        "false_negatives": int(((y_pred == 0) & (y == 1)).sum()),
        "feature_distribution": {
            "misclassified_cases": misclassified_df
            .drop(columns=["predicted", "misclassified"], errors="ignore")
            .select_dtypes(include=[np.number])
            .mean()
            .to_dict(),
            "correctly_classified_cases": correctly_classified_df
            .drop(columns=["predicted", "misclassified"], errors="ignore")
            .select_dtypes(include=[np.number])
            .mean()
            .to_dict()
        }
    }

def age_group_performance(self):
    """Computes model performance across different age groups."""

    target = self.target_variable

    if target not in self.df.columns or "age" not in self.df.columns:
        return {"error": f"Required column(s) missing from dataset."}

    df = self.df.drop(columns=["dataset", "id"], errors="ignore").copy()
    y = df[target].values.ravel()
    y_pred = self.model.predict(df.drop(columns=[target], errors="ignore"))

    # Define age groups
    age_groups = {
        "<40": df[df["age"] < 40],
        "40-60": df[(df["age"] >= 40) & (df["age"] <= 60)],
        ">60": df[df["age"] > 60]
    }

    results = {}
    for group, subset in age_groups.items():
        if not subset.empty:
            y_true_group = subset[target]
            y_pred_group = y_pred[subset.index]

            results[group] = {
                "accuracy": float(accuracy_score(y_true_group, y_pred_group)),
                "precision": float(
                    precision_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
                "f1_score": float(f1_score(y_true_group, y_pred_group, average="weighted", zero_division=0))
            }

    return results

def feature_interactions(self):
    """Computes feature interactions based on correlation analysis."""
    df = self.df_cleaned.select_dtypes(include=[np.number])  # Only numeric features

    correlation_matrix = df.corr()
    feature_names = correlation_matrix.columns.tolist()

    interaction_dict = {
        f"{feature_names[i]} & {feature_names[j]}": float(correlation_matrix.iloc[i, j])
        for i in range(len(feature_names)) for j in range(i + 1, len(feature_names))
    }

    # Sort by absolute correlation
    top_interactions = dict(sorted(interaction_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

    return {"top_feature_interactions": top_interactions}