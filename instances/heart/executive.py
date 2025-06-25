import pandas as pd
import joblib
import shap
import copy
import dice_ml
import json
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tabulate import tabulate
import random

INSTANCE_PATH = 'instances/heart/'

feature_metadata_path = INSTANCE_PATH + 'data/feature_metadata.json'

dataset = pd.read_csv(INSTANCE_PATH + 'data/test_set.csv')
dataset_full = copy.deepcopy(dataset)
y_values = dataset.pop('num')

target_variable = 'num'

feature_names = dataset.columns.tolist()
class_names = ["NEGATIVE", "POSITIVE"]

explanation_dataset = copy.deepcopy(dataset)
explanation_dataset = explanation_dataset.to_numpy()
explanation_dataset = shap.kmeans(explanation_dataset, 25)

with open(INSTANCE_PATH + 'model/best_model_3_DecisionTreeClassifier.pkl', 'rb') as file:
    model = joblib.load(file)
    
with open(INSTANCE_PATH + 'model/best_model_3_DecisionTreeClassifier_metadata.json', 'r') as file:
    model_metadata = json.load(file)
        
explainer = shap.KernelExplainer(model.predict, explanation_dataset)

dice_dataset = copy.deepcopy(dataset)
dice_dataset['prediction'] = model.predict(dice_dataset.to_numpy())

dice_data = dice_ml.Data(dataframe=dice_dataset, 
                         continuous_features=["age", "trestbps", "chol", "thalch", "oldpeak"],
                         categorical_features=["thal", "ca", "slope", "restecg", "cp", "sex", "fbs", "exang"], 
                         outcome_name='prediction')

dice_model = dice_ml.Model(model=model, backend="sklearn")

dice_exp = dice_ml.Dice(dice_data, dice_model, method="random")

dice_dataset.pop('prediction')

feature_metadata = {}
alias_lookup = {}

with open(feature_metadata_path, "r") as f:
    feature_metadata = json.load(f)

alias_lookup = {
    alias.lower(): feat
    for feat, meta in feature_metadata.items()
    for alias in ([feat] + meta.get("aliases", []))
}


def get_model_parameters():
    """Returns the exact training hyperparameters of the model (e.g., max_depth, criterion)."""

    if "parameters" in model_metadata:
        headers = ["Parameter", "Value"]
        table = [
            [param, str(value)]
            for param, value in model_metadata["parameters"].items()
        ]
        return { "data": model_metadata["parameters"], "text": "<p>Model training hyperparameters are:</p>"  + tabulate(table, headers, tablefmt='html')}
    return { "error": "Model parameters not found in metadata." }


def get_model_description():
    """Returns a general description of the model architecture and its purpose (e.g., DecisionTreeClassifier trained to predict heart disease)."""

    if "description" in model_metadata:
        return { "data": model_metadata["description"], "text": f"<p>Model description is: {model_metadata['description']}</p>" }
    return { "error": "Model description not found in metadata." }


def predict(patient_id: int):
    """Predict heart disease risk for a specific patient by ID."""

    if patient_id not in dataset.index:
        return {"error": f"Patient ID {patient_id} not found in the dataset.", "text": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset."}

    # Get a single row as a DataFrame
    patient_row = dataset.loc[patient_id].to_frame().T

    # Predict
    prediction = model.predict(patient_row)[0]
    probabilities = model.predict_proba(patient_row)[0].tolist()

    return { "data": {
        "patient_id": patient_id,
        "prediction": prediction,
        "probabilities": probabilities
    }, "text": f"<p>Patient <code>ID</code> <var>{patient_id}</var> has a predicted risk of heart disease: <var>{class_names[prediction]}</var>.</p> <p>The prediction class (positive, negative) probabilities are: <var>{[ round(prob, 2) for prob in probabilities ]}</var></p>" }


def feature_importance(patient_id=None):
    """Returns SHAP-based feature importance scores (global or patient-specific)."""

    # Ensure patient exists
    if patient_id not in dataset.index:
        return {"error": f"Patient ID {patient_id} not found in the dataset.", "text": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset."}

    # Prepare patient row for SHAP
    patient_row = dataset.loc[patient_id].to_frame().T
    shap_values = explainer.shap_values(patient_row, nsamples=10_000, silent=True)
    influences = shap_values.squeeze()
    result = pd.DataFrame(influences, columns=['Influence'], index=dataset.columns).sort_values(by='Influence', ascending=False)
    text = f"<p>For the patient with <code>ID</code> <var>{patient_id}</var> the feature importances are:</p>" + f"<p>{result.to_html()}</p>"

    return { "data": {"patient_id": patient_id, "feature_importance": influences },
             "text": text }


def dataset_summary(patient_id=None):
    """
    If a patient ID is provided, compare the patient's features to average features of patients
    with and without heart disease. If no ID is given, return only the group-level averages.
    """
    # Columns used for comparison
    feature_columns = [f for f in dataset.columns]

    # Compute averages
    avg_hd = dataset_full[dataset_full[target_variable] == 1][feature_columns].mean().round(3).to_dict()
    avg_nohd = dataset_full[dataset_full[target_variable] == 0][feature_columns].mean().round(3).to_dict()
    avg_all = dataset_full[feature_columns].mean().round(3).to_dict()

    result = {
        "comparison": {
            "heart_disease_average": avg_hd,
            "non_heart_disease_average": avg_nohd,
            "all_patients_average": avg_all
        }
    }
    
    text = "<p>Average features for patients with heart disease:</p>" + tabulate(avg_hd.items(), headers=["Feature", "Average"], tablefmt='html', numalign="left")
    text += "<p>Average features for patients without heart disease:</p>" + tabulate(avg_nohd.items(), headers=["Feature", "Average"], tablefmt='html', numalign="left")
    text += "<p>Average features for all patients:</p>" + tabulate(avg_all.items(), headers=["Feature", "Average"], tablefmt='html', numalign="left")

    # Optional patient comparison
    if patient_id is not None:
        try:
            patient_row = dataset.iloc[[patient_id]]
            patient_features = patient_row[feature_columns].iloc[0].round(3).to_dict()
            result["patient_id"] = patient_id
            result["comparison"]["patient"] = patient_features
            text += f"<p>Patient <code>ID</code> <var>{patient_id}</var> features:</p>" + tabulate(patient_features.items(), headers=["Feature", "Value"], tablefmt='html', numalign="left")
        except IndexError:
            return { "error": f"Patient <code>ID</code> <var>{patient_id}</var> is out of range. Dataset has <var>{len(dataset)}</var> patients." }
            
    return { "data": result, "text": text }


def performance_metrics(metrics: list = None):
    """Computes and returns selected performance metrics, including AUC-ROC."""

    y_pred = model.predict(dataset)

    try:
        y_prob = model.predict_proba(dataset)
    except AttributeError:
        y_prob = None  # Some models don't support this

    n_classes = len(np.unique(y_values))

    try:
        if y_prob is not None:
            if n_classes == 2:
                y_score = y_prob[:, 1]
                auc = roc_auc_score(y_values, y_score)
            else:
                auc = roc_auc_score(y_values, y_prob, multi_class="ovr")
        else:
            auc = 0.0
    except Exception:
        auc = 0.0

    all_metrics = {
        "accuracy": float(accuracy_score(y_values, y_pred)),
        "precision": float(precision_score(y_values, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_values, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_values, y_pred, average="weighted", zero_division=0)),
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
                return { "error": f"Metric <code>{m}</code> is not recognized. Valid metrics are: <code>{', '.join(metric_mapping.keys())}</code>" }
        data = {key: all_metrics[key] for key in normalized_metrics}
        text = "<p>Selected performance metrics are:</p>" + tabulate(data.items(), headers=["Metric", "Value"], tablefmt='html', numalign="left")
        return { "text": text, "data": data }
    else:
        data = all_metrics
        text = "<p>All performance metrics are:</p>" + tabulate(data.items(), headers=["Metric", "Value"], tablefmt='html', numalign="left")
        return { "text": text, "data": data }


def confusion_matrix_stats():
    """Returns the confusion matrix with counts of TN, FP, FN, TP, or the full matrix for multi-class."""

    y_pred = model.predict(dataset)

    cm = confusion_matrix(y_values, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        data = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
        text = f"<p>Confusion matrix statistics:</p>" + tabulate(data.items(), headers=["Statistic", "Count"], tablefmt='html', numalign="left")
        return { "text": text, "data": data }

    else:
        data = {
            "confusion_matrix": cm.tolist(),
            "classes": class_names
        }
        text = f"<p>Confusion matrix for multi-class classification:</p>" + tabulate(cm, headers=class_names, tablefmt='html', numalign="left")
        return { "text": text, "data": data }
    

def what_if(patient_id: int, feature: str, value_change: float):
    """Simulates how changing a single feature affects predictions for a given patient."""
    feature_key = alias_lookup.get(feature.lower(), feature)

    if patient_id not in dataset.index:
        return {"error": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset."}

    patient_row = dataset.loc[patient_id].to_frame().T

    if feature_key not in patient_row.columns:
        return {"error": f"Feature <code>{feature}</code> not found in patient data."}

    modified_row = patient_row.copy()
    modified_row[feature_key] += value_change

    original_prediction = int(model.predict(patient_row)[0])
    new_prediction = int(model.predict(modified_row)[0])

    original_prob = model.predict_proba(patient_row)[0].tolist()
    new_prob = model.predict_proba(modified_row)[0].tolist()
    
    data = {
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
    text = f"<p>For patient <code>ID</code> <var>{patient_id}</var>, modifying feature <code>{feature_key}</code> by <var>{value_change}</var> results in:</p>"
    text += f"<p>Original prediction: <var>{class_names[original_prediction]}</var> with probabilities: <var>{[round(prob, 2) for prob in original_prob]}</var></p>"
    text += f"<p>New prediction: <var>{class_names[new_prediction]}</var> with probabilities: <var>{[round(prob, 2) for prob in new_prob]}</var></p>"

    return { "data": data, "text": text }


def _get_change_string(cfe, original_instance):
    """Builds a string describing the changes between the cfe and original instance."""
    cfe_features = list(cfe.columns)
    original_features = list(original_instance.columns)
    message = "CFE features and Original Instance features are different!"
    assert set(cfe_features) == set(original_features), message

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
            change_string += f"{inc_dec} <code>{feature}</code> to <var>{str(round(cfe_f, 2))}</var>"
            change_string += " and "
    # Strip off last and
    change_string = change_string[:-5]
    return change_string


def counterfactual(patient_id: int):
    """Generates counterfactual explanations using DiCE for a given patient ID and target class."""

    # Check if patient exists
    if patient_id not in dataset.index:
        return {"error": f"Patient <code>ID</code> <var>{patient_id}</var> not found in the dataset."}
    
    original_prediction = model.predict(dice_dataset.loc[[patient_id]])[0]

    # Generate counterfactuals
    cfe = dice_exp.generate_counterfactuals(
        query_instances=dice_dataset.loc[[patient_id]],
        total_CFs=10,
        desired_class="opposite"
    )
    
    final_cfes = cfe.cf_examples_list[0].final_cfs_df
    final_cfe_ids = list(final_cfes.index)
    if 'prediction' in final_cfes.columns:
            final_cfes.pop('prediction')
    
    new_predictions = model.predict(final_cfes)
    
    original_instance = dice_dataset.loc[[patient_id]]
    
    output_string = f"<p>The original prediction for the data sample with <code>ID</code> <var>{patient_id}</var> is <samp>{class_names[original_prediction]}</samp>.</p>"
    output_string += "<p>Here are some options to change the prediction of this instance.</p>"
    
    output_string += "<ul>"
    output_string += "<li>First, if you"
    transition_words = ["Further,", "Also,", "In addition,", "Furthermore,"]
    
    for i, c_id in enumerate(final_cfe_ids):
        if i < 3 and i < len(final_cfe_ids):
            if i != 0:
                output_string += f"<li>{random.choice(transition_words)} if you"
            output_string += _get_change_string(final_cfes.loc[[c_id]], original_instance)
            new_prediction = new_predictions[i]
            output_string += f", the model will predict <samp>{class_names[new_prediction]}</samp>.</li>"
    output_string += "</ul>"
    
    data = {
        "patient_id": patient_id,
        "original_prediction": original_prediction,
        "counterfactuals": final_cfes.to_dict(orient='records'),
        "new_predictions": new_predictions.tolist()
    }

    return { "data": data, "text": output_string }

            
def misclassified_cases():
    """Identifies frequent misclassifications and extracts common feature patterns."""

    y_pred = model.predict(dataset)

    df_copy = dataset.copy()
    df_copy["predicted"] = y_pred
    df_copy["misclassified"] = df_copy["predicted"] != y_values

    misclassified_df = df_copy[df_copy["misclassified"]]
    correctly_classified_df = df_copy[~df_copy["misclassified"]]

    data = {
        "false_positives": int(((y_pred == 1) & (y_values == 0)).sum()),
        "false_negatives": int(((y_pred == 0) & (y_values == 1)).sum()),
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
    
    text = "<p>Misclassified cases statistics:</p>"
    text += f"<p>False Positives: <var>{data['false_positives']}</var></p>"
    text += f"<p>False Negatives: <var>{data['false_negatives']}</var></p>"
    text += "<p>Feature distribution for misclassified cases:</p>" + tabulate(data["feature_distribution"]["misclassified_cases"].items(), headers=["Feature", "Average"], tablefmt='html', numalign="left")
    text += "<p>Feature distribution for correctly classified cases:</p>" + tabulate(data["feature_distribution"]["correctly_classified_cases"].items(), headers=["Feature", "Average"], tablefmt='html', numalign="left")
    
    return { "data": data, "text": text }


def age_group_performance():
    """Computes model performance across different age groups."""

    if "age" not in dataset.columns:
        return {"error": f"Required column <code>age</code> is missing from dataset."}
    
    y_pred = model.predict(dataset)

    # Define age groups
    age_groups = {
        "<40": dataset[dataset["age"] < 0.4],
        "40-60": dataset[(dataset["age"] >= 0.4) & (dataset["age"] <= 0.6)],
        ">60": dataset[dataset["age"] > 0.6]
    }

    results = {}
    for group, subset in age_groups.items():
        if not subset.empty:
            y_true_group = y_values[subset.index]
            y_pred_group = y_pred[subset.index]

            results[group] = {
                "accuracy": float(accuracy_score(y_true_group, y_pred_group)),
                "precision": float(
                    precision_score(y_true_group, y_pred_group, average="weighted", zero_division=0)),
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

    return { "data": results, "text": text }


def feature_interactions():
    """Computes feature interactions based on correlation analysis."""
    df = dataset.select_dtypes(include=[np.number])  # Only numeric features

    correlation_matrix = df.corr()
    feature_names = correlation_matrix.columns.tolist()

    interaction_dict = {
        f"{feature_names[i]} & {feature_names[j]}": float(correlation_matrix.iloc[i, j])
        for i in range(len(feature_names)) for j in range(i + 1, len(feature_names))
    }

    # Sort by absolute correlation
    top_interactions = dict(sorted(interaction_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

    return {"top_feature_interactions": top_interactions}