import pandas as pd
import pickle
import shap
import copy
import dice_ml
import numpy as np
from sklearn.metrics import explained_variance_score, root_mean_squared_error, mean_absolute_error

dataset = pd.read_csv('data/summer_workday_test.csv')
y_values = dataset.pop('y')

explanation_dataset = copy.deepcopy(dataset)
explanation_dataset = explanation_dataset.to_numpy()
explanation_dataset = shap.kmeans(explanation_dataset, 25)

with open('model/custom_gp_model.pkl', 'rb') as file:
    model = pickle.load(file)
        
explainer = shap.KernelExplainer(model.predict, explanation_dataset, link="identity")

dice_dataset = copy.deepcopy(dataset)
dice_dataset['prediction'] = model.predict(dice_dataset.to_numpy())

dice_data = dice_ml.Data(dataframe=dice_dataset, 
                         continuous_features=['indoor_temperature', 'outdoor_temperature', 'past_electricity'], 
                         outcome_name='prediction')

dice_model = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")

dice_exp = dice_ml.Dice(dice_data, dice_model, method="random")

dice_dataset.pop('prediction')

# Providing information

def available_functions():
    text = "<p>Here are the available functions:</p>"
    text += "<ul>"
    text += "<li><code>count_all()</code>: Count all instances in the dataset.</li>"
    text += "<li><code>count_group()</code>: Count instances in a group, filtered by available features.</li>"
    text += "<li><code>show_ids()</code>: Show all available IDs.</li>"
    text += "<li><code>show_one()</code>: Show data for a specific ID.</li>"
    text += "<li><code>show_group()</code>: Show data for a group, filtered by available features.</li>"
    text += "<li><code>predict_one()</code>: Predict energy consumption for a specific ID.</li>"
    text += "<li><code>predict_group()</code>: Predict energy consumption for a group, filtered by available features.</li>"
    text += "<li><code>predict_new()</code>: Predict energy consumption for new data.</li>"
    text += "<li><code>mistake_one()</code>: Show prediction error for a specific ID.</li>"
    text += "<li><code>mistake_group()</code>: Show prediction errors for a group, filtered by available features.</li>"
    text += "<li><code>explain_one()</code>: Explain prediction for a specific ID using SHAP values.</li>"
    text += "<li><code>explain_group()</code>: Explain predictions for a group using SHAP values, filtered by available features.</li>"
    text += "<li><code>cfes_one()</code>: Generate counterfactual explanations for a specific ID using DiCE.</li>"
    text += "<li><code>what_if_one()</code>: Show what-if analysis for a specific ID.</li>"
    text += "</ul>"
    text += "<p>Let me know if you want to use any of these functions.</p>"
    return text

def about_dataset():
    text = "<p>The dataset <b>summer_workday_test</b> is a test dataset contains information about energy consumption.</p>"
    text += f"<p>It has <var>{len(dataset)}</var> instances and <var>{len(dataset.columns)}</var> features.</p>"
    text += f"<p>The features are: {', '.join([f'<code>{feature}</code>' for feature in dataset.columns])}.</p>"
    text += "<p>I can also give you the more in depth statistics of the dataset. Would you like to see it?</p>"
    return text

def about_dataset_in_depth():
    text = "<p>Here are the statistics of each feature in the dataset:</p>"
    text += f"<p>{dataset.describe().round(2).to_html()}</p>"
    return text

def about_model():
    text = "<p>The model is a <b>genetic programming symbolic regressor</b>.</p>"
    text += "<p>It was trained on the dataset features to predict energy consumption.</p>"
    text += "<p>Symbolic regression is a machine learning technique that aims to identify an underlying mathematical expression that best describes a relationship.</p>"
    text += "<p>It begins by building a population of naive random formulas to represent a relationship between known independent variables and their dependent variable targets in order to predict new data.</p>"
    text += "<p>Each successive generation of programs is then evolved from the one that came before it by selecting the fittest individuals from the population to undergo genetic operations.<p>"
    text += f"<p>The program that is currently loaded looks like this: <code>{model._program}</code></p>"
    text += f"<p>Where features are: {' '.join([f'X{i} = {dataset.columns[i]}' for i in range(len(dataset.columns))])}</p>"
    return text

def model_accuracy():
    pred = model.predict(dataset)
    explained_variance = explained_variance_score(y_values, pred)
    rmse = root_mean_squared_error(y_values, pred)
    mae = mean_absolute_error(y_values, pred)
    text = f"<p>The model has an <b>explained variance score</b> of <var>{explained_variance:.2f}</var>.</p>"
    text += f"<p>The <b>root mean squared error</b> of the model is <var>{rmse:.2f}</var>.</p>"
    text += f"<p>The <b>mean absolute error</b> of the model is <var>{mae:.2f}</var>.</p>"
    return text

def about_explainer():
    text = "<p>The explainer is a SHAP Kernel Explainer.</p>"
    text += "<p>SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. \
        It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.</p>"
    return text

# Showing data

def count_all():
    return f"<p>There are <var>{len(dataset)}</var> instances in the dataset.</p>"

def count_group(indoor_temperature_min=None, indoor_temperature_max=None,
                outdoor_temperature_min=None, outdoor_temperature_max=None, 
                past_electricity_min=None, past_electricity_max=None):
    intro = _format_group(indoor_temperature_min, indoor_temperature_max, outdoor_temperature_min, outdoor_temperature_max, past_electricity_min, past_electricity_max)
    result = dataset
    
    if indoor_temperature_min:
        result = result[result['indoor_temperature'] > indoor_temperature_min]
    if indoor_temperature_max:
        result = result[result['indoor_temperature'] < indoor_temperature_max]
    if outdoor_temperature_min:
        result = result[result['outdoor_temperature'] > outdoor_temperature_min]
    if outdoor_temperature_max:
        result = result[result['outdoor_temperature'] < outdoor_temperature_max]
    if past_electricity_min:
        result = result[result['past_electricity'] > past_electricity_min]
    if past_electricity_max:
        result = result[result['past_electricity'] < past_electricity_max]
        
    if result.empty:
        return f"<p>There is no data for the selected group.</p>"
          
    return intro + f"<p>There are <var>{len(result)}</var> instances in the selected group.</p>"

def show_ids():
    return f"<p>Available <code>ID</code> values are: {', '.join([f'<var>{id}</var>' for id in dataset.index.get_level_values(0).unique().sort_values()])}.</p>"

def show_one(id):
    if (id not in dataset.index):
        return f"<p>There is no data for <code>ID</code> <var>{id}</var>.</p>"
    intro = f"<p>Here is the data for <code>ID</code> <var>{id}</var>:</p>"
    renamed = dataset.rename(columns={'outdoor_temperature': 'outdoor temperature', 'indoor_temperature': 'indoor temperature', 'past_electricity': 'past electricity'})
    framed = pd.DataFrame(renamed)
    # Convert Series to DataFrame for HTML rendering
    table = f"<p>{framed.loc[id].to_frame().T.to_html()}</p>"
    return intro + table

def show_group(indoor_temperature_min=None, indoor_temperature_max=None,
               outdoor_temperature_min=None, outdoor_temperature_max=None, 
               past_electricity_min=None, past_electricity_max=None):
    intro = _format_group(indoor_temperature_min, indoor_temperature_max, outdoor_temperature_min, outdoor_temperature_max, past_electricity_min, past_electricity_max)
    result = dataset
    
    if indoor_temperature_min:
        result = result[result['indoor_temperature'] > indoor_temperature_min]
    if indoor_temperature_max:
        result = result[result['indoor_temperature'] < indoor_temperature_max]
    if outdoor_temperature_min:
        result = result[result['outdoor_temperature'] > outdoor_temperature_min]
    if outdoor_temperature_max:
        result = result[result['outdoor_temperature'] < outdoor_temperature_max]
    if past_electricity_min:
        result = result[result['past_electricity'] > past_electricity_min]
    if past_electricity_max:
        result = result[result['past_electricity'] < past_electricity_max]
        
    if result.empty:
        return f"<p>There is no data for the selected group.</p>"
    
    intro += f"<p>Showing the data for the selected group.</p>"

    renamed = result.rename(columns={'outdoor_temperature': 'outdoor temperature', 'indoor_temperature': 'indoor temperature', 'past_electricity': 'past electricity'})
    renamed.sort_index(inplace=True)
    table = f"<p>{renamed.to_html()}</p>"
    return intro + table

# Calculating predictions

def predict_one(id):
    if (id not in dataset.index):
        return f"<p>There is no data for <code>ID</code> <var>{id}</var>.</p>"
    data = dataset.loc[id].to_frame().T
    prediction = model.predict(data)
    rounded = round(prediction[0], 2)
    text = f"<p>The prediction for <code>ID</code> <var>{id}</var> is <samp>{rounded}</samp>.</p>"
    return text

def predict_group(indoor_temperature_min=None, indoor_temperature_max=None,
               outdoor_temperature_min=None, outdoor_temperature_max=None, 
               past_electricity_min=None, past_electricity_max=None):
    intro = _format_group(indoor_temperature_min, indoor_temperature_max, outdoor_temperature_min, outdoor_temperature_max, past_electricity_min, past_electricity_max)
    
    data = dataset
    if indoor_temperature_min:
        data = data[data['indoor_temperature'] > indoor_temperature_min]
    if indoor_temperature_max:
        data = data[data['indoor_temperature'] < indoor_temperature_max]
    if outdoor_temperature_min:
        data = data[data['outdoor_temperature'] > outdoor_temperature_min]
    if outdoor_temperature_max:
        data = data[data['outdoor_temperature'] < outdoor_temperature_max]
    if past_electricity_min:
        data = data[data['past_electricity'] > past_electricity_min]
    if past_electricity_max:
        data = data[data['past_electricity'] < past_electricity_max]
        
    if data.empty:
        return f"<p>There is no data for the selected group.</p>"
        
    intro += "<p>Here are the predictions for the selected group.</p>"
        
    prediction = model.predict(data)
    framed = pd.DataFrame(prediction, columns=['prediction'], index=data.index)
    framed.sort_index(inplace=True)

    table = f"<p>{framed.to_html()}</p>"
    return intro + table

def predict_new(indoor_temperature, outdoor_temperature, past_electricity):
    data = pd.DataFrame([[indoor_temperature, outdoor_temperature, past_electricity]], columns=['indoor_temperature', 'outdoor_temperature', 'past_electricity'])
    prediction = model.predict(data)
    rounded = round(prediction[0], 2)
    text = "<p>Let's consider a new data sample with the following features:</p>"
    text += f"<p><ul>{''.join([f'<li><code>{feature}</code> = <var>{_extract_value(value)}</var></li>' for feature, value in data.to_dict().items()])}</ul><p>"
    text += f"<p>The model prediction for the new data will be <samp>{rounded}</samp>.</p>"
    return text

# Showing mistakes

def mistake_one(id):
    if (id not in dataset.index):
        return f"<p>There is no data for <code>ID</code> <var>{id}</var>.</p>"
    data = dataset.loc[id].to_frame().T
    prediction = model.predict(data)[0]
    rounded = round(prediction, 2)
    actual = y_values.loc[id]
    text = f"<p>The prediction for <code>ID</code> <var>{id}</var> is <samp>{rounded}</samp>.</p>"
    text += f"<p>The actual value is <samp>{actual}</samp>.</p>"
    text += f"<p>The error is <samp>{round(abs(actual - prediction), 2)}</samp>.</p>"
    return text

def mistake_group(indoor_temperature_min=None, indoor_temperature_max=None,
                outdoor_temperature_min=None, outdoor_temperature_max=None, 
                past_electricity_min=None, past_electricity_max=None):
     intro = _format_group(indoor_temperature_min, indoor_temperature_max, outdoor_temperature_min, outdoor_temperature_max, past_electricity_min, past_electricity_max)
     
     data = dataset
     if indoor_temperature_min:
          data = data[data['indoor_temperature'] > indoor_temperature_min]
     if indoor_temperature_max:
          data = data[data['indoor_temperature'] < indoor_temperature_max]
     if outdoor_temperature_min:
          data = data[data['outdoor_temperature'] > outdoor_temperature_min]
     if outdoor_temperature_max:
          data = data[data['outdoor_temperature'] < outdoor_temperature_max]
     if past_electricity_min:
          data = data[data['past_electricity'] > past_electricity_min]
     if past_electricity_max:
          data = data[data['past_electricity'] < past_electricity_max]
          
     if data.empty:
        return f"<p>There is no data for the selected group.</p>"
          
     intro += "<p>Here are the errors for the selected group.</p>"
     
     prediction = model.predict(data)
     labels = y_values.loc[data.index].values
     errors = abs(labels - prediction).round(2)
     framed = pd.DataFrame(errors, columns=['error'], index=data.index)
     framed['prediction'] = prediction.round(2)
     framed['actual'] = labels
     framed = framed[['actual', 'prediction', 'error']]
     framed.sort_index(inplace=True)
     
     table = f"<p>{framed.to_html()}</p>"
     return intro + table

# SHAP feature importances

def explain_one(id):
    if (id not in dataset.index):
        return f"<p>There is no data for <code>ID</code> <var>{id}</var>.</p>"
    data = dataset.loc[id].to_frame().T
    shap_values = explainer.shap_values(data, nsamples=10_000, silent=True)
    influences = shap_values.squeeze()
    result = pd.DataFrame(influences, columns=['influence'], index=dataset.columns).sort_values(by='influence', key=abs, ascending=False)
    text = f"<p>For the instance with <code>ID</code> <var>{id}</var> the feature importances are:</p>" + f"<p>{result.to_html()}</p>"
    return text

def explain_group(indoor_temperature_min=None, indoor_temperature_max=None,
               outdoor_temperature_min=None, outdoor_temperature_max=None, 
               past_electricity_min=None, past_electricity_max=None):
    intro = _format_group(indoor_temperature_min, indoor_temperature_max, outdoor_temperature_min, outdoor_temperature_max, past_electricity_min, past_electricity_max)
    
    data = dataset
    if indoor_temperature_min:
        data = data[data['indoor_temperature'] > indoor_temperature_min]
    if indoor_temperature_max:
        data = data[data['indoor_temperature'] < indoor_temperature_max]
    if outdoor_temperature_min:
        data = data[data['outdoor_temperature'] > outdoor_temperature_min]
    if outdoor_temperature_max:
        data = data[data['outdoor_temperature'] < outdoor_temperature_max]
    if past_electricity_min:
        data = data[data['past_electricity'] > past_electricity_min]
    if past_electricity_max:
        data = data[data['past_electricity'] < past_electricity_max]
        
    if data.empty:
        return f"<p>There is no data for the selected group.</p>"
        
    intro += "<p>Here are the feature importances for the selected group.</p>"
        
    result = pd.DataFrame(index=data.index, columns=[])
        
    for id in data.index:
        shap_values = explainer.shap_values(data.loc[id], nsamples=10_000, silent=True)
        influences = shap_values.squeeze()
        influences = pd.DataFrame(influences, columns=['influence'], index=dataset.columns).sort_values(by='influence', key=abs, ascending=False)
        for feature in influences.index:
            result.loc[id, f"influence of {feature}"] = influences.loc[feature, 'influence']
    
    result.sort_index(inplace=True)
    
    table = f"<p>{result.to_html()}</p>"
        
    return intro + table

# DiCe counterfactual explanations

def cfes_one(id):
    if (id not in dataset.index):
        return f"<p>There is no data for <code>ID</code> <var>{id}</var>.</p>"
    
    original_prediction = model.predict(dice_dataset.loc[[id]])[0]
    
    cfe = dice_exp.generate_counterfactuals(dice_dataset.loc[[id]],
                                            total_CFs=10,
                                            desired_range=[0, original_prediction],
                                            features_to_vary=['indoor_temperature','outdoor_temperature'])
    
    final_cfes = cfe.cf_examples_list[0].final_cfs_df
    final_cfe_ids = list(final_cfes.index)
    if 'prediction' in final_cfes.columns:
            final_cfes.pop('prediction')
    
    
    new_predictions = model.predict(final_cfes)
    
    original_instance = dice_dataset.loc[[id]]
    
    output_string = f"<p>The original prediction for the data sample with <code>ID</code> <var>{id}</var> is <samp>{str(round(original_prediction, 2))}</samp>.</p>"
    output_string += "<p>Here are some options to change the prediction of this instance."
    
    output_string += "<p>First, if you"
    transition_words = ["Further,", "Also,", "In addition,", "Furthermore,"]
    
    for i, c_id in enumerate(final_cfe_ids):
        if i < 3 and i < len(final_cfe_ids):
            if i != 0:
                output_string += f"<p>{np.random.choice(transition_words)} if you"
            output_string += _get_change_string(final_cfes.loc[[c_id]], original_instance)
            new_prediction = str(round(new_predictions[i], 2))
            output_string += f", the model will predict <samp>{new_prediction}</samp>.</p>"

    return output_string

# What-If analysis

def what_if_one(id, indoor_temperature=None, outdoor_temperature=None, past_electricity=None):
    if (id not in dataset.index):
        return f"<p>There is no data for <code>ID</code> <var>{id}</var>.</p>"
    
    original_instance = dice_dataset.loc[[id]]
    original_prediction = model.predict(original_instance)[0]
    if isinstance(original_prediction, dict):  # Extract value if prediction is a dictionary with an index
        original_prediction = list(original_prediction.values())[0]
    changed_instance = original_instance.copy()
    if indoor_temperature:
        changed_instance['indoor_temperature'] = indoor_temperature
    if outdoor_temperature:
        changed_instance['outdoor_temperature'] = outdoor_temperature
    if past_electricity:
        changed_instance['past_electricity'] = past_electricity
    new_prediction = model.predict(changed_instance)[0]
    if isinstance(new_prediction, dict):  # Extract value if prediction is a dictionary with an index
        new_prediction = list(new_prediction.values())[0]
    text = f"<p>For the data sample with <code>ID</code> <var>{id}</var>, the original features are:</p>"
    text += f"<ul>{''.join([f'<li><code>{feature}</code> = <var>{_extract_value(value)}</var></li>' for feature, value in original_instance.to_dict().items()])}</ul>"
    text += f"<p>The model predicts <samp>{str(round(original_prediction, 2))}</samp> for this instance.</p>"
    text += f"<p>Let's change the features to: <ul>{''.join([f'<li><code>{feature}</code> = <var>{_extract_value(value)}</var></li>' for feature, value in changed_instance.to_dict().items()])}</ul></p>"
    text += f"<p>Then the model will predict <samp>{str(round(new_prediction, 2))}</samp>.</p>"
    return text

# Helper functions

def _format_group(indoor_temperature_min=None, indoor_temperature_max=None,
               outdoor_temperature_min=None, outdoor_temperature_max=None, 
               past_electricity_min=None, past_electricity_max=None):
    text = "<p>Grouping the data as follows:<ul>"
    if indoor_temperature_min:
        text += f"<li><code>indoor temperature</code> is more than <var>{indoor_temperature_min}</var></li>"
    if indoor_temperature_max:
        text += f"<li><code>indoor temperature</code> is less than <var>{indoor_temperature_max}</var></li>"
    if outdoor_temperature_min:
        text += f"<li><code>outdoor temperature</code> is more than <var>{outdoor_temperature_min}</var></li>"
    if outdoor_temperature_max:
        text += f"<li><code>outdoor temperature</code> is less than <var>{outdoor_temperature_max}</var></li>"
    if past_electricity_min:
        text += f"<li><code>past electricity</code> is more than <var>{past_electricity_min}</var></li>"
    if past_electricity_max:
        text += f"<li><code>past electricity</code> is less than <var>{past_electricity_max}</var></li>"
    text += "</ul></p>"
    return text

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

def _extract_value(value):
    """Extracts the numeric value if the input is a dictionary with an index."""
    if isinstance(value, dict):
        return list(value.values())[0]
    return value