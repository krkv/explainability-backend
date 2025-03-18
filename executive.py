import pandas as pd
import pickle
import shap
import copy
import dice_ml
import numpy as np

dataset = pd.read_csv('energy_test_data.csv', index_col=[0,1])
y_values = dataset.pop('y')

explanation_dataset = copy.deepcopy(dataset)
explanation_dataset = explanation_dataset.to_numpy()
explanation_dataset = shap.kmeans(explanation_dataset, 25)

with open('energy_gp_model.pkl', 'rb') as file:
        model = pickle.load(file)
        
explainer = shap.KernelExplainer(model.predict, explanation_dataset, link="identity")

dice_dataset = copy.deepcopy(dataset)
dice_dataset['prediction'] = model.predict(dice_dataset.to_numpy())

dice_data = dice_ml.Data(dataframe=dice_dataset, 
                         continuous_features=['outdoor_temperature', 'indoor_temperature', 'past_electricity'], 
                         outcome_name='prediction')

dice_model = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")

dice_exp = dice_ml.Dice(dice_data, dice_model, method="random")

dice_dataset.pop('prediction')
        
def format_group(indoor_temperature_min=None, indoor_temperature_max=None,
               outdoor_temperature_min=None, outdoor_temperature_max=None, 
               past_electricity_min=None, past_electricity_max=None):
    text = "<p>I am grouping the data as follows:<ul>"
    if indoor_temperature_min:
        text += f"<li>indoor temperature > {indoor_temperature_min}</li>"
    if indoor_temperature_max:
        text += f"<li>indoor temperature < {indoor_temperature_max}</li>"
    if outdoor_temperature_min:
        text += f"<li>outdoor temperature > {outdoor_temperature_min}</li>"
    if outdoor_temperature_max:
        text += f"<li>outdoor temperature < {outdoor_temperature_max}</li>"
    if past_electricity_min:
        text += f"<li>past electricity > {past_electricity_min}</li>"
    if past_electricity_max:
        text += f"<li>past electricity < {past_electricity_max}</li>"
    text += "</ul></p>"
    return text

def show_one(id):
    intro = f"<p>Showing data for ID {id}</p>"
    renamed = dataset.rename(columns={'outdoor_temperature': 'outdoor temperature', 'indoor_temperature': 'indoor temperature', 'past_electricity': 'past electricity'})
    table = f"<p>{renamed.loc[id].to_html()}</p>"
    return intro + table

def show_group(indoor_temperature_min=None, indoor_temperature_max=None,
               outdoor_temperature_min=None, outdoor_temperature_max=None, 
               past_electricity_min=None, past_electricity_max=None):
    intro = format_group(indoor_temperature_min, indoor_temperature_max, outdoor_temperature_min, outdoor_temperature_max, past_electricity_min, past_electricity_max)
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

    renamed = result.rename(columns={'outdoor_temperature': 'outdoor temperature', 'indoor_temperature': 'indoor temperature', 'past_electricity': 'past electricity'})
    renamed.sort_index(inplace=True)
    table = f"<p>{renamed.to_html()}</p>"
    return intro + table

def predict_one(id):
    data = dataset.loc[id]
    prediction = model.predict(data)
    rounded = round(prediction[0], 2)
    text = f"<p>The prediction for ID {id} is {rounded}</p>"
    return text

def predict_group(indoor_temperature_min=None, indoor_temperature_max=None,
               outdoor_temperature_min=None, outdoor_temperature_max=None, 
               past_electricity_min=None, past_electricity_max=None):
    intro = format_group(indoor_temperature_min, indoor_temperature_max, outdoor_temperature_min, outdoor_temperature_max, past_electricity_min, past_electricity_max)
    
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
        
    prediction = model.predict(data)
    framed = pd.DataFrame(prediction, columns=['prediction'], index=data.index)
    framed.sort_index(inplace=True)

    table = f"<p>{framed.to_html()}</p>"
    return intro + table

def explain_one(id):
    data = dataset.loc[id]
    shap_values = explainer.shap_values(data, nsamples=10_000, silent=True)
    influences = shap_values.squeeze()
    result = pd.DataFrame(influences, columns=['influence'], index=dataset.columns).sort_values(by='influence', key=abs, ascending=False)
    text = f"<p>For the instance with id <b>{id}</b> the feature importances are:</p>" + f"<p>{result.to_html()}</p>"
    return text

def explain_group(indoor_temperature_min=None, indoor_temperature_max=None,
               outdoor_temperature_min=None, outdoor_temperature_max=None, 
               past_electricity_min=None, past_electricity_max=None):
    intro = format_group(indoor_temperature_min, indoor_temperature_max, outdoor_temperature_min, outdoor_temperature_max, past_electricity_min, past_electricity_max)
    
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

def cfes_one(id):
    cfe = dice_exp.generate_counterfactuals(dice_dataset.loc[[id]],
                                            total_CFs=10,
                                            desired_range=[22, 8714])
    final_cfes = cfe.cf_examples_list[0].final_cfs_df
    final_cfe_ids = list(final_cfes.index)
    
    if 'prediction' in final_cfes.columns:
            final_cfes.pop('prediction')
    
    original_prediction = model.predict(dice_dataset.loc[[id]])[0]
    new_predictions = model.predict(final_cfes)
    
    original_instance = dice_dataset.loc[[id]]
    
    output_string = f"<p>The original prediction for the data sample with ID <b>{id}</b> is <b>{str(round(original_prediction, 2))}</b>.</p>"
    output_string += "<p>Here are some options to change the prediction of this instance:"
    output_string += "<ul>"
    
    output_string += "<li>First, if you"
    transition_words = ["Further,", "Also,", "In addition,", "Furthermore,"]
    
    for i, c_id in enumerate(final_cfe_ids):
        if i < 3:
            if i != 0:
                output_string += f"<li>{np.random.choice(transition_words)} if you"
            output_string += _get_change_string(final_cfes.loc[[c_id]], original_instance)
            new_prediction = str(round(new_predictions[i], 2))
            output_string += f"</em>, the model will predict <b>{new_prediction}</b>.</li>"
    
    output_string += "</ul>"

    return output_string

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
                inc_dec = " <em>increase</em>"
            else:
                inc_dec = " <em>decrease</em>"
            change_string += f"{inc_dec} {feature} to {str(round(cfe_f, 2))}"
            change_string += " and "
    # Strip off last and
    change_string = change_string[:-5]
    return change_string