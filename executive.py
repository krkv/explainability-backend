import pandas as pd
import pickle
import shap
import copy

dataset = pd.read_csv('energy_test_data.csv', index_col=[0,1])
y_values = dataset.pop('y')

explanation_dataset = copy.deepcopy(dataset)
explanation_dataset = explanation_dataset.to_numpy()
explanation_data = shap.kmeans(explanation_dataset, 25)

with open('energy_gp_model.pkl', 'rb') as file:
        model = pickle.load(file)
        
explainer = shap.KernelExplainer(model.predict, explanation_data, link="identity")
        
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

    table = f"<p>{framed.to_html()}</p>"
    return intro + table

def explain_one(id):
    data = dataset.loc[id]
    shap_values = explainer.shap_values(data, nsamples=10_000, silent=True)
    print(shap_values)
    text = f"<p>For the instance with id <b>{id}<b> the feature importances are:</p>"
    return text
