import pandas as pd
import pickle

dataset = pd.read_csv('energy_test_data.csv', index_col=[0,1])
y_values = dataset.pop('y')

with open('energy_gp_model.pkl', 'rb') as file:
        model = pickle.load(file)

def show(id):
    intro = f"<p>Showing data for ID {id}</p>"
    table = f"<p>{dataset.loc[id].to_html()}</p>"
    return intro + table

def predict(id):
    data = dataset.loc[id]
    prediction = model.predict(data)
    rounded = round(prediction[0], 2)
    text = f"<p>The prediction for ID {id} is {rounded}</p>"
    return text
