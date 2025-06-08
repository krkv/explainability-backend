import numpy as np
import pandas as pd
import pickle

from gplearn.genetic import SymbolicRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

train_data = pd.read_csv('../data/summer_workday_train.csv')
test_data = pd.read_csv('../data/summer_workday_test.csv')

X_train = train_data.copy().drop(['y'], axis=1)
y_train = train_data.copy()['y']

X_test = test_data.copy().drop(['y'], axis=1)
y_test = test_data.copy()['y']

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

regressor = SymbolicRegressor(
    feature_names = X_train.columns,
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt'],
    random_state=42,
)

gp_model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', regressor)
])

gp_model_pipeline.fit(X_train.values, y_train.values)

print(regressor._program)

with open("gp_model.pkl", "wb") as f:
    pickle.dump(regressor, f)
    
y_pred = gp_model_pipeline.predict(X_test.values)

def compute_metrics(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return np.round(rmse, 2), np.round(mae, 2), np.round(mape, 2)

metrics = compute_metrics(y_test, y_pred)

print('RMSE:',metrics[0],' MAE:',metrics[1], ' MAPE%:',metrics[2])