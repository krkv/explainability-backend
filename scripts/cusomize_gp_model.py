import numpy as np
import pandas as pd
import pickle

from gplearn._program import _Program

from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

test_data = pd.read_csv('../data/summer_workday_test.csv')

X_test = test_data.copy().drop(['y'], axis=1)
y_test = test_data.copy()['y']

with open("gp_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
    
function_objects = loaded_model._function_set
    
function_map = {}
for func in function_objects:
    function_map[func.name] = func
    
custom_program = [
    function_map['add'],
        function_map['div'],
            function_map['add'], 2, 0,
            function_map['div'], 0, 1,
        function_map['div'],
            function_map['add'], 0, 73937.288138,
            function_map['sqrt'], 2
]

function_set = loaded_model._function_set
arities = loaded_model._arities
p_point_replace = loaded_model.p_point_replace
parsimony_coefficient = loaded_model.parsimony_coefficient
random_state = 42
n_features = X_test.shape[1]
metric = loaded_model.metric
init_depth = loaded_model.init_depth
init_method = loaded_model.init_method
const_range = loaded_model.const_range


program = _Program(
    program=custom_program,
    function_set=function_set,
    arities=arities,
    p_point_replace=p_point_replace,
    parsimony_coefficient=parsimony_coefficient,
    random_state=random_state,
    n_features=n_features,
    metric=metric,
    init_depth=init_depth,
    init_method=init_method,
    const_range=const_range
)

loaded_model._program = program
    
print(loaded_model._program)

with open("custom_gp_model.pkl", "wb") as f:
    pickle.dump(loaded_model, f)

y_pred = loaded_model.predict(X_test.values)

def compute_metrics(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return np.round(rmse, 2), np.round(mae, 2), np.round(mape, 2)

metrics = compute_metrics(y_test, y_pred)

print('RMSE:',metrics[0],' MAE:',metrics[1], ' MAPE%:',metrics[2])