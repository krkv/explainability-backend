import json

with open('functions/show_one.json') as f:
    show_one = json.load(f)
    
show_one = json.dumps(show_one)

with open('functions/predict_one.json') as f:
    predict_one = json.load(f)
    
predict_one = json.dumps(predict_one)

with open('functions/show_group.json') as f:
    show_group = json.load(f)
    
show_group = json.dumps(show_group)

with open('functions/predict_group.json') as f:
    predict_group = json.load(f)
    
predict_group = json.dumps(predict_group)