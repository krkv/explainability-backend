from executive import *

def explain(id=None):
    return f"<p>explain({id})</p>"

def is_list_of_calls(calls):
    return calls.startswith('[') and calls.endswith(']')

def parse_calls(calls):
    results = []

    calls = calls.strip('[]').split(';')
    if len(calls) == 0:
        raise Exception("No calls found")
    for call in calls:
        try:
            result = eval(call)
            results.append(result)
        except Exception as e:
            raise Exception(f"Error parsing call: {call}")
    return '\n'.join(results)