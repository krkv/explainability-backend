from instances.energy.executive import *

def parse_calls(calls):
    results = []

    if len(calls) == 0:
        raise Exception("No calls found")
    for call in calls:
        try:
            result = eval(call)
            results.append(result)
        except Exception as e:
            raise Exception(f"Error parsing call: {call}")
    return '\n'.join(results)