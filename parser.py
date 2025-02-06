from executive import predict, show

def explain(id=None):
    return f"<p>explain({id})</p>"

def parse_calls(calls):
    results = ['<p>This is what I found.</p>']

    calls = calls.strip('[]').split(',')
    if len(calls) == 0:
        raise Exception("No calls found")
    for call in calls:
        try:
            result = eval(call)
            results.append(result)
        except Exception as e:
            raise Exception(f"Error parsing call: {call}")
    return '\n'.join(results)