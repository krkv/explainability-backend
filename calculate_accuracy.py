import pandas as pd

EVALUATION_LOG_FILE = 'eval_results/eval_dataset_c_gpt-5-mini.csv'

def calculate_accuracy(log_file):
    eval_log = pd.read_csv(log_file)
    eval_log = eval_log.drop_duplicates()
    correct_parses = 0
    correct_percent = 0
    expected_parses = eval_log['expected_parse'].values
    generated_parses = eval_log['generated_parse'].values
    log_size = len(expected_parses)
    for i in range(log_size):
        if expected_parses[i] == generated_parses[i]:
            correct_parses += 1
        else:
            print("Expected: " + expected_parses[i], "- Generated: " + generated_parses[i])
    print()
    print(f"Total evaluations: {log_size}, Correct parses: {correct_parses}.")
    if correct_parses > 0:
        correct_percent = round((correct_parses / log_size) * 100, 2)
        
    return str(correct_percent)

print(calculate_accuracy(EVALUATION_LOG_FILE))