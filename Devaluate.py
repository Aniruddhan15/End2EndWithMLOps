import pandas as pd
import yaml
import joblib
import json
import os
from sklearn.metrics import accuracy_score, classification_report

# Load config
params = yaml.safe_load(open("param.yaml"))['evaluate']

def evaluate_model(test_data_path, model_path, target_column, metrics_output):
    data = pd.read_csv(test_data_path, header=None)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model = joblib.load(model_path)
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
    with open(metrics_output, 'w') as f:
        json.dump({
            "accuracy": acc,
            "classification_report": report
        }, f, indent=4)

    print(f"Evaluation metrics saved to {metrics_output}")

if __name__ == "__main__":
    evaluate_model(
        params['test_data_path'],
        params['model_path'],
        params['target_column'],
        params['metrics_output']
    )
