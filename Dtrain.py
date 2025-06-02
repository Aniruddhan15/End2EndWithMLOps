import pandas as pd
import yaml
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load config
params = yaml.safe_load(open("param.yaml"))['train']

def train_model(train_data_path, model_path, target_column, model_type):
    data = pd.read_csv(train_data_path, header=None)

    # Assign headers for reference
    feature_cols = list(range(data.shape[1]))
    feature_cols.remove(data.shape[1] - 1)
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Choose model
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    else:
        raise ValueError("Unsupported model type")

    # Train and save model
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_model(
        params['train_data_path'],
        params['model_path'],
        params['target_column'],
        params['model_type']
    )
