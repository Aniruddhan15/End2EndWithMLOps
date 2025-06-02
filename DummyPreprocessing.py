import pandas as pd
import sys
import yaml
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# Load parameters from the YAML file
params = yaml.safe_load(open("param.yaml"))['preprocess']

def preprocess_data(input_file, output_file, target_col, scaler_type):
    data = pd.read_csv(input_file)

    # Encode target column if specified and present
    if target_col in data.columns:
        encoder = LabelEncoder()
        data[target_col] = encoder.fit_transform(data[target_col])
    else:
        print(f"Warning: Target column '{target_col}' not found in data.")

    # Separate features and target
    features = data.drop(columns=[target_col])
    target = data[target_col]

    # Choose scaler
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Unsupported scaler type. Use 'minmax' or 'standard'.")

    # Scale features
    scaled_features = scaler.fit_transform(features)
    scaled_data = pd.DataFrame(scaled_features, columns=features.columns)

    # Re-attach target column
    scaled_data[target_col] = target

    # Save processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    scaled_data.to_csv(output_file, header=None, index=False)

    print(f"Data preprocessed, scaled using {scaler_type}, and saved to {output_file}")

if __name__ == "__main__":
    preprocess_data(
        params['input_file'],
        params['output_file'],
        params['target_column'],
        params['scaler']
    )
