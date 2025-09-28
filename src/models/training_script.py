"""
A simple training script for Vertex AI to be used with pre-built containers.
This script is a placeholder to satisfy the CustomTrainingJob API.
"""

import os
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

# Vertex AI environment variables
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR")
AIP_TRAINING_DATA_URI = os.environ.get("AIP_TRAINING_DATA_URI")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target_column', type=str, required=True)
args = parser.parse_args()

# Load data
data = pd.read_csv(AIP_TRAINING_DATA_URI)
X = data.drop(args.target_column, axis=1)
y = data[args.target_column]

# Train model
model = xgb.XGBClassifier()
model.fit(X, y)

# Save model
model.save_model(os.path.join(AIP_MODEL_DIR, "model.bst"))
