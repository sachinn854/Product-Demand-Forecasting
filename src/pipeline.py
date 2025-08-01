# pipeline.py
from .model_train import train_model
import os
import joblib

def run_full_pipeline(data_path: str, model_output_path: str = "models/best_pipeline.pkl"):
    print("🔄 Starting Full ML Pipeline...\n")

    # Pass file path to train_model, not a DataFrame
    pipeline = train_model(data_path)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # Save the pipeline
    joblib.dump(pipeline, model_output_path)

    print(f"\n✅ Pipeline complete! Best model saved to: {model_output_path}")
