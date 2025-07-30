from .model_train import train_and_select_best_model
import os
import joblib

def run_full_pipeline(data_path: str, model_output_path: str = "models/best_pipeline.pkl"):
    print("🔄 Starting Full ML Pipeline...\n")

    pipeline = train_and_select_best_model(data_path)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)

    print(f"\n✅ Pipeline complete! Best model saved to: {model_output_path}")