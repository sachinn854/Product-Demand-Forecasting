from .model_train import train_model
import os
import joblib

def run_full_pipeline(data_path: str, model_output_path: str = "models/best_pipeline.pkl"):
    print("🔄 Starting Full ML Pipeline...\n")

    # Train model and get both model and preprocessor
    model = train_model(data_path)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # Save both as a tuple (model, preprocessor)
    joblib.dump((model), model_output_path, compress=8)

    print(f"\n✅ Pipeline complete! Best model saved to: {model_output_path}")
