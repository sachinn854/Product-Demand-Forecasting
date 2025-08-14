from .model_train_fast import train_model_fast
import os
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_full_pipeline(data_path: str, model_output_path: str = "models/best_pipeline.pkl"):
    """
    Run the complete ML pipeline from data to trained model.
    
    Args:
        data_path: Path to the training data CSV file
        model_output_path: Path where the trained pipeline will be saved
    
    Returns:
        The trained pipeline object
    """
    print("🔄 Starting Full ML Pipeline...\n")
    
    try:
        # Validate input data path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found at: {data_path}")
        
        # Train model and get complete pipeline
        print("🧠 Training model...")
        full_pipeline = train_model_fast(data_path, quick_mode=True)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        
        # Save the complete pipeline (includes feature engineering, preprocessing, and model)
        print(f"💾 Saving pipeline to: {model_output_path}")
        joblib.dump(full_pipeline, model_output_path, compress=9)
        
        # Verify the saved model
        model_size = os.path.getsize(model_output_path) / (1024 * 1024)  # Size in MB
        print(f"📁 Model saved successfully! Size: {model_size:.2f} MB")
        
        print(f"\n✅ Pipeline complete! Best model saved to: {model_output_path}")
        
        return full_pipeline
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"❌ Pipeline failed: {str(e)}")
        raise e