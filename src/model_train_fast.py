import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import numpy as np
import logging

from src.feature_engineering import FeatureEngineer
from src.preprocessing import preprocess_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model_fast(data_path, quick_mode=True):
    """
    Fast model training with optimized hyperparameters.
    
    Args:
        data_path: Path to the training data CSV file
        quick_mode: If True, use default good parameters (fastest)
    
    Returns:
        Complete pipeline with feature engineering, preprocessing, and best model
    """
    logger.info("Loading training data...")
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded. Shape: {df.shape}")

    # Target and features
    if 'unitssold' not in df.columns:
        raise ValueError("Target column 'unitssold' not found in the data")
    
    y = df['unitssold']
    X = df.drop(columns=['unitssold'])
    
    logger.info(f"Features: {X.shape[1]}, Target samples: {len(y)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=None
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Feature Engineering
    logger.info("Applying feature engineering...")
    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)
    
    logger.info(f"Features after engineering: {X_train_fe.shape[1]}")

    # Preprocessing
    logger.info("Preprocessing data...")
    X_train_processed, preprocessor = preprocess_data(X_train_fe, fit=True)
    X_test_processed, _ = preprocess_data(X_test_fe, preprocessor=preprocessor, fit=False)
    
    logger.info(f"Final feature shape: {X_train_processed.shape}")

    def adjusted_r2(r2, n, p):
        """Calculate adjusted R-squared"""
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    n_train, n_test = len(y_train), len(y_test)
    p = X_train_processed.shape[1]

    results = {}
    best_model = None
    best_score = float('-inf')
    best_name = None

    if quick_mode:
        print("\n⚡ Quick mode: Using optimized default parameters...")
        
        # Define models with good default parameters
        models = {
            'DecisionTree': DecisionTreeRegressor(
                max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=20, min_samples_split=5, 
                min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
            )
        }
        
        # Train and evaluate both models
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            model.fit(X_train_processed, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train_processed)
            y_test_pred = model.predict(X_test_processed)
            
            # Metrics
            r2_train = r2_score(y_train, y_train_pred)
            adj_r2_train = adjusted_r2(r2_train, n_train, p)
            r2_test = r2_score(y_test, y_test_pred)
            adj_r2_test = adjusted_r2(r2_test, n_test, p)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            results[name] = {
                'r2_train': round(r2_train, 4),
                'adj_r2_train': round(adj_r2_train, 4),
                'r2_test': round(r2_test, 4),
                'adj_r2_test': round(adj_r2_test, 4),
                'mae_test': round(mae_test, 4),
                'rmse_test': round(rmse_test, 4)
            }
            
            print(f"{name} Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")
            print(f"{name} MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}")
            
            if r2_test > best_score:
                best_score = r2_test
                best_model = model
                best_name = name
                
    else:
        print("\n🔍 Fast hyperparameter tuning...")
        
        # Reduced parameter grids for faster tuning
        dt_params = {
            'max_depth': [15, 25],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [15, 25],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt']
        }
        
        # Decision Tree tuning
        print("🌳 Tuning DecisionTree...")
        dt = DecisionTreeRegressor(random_state=42)
        dt_search = GridSearchCV(dt, dt_params, cv=2, scoring='r2', n_jobs=-1, verbose=0)
        dt_search.fit(X_train_processed, y_train)
        best_dt = dt_search.best_estimator_
        
        # Random Forest tuning
        print("🌲 Tuning RandomForest...")
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_search = RandomizedSearchCV(rf, rf_params, n_iter=6, cv=2, scoring='r2', 
                                     n_jobs=1, random_state=42, verbose=0)
        rf_search.fit(X_train_processed, y_train)
        best_rf = rf_search.best_estimator_
        
        # Evaluate both models
        models = {'DecisionTree': best_dt, 'RandomForest': best_rf}
        
        for name, model in models.items():
            y_train_pred = model.predict(X_train_processed)
            y_test_pred = model.predict(X_test_processed)
            
            r2_train = r2_score(y_train, y_train_pred)
            adj_r2_train = adjusted_r2(r2_train, n_train, p)
            r2_test = r2_score(y_test, y_test_pred)
            adj_r2_test = adjusted_r2(r2_test, n_test, p)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            results[name] = {
                'r2_train': round(r2_train, 4),
                'adj_r2_train': round(adj_r2_train, 4),
                'r2_test': round(r2_test, 4),
                'adj_r2_test': round(adj_r2_test, 4),
                'mae_test': round(mae_test, 4),
                'rmse_test': round(rmse_test, 4)
            }
            
            print(f"{name} Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")
            
            if r2_test > best_score:
                best_score = r2_test
                best_model = model
                best_name = name

    print("\n📊 Summary of model performances:")
    for model_name, score_dict in results.items():
        print(f" - {model_name}:")
        print(f"     Train R²: {score_dict['r2_train']}, Test R²: {score_dict['r2_test']}")
        print(f"     MAE: {score_dict['mae_test']}, RMSE: {score_dict['rmse_test']}")

    print(f"\n🏆 Best model selected: {best_name} with Test R² = {best_score:.4f}")

    # Retrain best model on full dataset for production
    print("\n🔄 Retraining best model on full dataset...")
    X_all_fe = fe.fit_transform(X)
    X_all_processed, final_preprocessor = preprocess_data(X_all_fe, fit=True)
    best_model.fit(X_all_processed, y)

    # Feature importance plot
    if hasattr(best_model, 'feature_importances_'):
        try:
            feature_names = (
                X_all_processed.columns.tolist()
                if hasattr(X_all_processed, 'columns')
                else [f'feat_{i}' for i in range(X_all_processed.shape[1])]
            )
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features

            plt.figure(figsize=(10,6))
            plt.title(f"Top 15 Feature Importances - {best_name}")
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("📊 Feature importance plot saved as 'feature_importance.png'")
        except Exception as e:
            print(f"⚠️ Could not create feature importance plot: {e}")

    # Create complete pipeline
    full_pipeline = Pipeline([
        ('feature_engineering', fe),
        ('preprocessing', final_preprocessor),
        ('model', best_model)
    ])

    # Save performance metrics
    performance_summary = {
        'best_model': best_name,
        'best_score': best_score,
        'all_results': results,
        'final_features': X_all_processed.shape[1],
        'training_samples': len(y)
    }
    
    # Save performance summary
    joblib.dump(performance_summary, "models/performance_summary.pkl")
    print("📊 Performance summary saved to 'models/performance_summary.pkl'")

    return full_pipeline
