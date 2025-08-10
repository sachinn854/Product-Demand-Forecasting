import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import numpy as np

from src.feature_engineering import FeatureEngineer
from src.preprocessing import preprocess_data

def train_model(data_path):
    df = pd.read_csv(data_path)

    # Target and features
    y = df['unitssold']
    X = df.drop(columns=['unitssold'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Feature Engineering
    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)

    # Preprocessing
    X_train_processed, preprocessor = preprocess_data(X_train_fe, fit=True)
    X_test_processed, _ = preprocess_data(X_test_fe, preprocessor=preprocessor, fit=False)

    # Define models and hyperparameters
    dt_params = {
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_params = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt','log2',None]
    }

    results = {}
    best_model = None
    best_score = float('-inf')
    best_name = None

    def adjusted_r2(r2, n, p):
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    n_train, n_test = len(y_train), len(y_test)
    p = X_train_processed.shape[1]

    # Decision Tree tuning with GridSearchCV (smaller search space, so okay)
    print("\n🔍 Tuning hyperparameters for DecisionTree...")
    dt = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(
        dt,
        dt_params,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_processed, y_train)
    best_dt = grid_search.best_estimator_

    # Evaluate Decision Tree
    y_train_pred = best_dt.predict(X_train_processed)
    r2_train = r2_score(y_train, y_train_pred)
    adj_r2_train = adjusted_r2(r2_train, n_train, p)
    y_test_pred = best_dt.predict(X_test_processed)
    r2_test = r2_score(y_test, y_test_pred)
    adj_r2_test = adjusted_r2(r2_test, n_test, p)

    results['DecisionTree'] = {
        'best_params': grid_search.best_params_,
        'r2_train': round(r2_train, 4),
        'adj_r2_train': round(adj_r2_train, 4),
        'r2_test': round(r2_test, 4),
        'adj_r2_test': round(adj_r2_test, 4)
    }

    print(f"Best params for DecisionTree: {grid_search.best_params_}")
    print(f"DecisionTree Train R²: {r2_train:.4f}, Adjusted Train R²: {adj_r2_train:.4f}")
    print(f"DecisionTree Test R²: {r2_test:.4f}, Adjusted Test R²: {adj_r2_test:.4f}")

    if r2_test > best_score:
        best_score = r2_test
        best_model = best_dt
        best_name = 'DecisionTree'

    # Random Forest tuning with RandomizedSearchCV (faster)
    print("\n🔍 Tuning hyperparameters for RandomForest...")
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        rf,
        rf_params,
        n_iter=20,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train_processed, y_train)
    best_rf = random_search.best_estimator_

    # Evaluate Random Forest
    y_train_pred = best_rf.predict(X_train_processed)
    r2_train = r2_score(y_train, y_train_pred)
    adj_r2_train = adjusted_r2(r2_train, n_train, p)
    y_test_pred = best_rf.predict(X_test_processed)
    r2_test = r2_score(y_test, y_test_pred)
    adj_r2_test = adjusted_r2(r2_test, n_test, p)

    results['RandomForest'] = {
        'best_params': random_search.best_params_,
        'r2_train': round(r2_train, 4),
        'adj_r2_train': round(adj_r2_train, 4),
        'r2_test': round(r2_test, 4),
        'adj_r2_test': round(adj_r2_test, 4)
    }

    print(f"Best params for RandomForest: {random_search.best_params_}")
    print(f"RandomForest Train R²: {r2_train:.4f}, Adjusted Train R²: {adj_r2_train:.4f}")
    print(f"RandomForest Test R²: {r2_test:.4f}, Adjusted Test R²: {adj_r2_test:.4f}")

    if r2_test > best_score:
        best_score = r2_test
        best_model = best_rf
        best_name = 'RandomForest'

    print("\n📊 Summary of model performances:")
    for model_name, score_dict in results.items():
        print(f" - {model_name}:")
        print(f"     Best Params: {score_dict['best_params']}")
        print(f"     Train R²: {score_dict['r2_train']}, Adjusted Train R²: {score_dict['adj_r2_train']}")
        print(f"     Test R²: {score_dict['r2_test']}, Adjusted Test R²: {score_dict['adj_r2_test']}")

    print(f"\n🏆 Best model selected: {best_name} with Test R² = {best_score:.4f}")

    # Retrain best model on full dataset
    X_all_fe = fe.fit_transform(X)
    X_all_processed, _ = preprocess_data(X_all_fe, preprocessor=preprocessor, fit=False)
    best_model.fit(X_all_processed, y)

    # Optional: Feature importance plot
    if hasattr(best_model, 'feature_importances_'):
        feature_names = (
            X_train_processed.columns
            if hasattr(X_train_processed, 'columns')
            else [f'feat_{i}' for i in range(X_train_processed.shape[1])]
        )
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12,6))
        plt.title(f"Feature Importances - {best_name}")
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

    # Save full pipeline
    full_pipeline = Pipeline([
        ('feature_engineering', fe),
        ('preprocessing', preprocessor),
        ('model', best_model)
    ])

    # joblib.dump(full_pipeline, "models/best_pipeline.pkl")

    return full_pipeline
