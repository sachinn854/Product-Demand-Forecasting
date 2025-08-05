import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib

from src.feature_engineering import FeatureEngineer
from src.preprocessing import preprocess_data

def train_model(data_path):
    df = pd.read_csv(data_path)

    # Target and features
    y = df['unitssold']
    X = df.drop(columns=['unitssold'])

    # Feature Engineering
    fe = FeatureEngineer()
    X = fe.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 1: Fit on train
    X_train_processed, preprocessor = preprocess_data(X_train, fit=True)

    # Step 2: Transform test
    X_test_processed, _ = preprocess_data(X_test, preprocessor=preprocessor, fit=False)

    # Models
    models = {
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_r2 = float('-inf')
    results = {}

    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        r2 = r2_score(y_test, y_pred)
        results[name] = round(r2, 4)

        if r2 > best_r2:
            best_model = model
            best_r2 = r2

    print("\n📊 Model Evaluation Results (R² Scores):")
    for model_name, score in results.items():
        print(f"  - {model_name}: {score}")

    # Final training on all data
    X_full_processed, _ = preprocess_data(X, preprocessor=preprocessor, fit=False)
    best_model.fit(X_full_processed, y)

    # Save the preprocessor
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    return best_model, preprocessor
