import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.feature_engineering import FeatureEngineer
from src.preprocessing import preprocess_data


def train_model(data_path):
    # Load CSV file
    df = pd.read_csv(data_path)

    # Step 1: Preprocessing
    df = preprocess_data(df)

    # Step 2: Drop target-related leakage columns (if present)
    columns_to_drop = ['unit_bin', 'rating_category']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # Step 3: Separate target variable
    y = df['unitssold']
    X = df.drop(columns=['unitssold'])

    # Step 4: Drop features suspected of leakage BEFORE transforming
    leak_features = [
        'discount_percent', 'adcampaign', 'stocklevel',
        'productrating', 'price_diff', 'season',
        'price_discount_interaction', 'stock_supplier_interaction'
    ]
    X = X.drop(columns=[col for col in leak_features if col in X.columns], errors='ignore')

    # Step 5: Apply feature engineering (including PCA)
    fe = FeatureEngineer(apply_pca=True)
    X = fe.fit_transform(X)

    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 7: Train multiple models and compare performance
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'DecisionTree': DecisionTreeRegressor(random_state=42)
    }

    best_model = None
    best_r2 = float('-inf')
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        results[name] = round(r2, 4)

        if r2 > best_r2:
            best_model = model
            best_r2 = r2

    print("\n📊 Model Evaluation Results (R² Scores):")
    for model_name, score in results.items():
        print(f"  - {model_name}: {score}")

    # Step 8: Retrain best model on full dataset
    best_model.fit(X, y)
    return best_model
