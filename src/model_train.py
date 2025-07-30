# src/model_train.py

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

from .preprocessing import build_preprocessor
from .feature_engineering import FeatureEngineer

import pandas as pd
import joblib


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📌 {model_name} Results:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    return {'name': model_name, 'model': model, 'r2': r2, 'rmse': rmse}


def train_and_select_best_model(data_path):
    # ===== Load data =====
    df = pd.read_csv(data_path)
    df=df.drop(columns=['productid'], errors='ignore')

    # ===== Feature Engineering =====
    fe = FeatureEngineer()
    df = fe.transform(df)

    # ===== Binary encoding =====
    df['isweekend'] = df['isweekend'].map({'yes': 1, 'no': 0})
    df['promocodeused'] = df['promocodeused'].map({'Yes': 1, 'No': 0})

    # ===== Drop rows with missing values =====
    print("🔍 Missing values per column:\n", df.isnull().sum())
    print("📊 Data shape before dropna:", df.shape)
    df = df.dropna(inplace=False)

    # ✅ Check if enough data is left
    if df.shape[0] < 5:
        raise ValueError(f"❌ Not enough data to train. Only {df.shape[0]} rows left after dropna.")

    print(f"✅ Data shape after dropna: {df.shape}")

    # ===== Target and Features =====
    X = df.drop(columns=['unit_bin'])
    y = df['unit_bin']

    # ===== Identify column types =====
    binary_cols = ['isweekend', 'promocodeused']
    categorical_cols = ['location', 'adcampaign', 'brand', 'category', 'material',
                        'warehouse', 'inventorytype', 'season', 'daytype', 'rating_category']
    numerical_cols = [col for col in X.columns if col not in categorical_cols + binary_cols]

    # ===== Split =====
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ===== Build preprocessor =====
    preprocessor = build_preprocessor(numerical_cols, categorical_cols, binary_cols)

    # ===== Define models =====
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    best_model = None
    best_rmse = float('inf')
    best_name = None

    for name, regressor in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])

        result = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)

        if result['rmse'] < best_rmse:
            best_rmse = result['rmse']
            best_model = result['model']
            best_name = result['name']

    # ===== Save the best model =====
    joblib.dump(best_model, 'best_model.pkl')
    print(f"\n✅ Best model ({best_name}) saved as best_model.pkl")

    return best_model
