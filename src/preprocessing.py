import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()

    # Separate types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Pipelines
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(df)

    # Return processed data as DataFrame
    encoded_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
    all_cols = numeric_cols + list(encoded_cols)
    return pd.DataFrame(X_processed, columns=all_cols)
