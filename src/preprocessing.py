import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(X: pd.DataFrame, preprocessor=None, fit=True):
    df = X.copy()

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Drop or limit high-cardinality categorical features
    for col in categorical_cols:
        if df[col].nunique() > 100:
            top_20 = df[col].value_counts().nlargest(20).index
            df[col] = df[col].where(df[col].isin(top_20), other='__OTHER__')

    if fit:
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ]
        )

        X_processed = preprocessor.fit_transform(df)
    else:
        X_processed = preprocessor.transform(df)

    # Get column names
    encoded_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
    all_cols = numeric_cols + list(encoded_cols)

    return pd.DataFrame(X_processed, columns=all_cols).astype('float32'), preprocessor
