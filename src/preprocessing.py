import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col not in categorical_cols]

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols),
        ('bin', 'passthrough', binary_cols)
    ])

    # Apply transformation
    df_transformed = preprocessor.fit_transform(df)

    # Get transformed column names
    cat_encoded_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
    final_columns = numerical_cols + list(cat_encoded_cols) + binary_cols

    # Return as DataFrame
    return pd.DataFrame(df_transformed, columns=final_columns)
