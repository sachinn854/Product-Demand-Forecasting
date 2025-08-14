import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.binary_ordinal_transformer import BinaryOrdinalTransformer

def preprocess_data(X: pd.DataFrame, preprocessor=None, fit=True):
    """
    Preprocess the data with proper handling of all feature types.
    
    Args:
        X: Input DataFrame
        preprocessor: Existing preprocessor (if any)
        fit: Whether to fit the preprocessor
    
    Returns:
        Processed DataFrame and preprocessor
    """
    df = X.copy()
    
    # Define feature categories based on your actual data
    # Categorical features that need One-Hot Encoding
    categorical_features = [
        'location', 'adcampaign', 'brand', 'category',
        'material', 'warehouse', 'inventorytype', 'rating_category'
    ]
    
    # Ordinal features that need Label/Ordinal Encoding
    ordinal_features = {
        'season': ['spring', 'summer', 'autumn', 'winter'],
        'daytype': ['weekday', 'weekend']
    }
    
    # Define feature categories based on your actual data
    # Categorical features that need One-Hot Encoding
    categorical_features = [
        'location', 'adcampaign', 'brand', 'category',
        'material', 'warehouse', 'inventorytype', 'rating_category'
    ]
    
    # Ordinal features that need Label/Ordinal Encoding
    ordinal_features = {
        'season': ['spring', 'summer', 'autumn', 'winter'],
        'daytype': ['weekday', 'weekend']
    }
    
    # Binary features that can be mapped to 0/1
    binary_features = ['isweekend', 'promocodeused']

    # Handle missing adcampaign values
    if 'adcampaign' in df.columns:
        df['adcampaign'] = df['adcampaign'].replace(['unknown', 'unk', 'none', 'nan', ''], pd.NA)
        
    # Get final categorical features (only those that exist in the data)
    categorical_features = [col for col in categorical_features if col in df.columns]
    
    # Get numeric columns (exclude target if present)
    numeric_cols = []
    for col in df.columns:
        if (col not in categorical_features + binary_features + list(ordinal_features.keys()) + ['unitssold', 'unit_bin'] 
            and df[col].dtype in ['int64', 'float64', 'int32', 'float32']):
            numeric_cols.append(col)
    
    # Drop or limit high-cardinality categorical features
    for col in categorical_features.copy():
        if col in df.columns and df[col].nunique() > 100:
            top_20 = df[col].value_counts().nlargest(20).index
            df[col] = df[col].where(df[col].isin(top_20), other='__OTHER__')

    if fit:
        # Numeric pipeline
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ])

        # Binary and ordinal pipeline
        binary_ordinal_pipeline = BinaryOrdinalTransformer()

        # Combine transformers
        transformers = []
        
        if numeric_cols:
            transformers.append(('num', numeric_pipeline, numeric_cols))
        
        if categorical_features:
            transformers.append(('cat', categorical_pipeline, categorical_features))
        
        # Use custom transformer for binary and ordinal features
        binary_ordinal_cols = binary_features + [col for col in ordinal_features.keys() if col in df.columns]
        if binary_ordinal_cols:
            transformers.append(('binary_ordinal', binary_ordinal_pipeline, binary_ordinal_cols))

        preprocessor = ColumnTransformer(transformers=transformers)
        X_processed = preprocessor.fit_transform(df)
    else:
        if preprocessor is None:
            raise ValueError("Preprocessor must be provided when fit=False")
        X_processed = preprocessor.transform(df)

    # Get column names for the processed data
    feature_names = []
    
    # Add numeric column names
    if numeric_cols:
        feature_names.extend(numeric_cols)
    
    # Add categorical column names
    if categorical_features:
        if hasattr(preprocessor.named_transformers_['cat']['encoder'], 'get_feature_names_out'):
            encoded_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
            feature_names.extend(encoded_cols)
    
    # Add binary/ordinal column names
    binary_ordinal_cols = binary_features + [col for col in ordinal_features.keys() if col in df.columns]
    if binary_ordinal_cols:
        feature_names.extend(binary_ordinal_cols)

    # Create final DataFrame
    result_df = pd.DataFrame(X_processed, columns=feature_names).astype('float32')
    
    return result_df, preprocessor
