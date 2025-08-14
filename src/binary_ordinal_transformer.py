import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class BinaryOrdinalTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for binary and ordinal features"""
    
    def __init__(self):
        self.binary_mappings = {}
        self.ordinal_mappings = {}
        self.fitted = False
    
    def fit(self, X, y=None):
        """Fit the transformer on the data"""
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Define binary features
        binary_features = ['isweekend', 'promocodeused']
        
        # Define ordinal features
        ordinal_features = {
            'season': ['spring', 'summer', 'autumn', 'winter'],
            'daytype': ['weekday', 'weekend']
        }
        
        # Fit binary mappings
        for col in binary_features:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                if col == 'isweekend':
                    self.binary_mappings[col] = {'yes': 1, 'no': 0}
                elif col == 'promocodeused':
                    self.binary_mappings[col] = {'yes': 1, 'no': 0}
        
        # Fit ordinal mappings
        for col, order in ordinal_features.items():
            if col in df.columns:
                order_map = {val: idx for idx, val in enumerate(order)}
                self.ordinal_mappings[col] = order_map
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform the data"""
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Transform binary features
        for col, mapping in self.binary_mappings.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].map(mapping)
                df[col] = df[col].fillna(0).astype(int)
        
        # Transform ordinal features
        for col, mapping in self.ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = df[col].map(mapping)
                # Fill missing values with the most common value (mode)
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
                df[col] = df[col].fillna(fill_val).astype(int)
        
        return df.values
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        if input_features is None:
            return None
        return input_features
