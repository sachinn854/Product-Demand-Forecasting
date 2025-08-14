import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.fitted = False

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Handle date features if date column exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['month'] = df['date'].dt.month
            df['dayofweek'] = df['date'].dt.dayofweek
            df['week'] = df['date'].dt.isocalendar().week
            df['year'] = df['date'].dt.year
            # Drop the original date column after feature extraction
            df = df.drop(columns=['date'])

        # 2. Create price difference feature (if both columns exist)
        if 'price' in df.columns and 'competitorprice' in df.columns:
            df['price_diff'] = df['price'] - df['competitorprice']
            # Drop competitorprice as it's now captured in price_diff
            df = df.drop(columns=['competitorprice'])

        # 3. Drop final price if it exists (redundant with price and discount)
        if 'finalprice' in df.columns:
            df = df.drop(columns=['finalprice'])

        # 4. Create interaction features
        if 'price' in df.columns and 'discount_percent' in df.columns:
            df['price_discount_interaction'] = df['price'] * df['discount_percent']
        
        if 'stocklevel' in df.columns and 'supplierdelay(days)' in df.columns:
            df['stock_supplier_interaction'] = df['stocklevel'] * df['supplierdelay(days)']

        # 5. Create rating category
        if 'productrating' in df.columns:
            def rating_bin(rating):
                if rating < 3:
                    return 'low'
                elif 3 <= rating < 4:
                    return 'medium'
                else:
                    return 'high'
            df['rating_category'] = df['productrating'].apply(rating_bin)

        # 6. Clean categorical values and handle missing values
        categorical_columns = ['adcampaign', 'season', 'isweekend', 'daytype', 'promocodeused']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                
        # Handle missing adcampaign values
        if 'adcampaign' in df.columns:
            df['adcampaign'] = df['adcampaign'].replace(['unknown', 'unk', 'none', 'nan', ''], pd.NA)
            # Fill missing values with random sampling from existing values
            if df['adcampaign'].isnull().any():
                n_missing = df['adcampaign'].isnull().sum()
                if df['adcampaign'].dropna().shape[0] > 0:
                    random_fill = df['adcampaign'].dropna().sample(n_missing, replace=True, random_state=42)
                    random_fill.index = df[df['adcampaign'].isnull()].index
                    df.loc[df['adcampaign'].isnull(), 'adcampaign'] = random_fill

        # 7. Drop unwanted columns (only drop what we're sure about)
        columns_to_drop = ['productid']  # Only drop productid as it's not useful for prediction
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

        return df

    def fit(self, X: pd.DataFrame):
        """Fit method to maintain sklearn-like interface"""
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame):
        """Transform method that applies feature engineering"""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        return self._preprocess(X)

    def fit_transform(self, X: pd.DataFrame):
        """Combined fit and transform method"""
        return self.fit(X).transform(X)