import pandas as pd

class FeatureEngineer:
    def __init__(self):
        pass

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Convert date column and extract features
        # df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # df['month'] = df['date'].dt.month
        # df['dayofweek'] = df['date'].dt.dayofweek
        # df['week'] = df['date'].dt.isocalendar().week
        # df['year'] = df['date'].dt.year

        # Drop unhelpful/leaking columns
        columns_to_drop = ['productid', 'date', 'location', 'warehouse', 'inventorytype','price','discount_percent','launchyear','']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

        # Clean categorical values
        for col in ['adcampaign', 'season', 'isweekend', 'daytype']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()

        return df

    def fit(self, X: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame):
        return self._preprocess(X)

    def fit_transform(self, X: pd.DataFrame):
        return self._preprocess(X)