import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, apply_pca=False):
        self.apply_pca = apply_pca
        self.pca = None

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Rename supplierdelay column first
        df.rename(columns={'supplierdelay(days)': 'supplierdelay'}, inplace=True)

        # Feature: discount_percent
        df['discount_percent'] = (df['price'] - df['finalprice']) / df['price']
        df['discount_percent'] = df['discount_percent'].fillna(0)

        df['price_diff'] = df['price'] - df['finalprice']
        df['price_discount_interaction'] = df['price_diff'] * df['discount_percent']

        return df

    def fit(self, X: pd.DataFrame):
        from sklearn.decomposition import PCA
        df = self._preprocess(X)

        if self.apply_pca:
            self.pca = PCA(n_components=0.95)
            self.pca.fit(df)

    def transform(self, X: pd.DataFrame):
        df = self._preprocess(X)

        if self.apply_pca and self.pca:
            df = pd.DataFrame(self.pca.transform(df), index=df.index)

        return df

    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)
