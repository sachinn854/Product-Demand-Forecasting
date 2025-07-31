# src/feature_engineering.py

import pandas as pd
import numpy as np

class FeatureEngineer:
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.copy()

        # Drop 'finalprice'
        df.drop(['finalprice'], axis=1, inplace=True, errors='ignore')

        # Calculate price difference
        df['price_diff'] = df['price'] - df['competitorprice']

        # Drop highly correlated columns (e.g., competitorprice)
        df.drop(['competitorprice'], axis=1, inplace=True, errors='ignore')

        # Date features: convert 'date' and extract components, then drop it
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.month
        df['dayofweek'] = df['date'].dt.dayofweek
        df['week'] = df['date'].dt.isocalendar().week
        df['year'] = df['date'].dt.year
        df.drop('date', axis=1, inplace=True, errors='ignore')

        # Clean string columns: lower case and strip spaces
        df['adcampaign'] = df['adcampaign'].astype(str).str.lower().str.strip()
        df['isweekend'] = df['isweekend'].astype(str).str.lower().str.strip()
        df['season'] = df['season'].astype(str).str.lower().str.strip()

        # Generate unitssold using modified logic for more realistic noise and non-deterministic behavior:
        def generate_unitssold(row):
            base = 5.0
            # Discount effect: slightly lower coefficient
            base += (row['discount_percent'] / 12)
            # Ad campaign effect: less impactful than before
            if row['adcampaign'] == 'tv':
                base += 1.0
            # Weekend effect: slightly reduced impact
            if row['isweekend'] == 'yes':
                base += 0.8
            # Stock level effect: adjust impact ranges
            if row['stocklevel'] < 20:
                base += 1.0
            elif row['stocklevel'] > 80:
                base -= 0.8
            # Product rating effect: slight adjustment
            base += (row['productrating'] - 5) * 0.4
            # Price difference effect: reduced impact factor
            base += (row['price_diff'] / 25)
            # Season effect: reduced impact
            if row['season'] in ['holiday', 'summer']:
                base += 0.8
            elif row['season'] in ['winter', 'autumn']:
                base -= 0.4
            # Add increased noise to introduce randomness
            noise = np.random.normal(0, 2.0)
            base += noise
            return int(np.clip(round(base), 1, 10))

        df['unitssold'] = df.apply(generate_unitssold, axis=1)

        # Bin unitssold into discrete categories
        def bin_unitssold(x):
            if x <= 3:
                return 'Low'
            elif x <= 7:
                return 'Medium'
            else:
                return 'High'
        df['unit_bin'] = df['unitssold'].apply(bin_unitssold)

        # Create interaction features
        df['price_discount_interaction'] = df['price'] * df['discount_percent']
        df['stock_supplier_interaction'] = df['stocklevel'] * df['supplierdelay(days)']

        # Generate rating category as a binned feature
        def rating_bin(rating):
            if rating < 3:
                return 'low'
            elif rating < 4:
                return 'medium'
            else:
                return 'high'
        df['rating_category'] = df['productrating'].apply(rating_bin)

        return df
