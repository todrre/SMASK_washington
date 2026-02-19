import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns_periods, dropOriginal=False):
        self.columns_periods = columns_periods
        self.dropOriginal = dropOriginal
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, period in self.columns_periods.items():
            if col in X.columns:
                values = X[col] - 1 if col == 'month' else X[col]
                X[f'{col}_sin'] = np.sin(2 * np.pi * values / period)
                X[f'{col}_cos'] = np.cos(2 * np.pi * values / period)
        if self.dropOriginal:
            X.drop(columns=self.columns_periods.keys(), inplace=True)
        return X

class RushHourEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_std=1.45, dropOriginal=False):
        self.n_std = n_std
        self.dropOriginal = dropOriginal
    
    def fit(self, X, y):
        high_demand_hours = X.loc[y == 1, 'hour_of_day']
        self.mean_time = high_demand_hours.mean()
        self.std_time = high_demand_hours.std()
        return self
    
    def transform(self, X):
        X = X.copy()
        X['rush_hour'] = (
            (X['hour_of_day'] >= self.mean_time - self.n_std * self.std_time) & 
            (X['hour_of_day'] <= self.mean_time + self.n_std * self.std_time)
        ).astype(int)
        if self.dropOriginal:
            X.drop(columns=['hour_of_day'], inplace=True)
        return X

class DryWarmIndexEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, dropOriginal=False):
        self.dropOriginal = dropOriginal

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['dry_warm_index'] = X['temp'] * (100 - X['humidity'])
        if self.dropOriginal:
            X.drop(columns=['temp', 'humidity'], inplace=True)
        return X