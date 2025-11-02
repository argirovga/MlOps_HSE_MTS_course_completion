import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        self.keys = [keys] if isinstance(keys, str) else list(keys)

    def fit(self, X, y=None): 
        return self

    def transform(self, X):
        return X[self.keys]

class ModelWrapper:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model = joblib.load(model_path)
        self.df_to_score: pd.DataFrame | None = None
        self.threshold = threshold

    def update_scoring_file(self, _df_to_score: pd.DataFrame):
        self.df_to_score = _df_to_score.copy()

    def preprocess(self):
        # все как было в прошлом дз
        numerical_features = ['amount', 'lat', 'lon', 'population_city']
        categorical_features = ['cat_id', 'us_state']
        keep = set(numerical_features + categorical_features)
        drop_cols = [c for c in self.df_to_score.columns if c not in keep]
        if drop_cols:
            self.df_to_score.drop(columns=drop_cols, inplace=True, errors="ignore")

    def predict_proba(self) -> np.ndarray:
        if self.df_to_score is None:
            raise ValueError("[ERROR] No data set. Call update_scoring_file() first.")
        # как и в прошлом дз у меня препроцессинг зашит в предикт это правильный подход всегда так делаю
        proba = self.model.predict_proba(self.df_to_score)[:, 1]
        return proba

    def predict_flag(self) -> np.ndarray:
        p = self.predict_proba()
        return (p >= self.threshold).astype(int)