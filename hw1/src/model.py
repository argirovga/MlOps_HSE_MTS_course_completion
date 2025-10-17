import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        if isinstance(keys, str):
            self.keys = [keys]
        else:
            self.keys = keys

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.keys]

class ModelWrapper:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.df_to_score = None

    def update_scoring_file(self, _df_to_score: pd.DataFrame):
        self.df_to_score = _df_to_score

    def preprocess(self):
        numerical_features = ['amount', 'lat', 'lon', 'population_city']
        categorical_features = ['cat_id', 'us_state']
        self.df_to_score.drop(columns=[col for col in self.df_to_score.columns if col not in numerical_features + categorical_features], inplace=True)

    def inference(self, df: pd.DataFrame):
        # основной препроцессинг у меня зашит в пайп модели (можно посмотреть в ноутбуке как он выглядит `model_training/simple_training.ipynb`)
        y_pred = self.model.predict(df)

        submission = pd.DataFrame({
            'id': self.df_to_score.index,
            'target': y_pred
        })

        return submission