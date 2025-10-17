import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path

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
        # я не вижу смысла делать препроцессинг в отдельном скрипте, всегда делаю их либо внутри пайпа либо внутри модел врапера
        y_pred = self.model.predict(df)

        submission = pd.DataFrame({
            'id': self.df_to_score.index,
            'target': y_pred
        })

        return submission
    
    def export_top5_feature_importances(self, out_path="artifacts/top5_feature_importances.json"):
        model = self.model.named_steps["RF"]
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            raise AttributeError("feature_importances_ not found")
        idx = np.argsort(imp)[::-1][:5]
        data = {str(int(i)): float(imp[i]) for i in idx}
        return data

    def save_scores_density_plot(self, X, out_path="artifacts/scores_density.png", bins=50):
        plt.figure()
        proba = self.model.predict_proba(X)[:, 1]
        plt.hist(proba, bins=bins, density=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()