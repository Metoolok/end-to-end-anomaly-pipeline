from sklearn.ensemble import IsolationForest
import joblib

class AnomalyModel:
    """
    Model wrapper for Isolation Forest ensemble.
    Abstracts model persistence and prediction logic.
    """
    def __init__(self, config):
        self.config = config
        self.model = IsolationForest(
            contamination=config.CONTAMINATION,
            random_state=config.RANDOM_SEED
        )

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        preds = self.model.predict(X)
        return (preds == -1).astype(int)

    def save(self):
        joblib.dump(self.model, self.config.MODEL_PATH)

    def load(self):
        self.model = joblib.load(self.config.MODEL_PATH)