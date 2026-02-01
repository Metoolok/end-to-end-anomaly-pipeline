from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import RobustScaler
import pandas as pd


class FeatureEngineer:
    """
    Performs signal decomposition and feature normalization.
    Uses STL to isolate residuals from seasonal and trend components.
    """

    def __init__(self, config):
        self.config = config
        self.scaler = RobustScaler()

    def transform(self, df: pd.DataFrame, is_train: bool = True):
        # Extract residuals to focus on stochastic variations
        stl = STL(df['value'], period=self.config.STL_PERIOD, robust=True)
        res = stl.fit()

        features = pd.DataFrame(index=df.index)
        features['residual'] = res.resid
        features['resid_volatility'] = res.resid.rolling(window=5).std().fillna(0)

        if is_train:
            return pd.DataFrame(self.scaler.fit_transform(features), index=df.index)
        return pd.DataFrame(self.scaler.transform(features), index=df.index)