import pandas as pd
import os
import logging

logger = logging.getLogger("DataLoader")


class TimeSeriesDataLoader:
    """
    Advanced Data Loader for Financial and Industrial Time-Series.
    Supports automatic column detection for Date and Value.
    """

    def __init__(self, config):
        self.config = config

    def load_from_csv(self, file_name: str):
        """
        Manually loads a CSV file and prepares it for the pipeline.
        Targeted at financial datasets (Stock prices, Crypto, etc.)
        """
        file_path = os.path.join(self.config.BASE_DIR, "data", file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")

        df = pd.read_csv(file_path)

        time_col = next((c for c in df.columns if any(k in c.lower() for k in ['time', 'date', 'day'])), None)

        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
            logger.info(f"Using '{time_col}' as time index for {file_name}")
        else:
            logger.warning(f"No time column found in {file_name}. Using range index.")


        target_keywords = ['close', 'adj close', 'price', 'value', 'amount']
        val_col = next((c for c in df.columns if any(k == c.lower() for k in target_keywords)), None)

        if not val_col:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                val_col = numeric_cols[0]

        if val_col:
            df = df.rename(columns={val_col: 'value'})
            logger.info(f"Target column detected: '{val_col}'")
        else:
            raise ValueError(f"Could not find a numeric value column in {file_name}")

        df = df.sort_index()
        df['value'] = df['value'].ffill().bfill()

        return df[['value']]