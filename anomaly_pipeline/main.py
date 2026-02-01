"""
Anomaly Detection Pipeline - Main Orchestrator
Target: Real-world Financial/Time-series Datasets
Author: Metin Mert Turan
"""

from config import PipelineConfig
from data_loader import TimeSeriesDataLoader
from feature_engineer import FeatureEngineer
from models import AnomalyModel
from drift_detector import DriftDetector
import logging
import matplotlib.pyplot as plt

# Professional Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AnomalySystem")


def plot_results(df, filename, psi_score):
    """
    Visualization Layer: Highlights detected anomalies on the signal.
    Essential for demoing the model's performance to stakeholders.
    """
    plt.figure(figsize=(15, 7))

    # Base Signal
    plt.plot(df.index, df['value'], color='steelblue', label='Price/Value', alpha=0.7, linewidth=1.5)

    # Anomalies - Marked in Red
    anomalies = df[df['pred'] == 1]
    plt.scatter(anomalies.index, anomalies['value'],
                color='red', label='Detected Anomaly', s=40, edgecolors='black', zorder=5)

    plt.title(f"Analysis for: {filename} | Drift Index (PSI): {psi_score:.4f}", fontsize=14)
    plt.xlabel("Date / Time")
    plt.ylabel("Magnitude")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_pipeline():
    """
    Orchestrates the end-to-end ML Pipeline.
    INTERVIEW NOTE: This demonstrates a full production cycle:
    Data Ingestion -> Feature Extraction -> Baseline Training -> Monitoring -> Inference.
    """
    # 1. Initialization (Dependency Injection)
    cfg = PipelineConfig()
    loader = TimeSeriesDataLoader(cfg)
    engineer = FeatureEngineer(cfg)
    model = AnomalyModel(cfg)

    # 2. Manual Data Loading
    # TODO: CSV dosyanın adını buraya yaz (Örn: "BTC_USD.csv")
    csv_name = "/home/metin-mert-turan/PycharmProjects/PythonProject/anomaly_pipeline/AAL_data.csv"

    logger.info(f"--- PHASE 1: Data Ingestion for {csv_name} ---")

    try:
        # Load and clean the manual dataset
        full_df = loader.load_from_csv(csv_name)

        # Chronological Split (70% Train / 30% Test)
        # INTERVIEW NOTE: We split chronologically to prevent 'look-ahead bias' in time-series.
        split_idx = int(len(full_df) * 0.7)
        train_df = full_df.iloc[:split_idx]
        test_df = full_df.iloc[split_idx:]

        # 3. Baseline Training
        logger.info(f"Training model on {len(train_df)} baseline samples...")
        X_train = engineer.transform(train_df, is_train=True)
        model.fit(X_train.values)
        model.save()

        # 4. Real-time Monitoring & Inference
        logger.info(f"Running inference on {len(test_df)} unseen samples...")
        X_test = engineer.transform(test_df, is_train=False)

        # MLOps Check: Concept Drift Detection using PSI
        # INTERVIEW NOTE: Monitoring the health of the model in production.
        psi_score = DriftDetector.calculate_psi(X_train.values[:, 0], X_test.values[:, 0])
        logger.info(f"Monitoring Alert - PSI Score: {psi_score:.4f}")

        if psi_score > cfg.PSI_THRESHOLD:
            logger.warning("CRITICAL: Data drift detected. Retraining cycle triggered!")

        # 5. Final Prediction
        test_df.loc[:, 'pred'] = model.predict(X_test.values)
        logger.info(f"Analysis Complete: Found {test_df['pred'].sum()} anomalies.")

        # 6. Visualization
        plot_results(test_df, csv_name, psi_score)

    except FileNotFoundError as e:
        logger.error(f"Data loading failed: {e}")
    except Exception as e:
        logger.error(f"Critical error in pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    run_pipeline()