🚀 End-to-End Anomaly Detection & Drift Monitoring Pipeline
This repository contains a professional-grade machine learning pipeline designed to detect anomalies in time-series data and monitor statistical Data Drift. Developed on Ubuntu, the project demonstrates a modular approach to building production-ready AI systems.

🌟 Key Features
Advanced Anomaly Detection: Uses the Isolation Forest algorithm to identify outliers and suspicious patterns in time-series datasets.

Data Drift Monitoring (PSI): Implements Population Stability Index (PSI) to detect changes in data distribution over time, ensuring model reliability.

Modular Architecture: Cleanly separated logic into data_loader, feature_engineer, and models for high maintainability.

Container Ready: Includes Dockerfile and docker-compose.yml for seamless deployment across environments.



📂 Project Structure
Plaintext
├── artifacts/          # Saved model (.pkl) and scaler files
├── data/               # Raw datasets (e.g., AAL_data.csv)
├── src/
│   ├── config.py       # Centralized configuration and hyperparameters
│   ├── data_loader.py  # Robust data ingestion and cleaning
│   ├── drift_detector.py # PSI-based drift analysis engine
│   ├── feature_engineer.py # Signal processing and scaling logic
│   ├── models.py       # Isolation Forest model implementation
│   └── main.py         # Pipeline orchestration and execution
├── Dockerfile          # Production environment setup
└── requirements.txt    # Project dependencies





📊 Results & Visualization
The pipeline processes time-series data and generates a visualization that highlights anomalies. It also reports a Drift Score to alert when the model needs retraining due to statistical shifts in the input data.

Tip: You can insert the anomaly graph screenshot from your video (around 00:24) here!





🚀 How to Run
Clone the repo: git clone https://github.com/Metoolok/end-to-end-anomaly-pipeline.git

Install dependencies: pip install -r requirements.txt

Execute the pipeline: python src/main.py
