# 🚀 End-to-End Anomaly Detection & Drift Monitoring Pipeline

This repository contains a professional-grade machine learning pipeline designed to detect anomalies in time-series data and monitor statistical data drift. Developed on Ubuntu, the project demonstrates a modular approach to building production-ready AI systems.

---

## 🌟 Key Features

- **Advanced Anomaly Detection**: Uses the Isolation Forest algorithm to identify outliers and suspicious patterns in time-series datasets.
- **Data Drift Monitoring (PSI)**: Implements Population Stability Index (PSI) to detect changes in data distribution over time, ensuring model reliability.
- **Modular Architecture**: Clean separation of logic into `data_loader`, `feature_engineer`, and `models` for high maintainability.
- **Container Ready**: Includes `Dockerfile` and `docker-compose.yml` for seamless deployment across environments.

---

## 📂 Project Structure

anomaly_pipeline/
├── artifacts/ # Saved model (.pkl) and scaler files
├── data/ # Raw datasets (e.g., AAL_data.csv)
├── src/
│ ├── config.py # Centralized configuration and hyperparameters
│ ├── data_loader.py # Robust data ingestion and cleaning
│ ├── drift_detector.py # PSI-based drift analysis engine
│ ├── feature_engineer.py# Signal processing and scaling logic
│ ├── models.py # Isolation Forest model implementation
│ └── main.py # Pipeline orchestration and execution
├── Dockerfile # Production environment setup
└── requirements.txt # Project dependencie





---

## 📊 Results & Visualization

The pipeline processes time-series data and generates visualizations highlighting anomalies. It also reports a **Drift Score** to alert when the model needs retraining due to statistical shifts in the input data.



---

https://github.com/user-attachments/assets/75dc7757-b9f6-4486-82fd-4a8d783686b0

<img width="1500" height="700" alt="Yapıştırılan resim" src="https://github.com/user-attachments/assets/cdba982d-a041-4d5f-816a-f20ed921dbbc" />

<img width="1500" height="700" alt="Yapıştırılan resim (2)" src="https://github.com/user-attachments/assets/a4d43558-bf53-48d6-ad9a-c44894c1510e" />

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Metoolok/end-to-end-anomaly-pipeline.git

# Navigate to the project
cd end-to-end-anomaly-pipeline

# Install dependencies
pip install -r requirements.txt

# Execute the pipeline
python src/main.py


---

## ⚡ Notes

- Ensure your Python version is compatible (e.g., 3.9+ recommended)
- Docker setup allows running the pipeline in a consistent environment:

```bash
docker-compose up --build


author:metin mert turan ai engineer

