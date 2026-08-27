# Insight

INSIGHT is an explainable machine learning-based predictive and early-warning system designed for monitoring government infrastructure projects. Existing platforms such as PAIMANA by the Ministry of Statistics and Programme Implementation (MoSPI) maintain and monitor data on major government infrastructure projects. INSIGHT aims to build upon such monitoring by introducing a predictive approach to project risk management.

## Features

- Project Risk Prediction
- Early Warning System
- Explainable ML
- Project Progress Monitoring
- Risk Factor Identification
- Interactive Dashboard
- Centralised Project Management
- Data-Driven Decision Support

## Project Structure

```
Insight/
├── Database/
│   ├── insight
└── Datasets
    └── demo_dataset.csv
    └── test_dataset.csv
    └── train_dataset.csv
    └── README.txt
├── Models/
    └── models/
            └── cost_overrun_model.pkl
            └── time_overrun_model.pkl
            └── 01_model.ipynb
├── Results/
      └── risk_results.csv
      └── risk_results_with_alerts.csv
├── src/
      └── __pycache__/
              └── app.cpython-314.pyc
      └── app.py
      └── Insight_logo.png
├── Requirements.txt
