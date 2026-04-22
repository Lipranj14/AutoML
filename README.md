# AutoML Dashboard

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189AB4)](https://xgboost.readthedocs.io/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://automl-lipranj-daharwal.streamlit.app/)

**An end-to-end Automated Machine Learning web application — upload any tabular dataset and get a production-ready model with zero code.**

[Live Demo](https://automl-lipranj-daharwal.streamlit.app/) 

</div>

---

## Overview

AutoML Dashboard automates the entire machine learning workflow for tabular data. From raw CSV upload to a downloadable trained model, the pipeline handles everything — preprocessing, model selection, hyperparameter tuning, and evaluation — through an interactive web interface.

**No ML knowledge required. No code. Just upload and go.**

---

## Features

### Training Pipeline
- **Auto task detection** — automatically detects Classification vs Regression
- **Data Quality Report** — scans for missing values, outliers (IQR), class imbalance, and target leakage
- **5-Model Zoo** benchmarked via 3-fold cross-validation:

  | Task | Models |
  |------|--------|
  | Classification | Logistic Regression, Random Forest, XGBoost, SVC, KNN |
  | Regression | Linear Regression, Random Forest, XGBoost, SVR, KNN |

- **Automated Hyperparameter Tuning** via `RandomizedSearchCV` on the best model
- **Feature Importance** bar chart (top 15 features)
- **Full Hold-out Evaluation** on 20% test split:

  | Classification | Regression |
  |---|---|
  | Accuracy, F1-Score, ROC-AUC | RMSE, MAE, MSE, R² |
  | Confusion Matrix (normalised) | Actual vs Predicted scatter |
  | Per-class Classification Report | Residuals Distribution |
  | ROC Curve (binary problems) | |

- **On-demand Model Download** — model serialized in memory, nothing written to disk until you click Download

### What-If Single Predictor
- Dynamic form auto-generated from the training schema
- Instant prediction with confidence scores
- Class probability bar chart (classification)

### Batch Prediction
- Upload a `.pkl` model + any new dataset
- Automatic schema validation — catches missing columns before inference
- **Prediction Analytics**: summary table with count, percentage, and confidence stats
- Download results as CSV

---

## Project Structure

```
AutoML/
├── app.py                  # Streamlit UI — Training, What-If, Batch tabs
├── automl_core.py          # Core ML pipeline (preprocessing, training, evaluation)
├── requirements.txt        # Python dependencies with version pins
├── .streamlit/
│   └── config.toml         # Streamlit dark theme & server config
├── LICENSE
└── .gitignore
```

---

## Getting Started

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Lipranj14/AutoML.git
cd AutoML

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```


---

## Usage Guide

### Step 1 — Train a Model
1. Go to the **Training Pipeline** tab
2. Upload a `.csv` or `.xlsx` dataset
3. Select the target column
4. Click **Start Auto-ML Pipeline**
5. Review the leaderboard, metrics, and evaluation charts
6. Click **Download Versioned Model (.pkl)** to save the model

### Step 2 — Run Batch Predictions
1. Go to the **Batch Prediction** tab
2. Upload the `.pkl` model you downloaded
3. Upload a new dataset with the same feature columns
4. Click **Generate Predictions**
5. Download the predictions as CSV

### Step 3 — What-If Analysis
1. After training, go to the **What-If Predictor** tab
2. Enter individual feature values
3. Click **Predict** for an instant result with confidence

---

## Deployment

This app is deployed on **Streamlit Community Cloud** (free hosting for public GitHub repos).

### Deploy your own instance

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**
4. Select your forked repo, branch `main`, and main file `app.py`
5. Click **Deploy** — live in ~2 minutes

> No server setup, no Docker, no cloud account needed.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | [Streamlit](https://streamlit.io/) |
| ML Framework | [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/) |
| Data Processing | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| Visualization | [Plotly](https://plotly.com/python/) |
| Model Serialization | [joblib](https://joblib.readthedocs.io/) |

---

## Supported Formats

| Input | Accepted Formats |
|---|---|
| Training / Prediction Data | `.csv`, `.xlsx` |
| Model File | `.pkl` |
| Prediction Output | `.csv` (download) |

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please open an issue first to discuss any significant changes.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
Made with Streamlit & scikit-learn
</div>
