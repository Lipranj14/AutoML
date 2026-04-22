# AutoML Dashboard

An end-to-end **Automated Machine Learning** web application built with Python and Streamlit. Upload any tabular dataset, and the pipeline automatically handles preprocessing, model selection, hyperparameter tuning, and deployment — no code required.

---

## ✨ Features

### 🏋️ Training Pipeline
- **Auto task detection** — automatically identifies Classification or Regression problems
- **Data Quality Report** — detects missing values, outliers (IQR), class imbalance, and potential target leakage
- **Model Zoo** — benchmarks 5 algorithms using 3-fold cross-validation:
  - Logistic / Linear Regression
  - Random Forest
  - XGBoost
  - SVM (SVC / SVR)
  - K-Nearest Neighbors
- **Hyperparameter Tuning** — `RandomizedSearchCV` on the best model
- **Feature Importance** visualization (bar chart)
- **Full Evaluation on Hold-out Test Set**:
  - Classification → Accuracy, F1, ROC-AUC, Confusion Matrix, ROC Curve, Per-class Report
  - Regression → RMSE, MAE, MSE, R², Actual vs Predicted, Residuals Distribution
- **Model Download** — serialized to memory, downloaded only on user request (nothing saved to disk automatically)

### 🔎 What-If Single Predictor
- Dynamically generated input form based on training schema
- Instant predictions with confidence scores and class probability bar chart

### 📦 Batch Prediction
- Upload any `.pkl` model + a new dataset
- Automatic schema validation
- Prediction analytics:
  - Classification → Prediction Summary table (count, %, avg/min/max confidence)
  - Regression → Mean, Median, Max, Min metric cards
- Download results as CSV

---

## 🗂️ Project Structure

```
AutoML/
├── app.py              # Streamlit frontend — all 3 tabs
├── automl_core.py      # ML pipeline: preprocessing, training, evaluation
├── requirements.txt    # Python dependencies
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Lipranj14/AutoML.git
cd AutoML
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 🧪 Usage

### Training a Model
1. Go to the **Training Pipeline** tab
2. Upload a `.csv` or `.xlsx` dataset
3. Select the target column
4. Click **Start Auto-ML Pipeline**
5. Review metrics, feature importances, and evaluation charts
6. Download the trained model as a `.pkl` file

### Batch Predictions
1. Go to the **Batch Prediction** tab
2. Upload the downloaded `.pkl` model
3. Upload a new dataset (must have the same feature columns)
4. Click **Generate Predictions**
5. Download the results as CSV

### What-If Analysis
1. Train a model first in the Training Pipeline tab
2. Go to the **What-If Predictor** tab
3. Fill in feature values and click **Predict**

---

## 📦 Supported File Formats

| Input | Formats |
|---|---|
| Training Data | `.csv`, `.xlsx` |
| Prediction Data | `.csv`, `.xlsx` |
| Model File | `.pkl` |
| Prediction Output | `.csv` |

---

## 🛠️ Tech Stack

| Layer | Library |
|---|---|
| Frontend | Streamlit |
| ML Pipeline | scikit-learn, XGBoost |
| Data | pandas, NumPy |
| Visualization | Plotly |
| Serialization | joblib |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
