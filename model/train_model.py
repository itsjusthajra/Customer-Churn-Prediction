import os
import sys
import json
import logging
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import build_pipeline
from utils.visualization import (
    feature_importance_chart, roc_curve_data,
    confusion_matrix_data, churn_distribution,
    monthly_charges_by_churn, contract_churn,
    tenure_distribution, satisfaction_churn
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
# Assuming the data is in a "data" folder at the same level as "model"
DATA_PATH = os.path.join(os.path.dirname(MODEL_DIR), "data", "TelcoCustomerChurn.csv")

# This file is the main training script. It loads and preprocesses the data, trains multiple models, evaluates them, and saves the best one along with relevant metrics and charts for the dashboard.
def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

# The main training function that orchestrates the entire process
def train(data_path=None):
    path = data_path or DATA_PATH
    logger.info(f"Loading data from {path}")
# Build the data pipeline: load, clean, encode, split, and scale
    X_train, X_test, y_train, y_test, scaler, encoders, feature_names, df, target_col = build_pipeline(path)
# Define candidate models with some basic hyperparameters. In a real scenario, you might want to do more extensive hyperparameter tuning.
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    }
# Dictionary to store results and trained models
    results = {}
    trained = {}
# Train each candidate model, evaluate it, and store the results
    for name, model in candidates.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        trained[name] = model
        logger.info(f"  {name}: AUC={metrics['roc_auc']}  F1={metrics['f1']}")
# Select the best model based on AUC Score
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained[best_name]
    logger.info(f"Best model: {best_name} (AUC={results[best_name]['roc_auc']})")
# Generate predictions and probabilities for the test set using the best model 
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
# Calculate feature importances for the best model (works for tree-based models and logistic regression)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])
# Generate charts data for the dashboard
    charts = {
        "feature_importance": feature_importance_chart(feature_names, importances),
        "roc_curve": roc_curve_data(y_test, y_proba),
        "confusion_matrix": confusion_matrix_data(y_test, y_pred),
        "churn_distribution": churn_distribution(df, target_col),
        "monthly_charges": monthly_charges_by_churn(df, target_col),
        "contract_churn": contract_churn(df, target_col),
        "tenure_distribution": tenure_distribution(df, target_col),
        "satisfaction_churn": satisfaction_churn(df, target_col),
    }
# Create an artifact dictionary that contains the model, encoders, scaler, feature names, evaluation metrics, and charts. This will be saved to disk and can be loaded by the dashboard for visualization and inference.
    artifact = {
        "model": best_model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_names,
        "best_model_name": best_name,
        "all_metrics": results,
        "charts": charts,
        "churn_rate": float(y_train.mean()),
        "total_customers": int(len(df)),
        "churned_customers": int(df[target_col].str.lower().isin(["yes", "churned", "1"]).sum()
                                   if df[target_col].dtype == object else (df[target_col] == 1).sum()),
    }
# Save the artifact to disk using joblib. This allows us to easily load the model and related data later for inference and visualization in the dashboard.
    out_path = os.path.join(MODEL_DIR, "churn_model.pkl")
    joblib.dump(artifact, out_path)
    logger.info(f"Model saved to {out_path}")
    return artifact

# If this script is run directly, it will execute the train function. You can optionally provide a custom data path as a command-line argument. If no argument is provided, it will use the default DATA_PATH defined at the top of the script.
if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else None
    train(data)
