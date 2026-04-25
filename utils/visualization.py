import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix


def _safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def churn_distribution(df: pd.DataFrame, target_col: str) -> dict:
    counts = df[target_col].value_counts()
    labels = [str(k) for k in counts.index.tolist()]
    values = [int(v) for v in counts.values.tolist()]
    return {"labels": labels, "values": values}


def monthly_charges_by_churn(df: pd.DataFrame, target_col: str) -> dict:
    charge_col = next((c for c in ["MonthlyCharge", "MonthlyCharges"] if c in df.columns), None)
    if charge_col is None:
        return {}
    groups = df.groupby(target_col)[charge_col].apply(list)
    result = {}
    for k, v in groups.items():
        result[str(k)] = [float(x) for x in v if pd.notna(x)]
    return result


def contract_churn(df: pd.DataFrame, target_col: str) -> dict:
    contract_col = next((c for c in ["Contract", "contract_type"] if c in df.columns), None)
    if contract_col is None:
        return {}
    ct = df.groupby([contract_col, target_col]).size().unstack(fill_value=0)
    return {
        "labels": ct.index.tolist(),
        "datasets": {str(col): ct[col].tolist() for col in ct.columns}
    }


def feature_importance_chart(feature_names: list, importances: np.ndarray, top_n=15) -> dict:
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [p[0] for p in pairs]
    values = [float(p[1]) for p in pairs]
    return {"labels": labels, "values": values}


def roc_curve_data(y_test, y_proba) -> dict:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr]
    }


def confusion_matrix_data(y_test, y_pred) -> dict:
    cm = confusion_matrix(y_test, y_pred)
    return {"matrix": cm.tolist()}


def tenure_distribution(df: pd.DataFrame, target_col: str) -> dict:
    tenure_col = next((c for c in ["TenureinMonths", "tenure"] if c in df.columns), None)
    if tenure_col is None:
        return {}
    bins = [0, 12, 24, 36, 48, 60, 9999]
    labels_text = ["0-12m", "13-24m", "25-36m", "37-48m", "49-60m", "60m+"]
    df = df.copy()
    df["tenure_bin"] = pd.cut(df[tenure_col], bins=bins, labels=labels_text)
    ct = df.groupby(["tenure_bin", target_col], observed=False).size().unstack(fill_value=0)
    return {
        "labels": labels_text,
        "datasets": {str(col): ct.reindex(labels_text).fillna(0)[col].tolist() for col in ct.columns}
    }

