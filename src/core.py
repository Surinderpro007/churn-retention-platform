"""Core prediction logic: model loading, risk scoring, bands and recommendations."""
import joblib
import numpy as np
import pandas as pd

from src import config
from src.config import CAT, FEATURES, HIGH_RISK, MEDIUM_RISK, NUM, TARGET  # noqa: F401  (re-exported)


def load_model(path=config.MODEL_PATH):
    return joblib.load(path)


def predict_batch(model, df: pd.DataFrame) -> np.ndarray:
    """Churn probability for every row of ``df`` (needs the FEATURES columns)."""
    return model.predict_proba(df[FEATURES])[:, 1]


def predict_risk(model, row: dict) -> float:
    """Churn probability for a single customer given as a dict."""
    return float(predict_batch(model, pd.DataFrame([row]))[0])


def risk_band(p: float) -> str:
    return "High" if p >= HIGH_RISK else "Medium" if p >= MEDIUM_RISK else "Low"


def revenue_at_risk(monthly_charges, p):
    """Expected monthly revenue lost = monthly charges x churn probability.

    Works on scalars as well as on pandas Series / numpy arrays.
    """
    return monthly_charges * p


def recommend(row: dict, p: float) -> list:
    if risk_band(p) == "Low":
        return ["Koi turant action nahi. Regular loyalty updates bhejte rahein."]
    tips = []
    if row.get("contract") == "Month-to-month":
        tips.append("Discount ke saath 1 ya 2 saal ke contract ka offer do.")
    if row.get("support_calls", 0) >= 3:
        tips.append("Senior support agent se proactive call karwao.")
    if row.get("monthly_charges", 0) > 80:
        tips.append("Cheaper plan ya bundle suggest karo.")
    if row.get("tenure_months", 99) < 12:
        tips.append("Naye customer ke liye onboarding follow-up karo.")
    return tips or ["Retention team se personal call karwao."]
