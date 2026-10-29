"""Batch scoring: validate an uploaded customer file and score every row."""
import pandas as pd

from src.config import CAT, CATEGORY_OPTIONS, FEATURES, NUM
from src.core import predict_batch, recommend, revenue_at_risk, risk_band

REQUIRED_COLUMNS = FEATURES


def validate_batch(df: pd.DataFrame):
    """Check an uploaded file. Returns (errors, warnings); scoring is safe only if errors is empty."""
    errors, warnings = [], []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))
        return errors, warnings
    if df.empty:
        errors.append("The file has no rows.")
        return errors, warnings

    for c in NUM:
        bad = int(pd.to_numeric(df[c], errors="coerce").isna().sum())
        if bad:
            errors.append(f"Column '{c}' has {bad} empty or non-numeric value(s).")
    for c in CAT:
        nulls = int(df[c].isna().sum())
        if nulls:
            errors.append(f"Column '{c}' has {nulls} empty value(s).")
        unknown = sorted(set(df[c].dropna().astype(str)) - set(CATEGORY_OPTIONS[c]))
        if unknown:
            warnings.append(f"Column '{c}' has values the model has not seen: {', '.join(unknown)}. "
                            "They are scored as 'none of the known categories'.")
    return errors, warnings


def score_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    """Add churn_probability, risk_band, monthly_revenue_at_risk and suggested_action; riskiest first."""
    out = df.copy()
    for c in NUM:
        out[c] = pd.to_numeric(out[c])
    p = predict_batch(model, out)
    out["churn_probability"] = p
    out["risk_band"] = [risk_band(x) for x in p]
    out["monthly_revenue_at_risk"] = revenue_at_risk(out["monthly_charges"], p).round(2)
    rows = out[FEATURES].to_dict("records")
    out["suggested_action"] = [recommend(r, x)[0] for r, x in zip(rows, p)]
    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True)
