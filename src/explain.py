"""Explain *why* a customer is risky, in plain terms.

The method is model-agnostic: for each feature we swap the customer's value with
the "typical" value (median / most common) and see how much the churn probability
changes. A positive impact means that feature pushes this customer's risk UP
compared with a typical customer.
"""
import pandas as pd

from src.config import CAT, CATEGORY_OPTIONS, FEATURES, NUM
from src.core import predict_batch, predict_risk

FEATURE_LABELS = {
    "tenure_months": "Tenure (months)",
    "monthly_charges": "Monthly charges",
    "support_calls": "Support calls",
    "contract": "Contract",
    "internet_service": "Internet service",
    "payment_method": "Payment method",
}


def reference_row(df: pd.DataFrame) -> dict:
    """A 'typical' customer: median of numeric columns, most common category."""
    ref = {c: float(df[c].median()) for c in NUM}
    ref.update({c: df[c].mode().iloc[0] for c in CAT})
    return ref


def explain_prediction(model, row: dict, reference: dict) -> pd.DataFrame:
    """Per-feature impact (in probability points, e.g. 0.12 = +12 pts) vs a typical customer.

    Returns columns: feature, label, value, typical, impact - sorted by |impact|.
    """
    variants = [dict(row)]
    for f in FEATURES:
        v = dict(row)
        v[f] = reference[f]
        variants.append(v)
    probs = predict_batch(model, pd.DataFrame(variants))
    base = probs[0]
    out = pd.DataFrame({
        "feature": FEATURES,
        "label": [FEATURE_LABELS[f] for f in FEATURES],
        "value": [row[f] for f in FEATURES],
        "typical": [reference[f] for f in FEATURES],
        "impact": [float(base - probs[i + 1]) for i in range(len(FEATURES))],
    })
    out["abs_impact"] = out["impact"].abs()
    return out.sort_values("abs_impact", ascending=False).drop(columns="abs_impact").reset_index(drop=True)


def what_if_contracts(model, row: dict) -> dict:
    """Churn probability for the same customer under each contract type."""
    return {c: predict_risk(model, {**row, "contract": c}) for c in CATEGORY_OPTIONS["contract"]}
