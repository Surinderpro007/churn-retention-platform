import json

import pandas as pd

from src import config
from src.core import load_model, predict_batch, predict_risk, recommend, revenue_at_risk, risk_band

RISKY = dict(tenure_months=2, monthly_charges=95, support_calls=6, contract="Month-to-month",
             internet_service="Fiber", payment_method="UPI")
SAFE = dict(tenure_months=60, monthly_charges=40, support_calls=0, contract="Two year",
            internet_service="DSL", payment_method="Card")


def test_data_clean():
    df = pd.read_csv(config.DATA_PATH)
    assert df.isna().sum().sum() == 0 and set(df.churn) == {0, 1}


def test_risk_band_thresholds():
    assert (risk_band(0.1), risk_band(0.4), risk_band(0.8)) == ("Low", "Medium", "High")


def test_risky_customer_scores_higher():
    m = load_model()
    assert predict_risk(m, RISKY) > predict_risk(m, SAFE)


def test_batch_matches_single_prediction():
    m = load_model()
    df = pd.DataFrame([RISKY, SAFE])
    probs = predict_batch(m, df)
    assert abs(probs[0] - predict_risk(m, RISKY)) < 1e-9
    assert abs(probs[1] - predict_risk(m, SAFE)) < 1e-9


def test_recommendations_for_high_risk():
    row = dict(tenure_months=2, monthly_charges=95, support_calls=6, contract="Month-to-month")
    assert len(recommend(row, 0.9)) >= 2


def test_recommend_handles_missing_keys():
    assert recommend({}, 0.9)  # never crashes, always returns at least one action


def test_low_risk_gets_no_urgent_action():
    assert len(recommend(SAFE, 0.05)) == 1


def test_revenue_at_risk():
    assert revenue_at_risk(100.0, 0.25) == 25.0


def test_model_quality():
    m = json.load(open(config.METRICS_PATH))
    assert m["accuracy"] > 0.6 and m["recall"] > 0.5 and m["roc_auc"] > 0.7
