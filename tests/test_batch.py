import pandas as pd

from src import config
from src.batch import score_batch, validate_batch
from src.core import load_model


def _sample():
    return pd.read_csv(config.SAMPLE_BATCH_PATH)


def test_sample_file_is_valid():
    errors, warnings = validate_batch(_sample())
    assert errors == [] and warnings == []


def test_missing_columns_are_reported():
    errors, _ = validate_batch(_sample().drop(columns=["contract", "support_calls"]))
    assert len(errors) == 1 and "contract" in errors[0] and "support_calls" in errors[0]


def test_empty_and_non_numeric_values_are_errors():
    df = _sample()
    df["tenure_months"] = df["tenure_months"].astype(object)  # a real CSV with a bad cell loads as text
    df.loc[0, "tenure_months"] = "abc"
    df.loc[1, "contract"] = None
    errors, _ = validate_batch(df)
    assert any("tenure_months" in e for e in errors) and any("contract" in e for e in errors)


def test_unknown_category_is_only_a_warning():
    df = _sample()
    df.loc[0, "payment_method"] = "Crypto"
    errors, warnings = validate_batch(df)
    assert errors == [] and any("Crypto" in w for w in warnings)


def test_empty_file_is_an_error():
    errors, _ = validate_batch(_sample().iloc[0:0])
    assert errors


def test_scored_output_is_complete_and_ranked():
    df = _sample()
    out = score_batch(load_model(), df)
    assert len(out) == len(df)
    for col in ["churn_probability", "risk_band", "monthly_revenue_at_risk", "suggested_action"]:
        assert col in out.columns
    assert out["churn_probability"].is_monotonic_decreasing
    assert set(out["risk_band"]) <= {"Low", "Medium", "High"}
    assert out["churn_probability"].between(0, 1).all()


def test_scoring_keeps_original_columns():
    out = score_batch(load_model(), _sample())
    assert config.ID_COL in out.columns
