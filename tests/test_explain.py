import pandas as pd

from src import config
from src.config import FEATURES
from src.core import load_model, predict_risk
from src.explain import explain_prediction, reference_row, what_if_contracts

RISKY = dict(tenure_months=2, monthly_charges=95, support_calls=6, contract="Month-to-month",
             internet_service="Fiber", payment_method="UPI")


def _ref():
    return reference_row(pd.read_csv(config.DATA_PATH))


def test_reference_row_has_all_features():
    assert set(_ref()) == set(FEATURES)


def test_explanation_covers_every_feature_sorted_by_size():
    imp = explain_prediction(load_model(), RISKY, _ref())
    assert set(imp["feature"]) == set(FEATURES)
    assert list(imp["impact"].abs()) == sorted(imp["impact"].abs(), reverse=True)


def test_risky_features_raise_risk_and_safe_ones_lower_it():
    m, ref = load_model(), _ref()
    risky = explain_prediction(m, RISKY, ref).set_index("feature")["impact"]
    assert risky["support_calls"] > 0 and risky["tenure_months"] > 0 and risky["monthly_charges"] > 0
    safe_row = {**RISKY, "contract": "Two year", "tenure_months": 60, "support_calls": 0}
    safe = explain_prediction(m, safe_row, ref).set_index("feature")["impact"]
    assert safe["contract"] < 0 and safe["tenure_months"] < 0 and safe["support_calls"] < 0


def test_feature_equal_to_typical_has_zero_impact():
    ref = _ref()
    row = {**RISKY, "contract": ref["contract"]}
    imp = explain_prediction(load_model(), row, ref).set_index("feature")["impact"]
    assert abs(imp["contract"]) < 1e-12


def test_longer_contract_lowers_risk():
    s = what_if_contracts(load_model(), RISKY)
    assert s["Month-to-month"] > s["One year"] > s["Two year"]
    assert abs(s["Month-to-month"] - predict_risk(load_model(), RISKY)) < 1e-9
