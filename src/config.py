"""Central configuration: paths, column names, thresholds and colours.

Everything that more than one module needs lives here, so there is a single
place to change when a real dataset replaces the sample data.
"""
from pathlib import Path

# --- Paths (absolute, so the app works no matter where it is launched from) ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customers.csv"
SAMPLE_BATCH_PATH = BASE_DIR / "data" / "sample_batch.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
EVAL_PATH = MODEL_DIR / "eval_data.csv"  # held-out test labels + predicted probabilities

# --- Columns ---
ID_COL = "customer_id"  # identifier only, never used as a model feature
TARGET = "churn"
NUM = ["tenure_months", "monthly_charges", "support_calls"]
CAT = ["contract", "internet_service", "payment_method"]
FEATURES = NUM + CAT

CATEGORY_OPTIONS = {
    "contract": ["Month-to-month", "One year", "Two year"],
    "internet_service": ["Fiber", "DSL", "No Internet"],
    "payment_method": ["UPI", "Card", "Net Banking", "Cash"],
}

# --- Risk bands (probability cut-offs) ---
HIGH_RISK = 0.6
MEDIUM_RISK = 0.3

# --- Training ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- Display ---
CURRENCY = "Rs"
ACCENT = "#4f46e5"
RISK_COLORS = {"Low": "#2e9e6b", "Medium": "#e0a100", "High": "#d64545"}
RISK_ORDER = ["Low", "Medium", "High"]
