"""Create sample customer data (data/customers.csv) and a small batch-upload demo file.

Run from the project root:  python -m scripts.generate_data

If you have a real dataset, put it at data/customers.csv instead (same columns, or
edit NUM / CAT in src/config.py) and skip this script.
"""
import numpy as np
import pandas as pd

from src import config

N_ROWS = 2000
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    n = N_ROWS
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "tenure_months": rng.integers(1, 73, n),
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2]),
        "monthly_charges": rng.normal(65, 25, n).clip(20, 120).round(2),
        "support_calls": rng.poisson(1.5, n),
        "internet_service": rng.choice(["Fiber", "DSL", "No Internet"], n, p=[.45, .4, .15]),
        "payment_method": rng.choice(["UPI", "Card", "Net Banking", "Cash"], n),
    })
    z = (-1.2 - 0.04 * df.tenure_months + 1.3 * (df.contract == "Month-to-month")
         - 0.8 * (df.contract == "Two year") + 0.02 * (df.monthly_charges - 65)
         + 0.35 * df.support_calls + 0.4 * (df.internet_service == "Fiber"))
    df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    config.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA_PATH, index=False)
    print(f"rows: {n} | churn rate: {df.churn.mean() * 100:.1f}% -> {config.DATA_PATH}")

    # Demo file for the "Batch Scoring" tab: 60 customers without the churn label.
    df.sample(60, random_state=SEED).drop(columns=[config.TARGET]).to_csv(config.SAMPLE_BATCH_PATH, index=False)
    print(f"sample batch file -> {config.SAMPLE_BATCH_PATH}")


if __name__ == "__main__":
    main()
