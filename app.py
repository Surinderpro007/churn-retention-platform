"""Streamlit entry point. Run with:  python -m streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from src import config
from src.core import load_model, predict_batch, risk_band
from src.explain import reference_row
from src.views import batch_view, dashboard, model_report, predict
from src.views.common import inject_css

st.set_page_config(page_title="Customer Retention Platform", page_icon="📉", layout="wide")
inject_css()


@st.cache_data
def load_customers() -> pd.DataFrame:
    return pd.read_csv(config.DATA_PATH)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def load_metrics() -> dict:
    with open(config.METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_eval():
    return pd.read_csv(config.EVAL_PATH) if config.EVAL_PATH.exists() else None


@st.cache_data
def score_customers(_model, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["churn_probability"] = predict_batch(_model, out)
    out["risk_band"] = out["churn_probability"].map(risk_band)
    return out


try:
    customers = load_customers()
    model = get_model()
    metrics = load_metrics()
except Exception as exc:  # missing/incompatible model or data file
    st.error("Could not load the data or the model. Run `python -m scripts.train_model` once "
             "(and `python -m scripts.generate_data` if data/customers.csv is missing), then refresh.")
    st.exception(exc)
    st.stop()

scored = score_customers(model, customers)

with st.sidebar:
    st.header("About")
    st.write("Predicts which customers are likely to leave and suggests how to keep them.")
    st.markdown(f"**Model:** {metrics.get('model_name', 'Logistic Regression')}  \n"
                f"**ROC-AUC:** {metrics.get('roc_auc', float('nan')):.2f}  \n"
                f"**Customers:** {len(customers):,}")
    st.divider()
    st.markdown("**How to use**\n"
                "1. *Dashboard*: see where churn is concentrated.\n"
                "2. *Predict Churn*: score one customer and see why.\n"
                "3. *Batch Scoring*: upload a CSV to score everyone.\n"
                "4. *Model Report*: check quality and tune the threshold.")

st.title("Customer Retention Platform")
st.markdown('<p class="subtitle">AI-powered churn prediction, explanation and retention actions</p>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Predict Churn", "Batch Scoring", "Model Report"])
with tab1:
    dashboard.render(scored)
with tab2:
    predict.render(model, customers, reference_row(customers))
with tab3:
    batch_view.render(model)
with tab4:
    model_report.render(metrics, load_eval())
