"""Batch Scoring tab: upload a CSV, score every customer, download the result."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.batch import REQUIRED_COLUMNS, score_batch, validate_batch
from src.views.common import inr, pct, style_fig


def render(model):
    st.write("Upload a CSV of customers to score all of them at once and download the ranked result.")
    st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS)
               + f". Optional: {config.ID_COL}. Any other columns are kept as they are.")

    left, right = st.columns([3, 1])
    uploaded = left.file_uploader("Customer CSV", type=["csv"])
    if config.SAMPLE_BATCH_PATH.exists():
        sample_bytes = config.SAMPLE_BATCH_PATH.read_bytes()
        right.download_button("Download sample CSV", sample_bytes, file_name="sample_batch.csv", mime="text/csv")
        use_sample = right.checkbox("Try with the sample file")
    else:
        sample_bytes, use_sample = None, False

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
        except Exception as exc:  # unreadable / not a CSV
            st.error(f"Could not read this file as CSV: {exc}")
            return
    elif use_sample:
        data = pd.read_csv(config.SAMPLE_BATCH_PATH)
    else:
        st.info("Upload a file (or tick 'Try with the sample file') to start.")
        return

    errors, warnings = validate_batch(data)
    for w in warnings:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        return

    scored = score_batch(model, data)
    high = scored[scored["risk_band"] == "High"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers scored", f"{len(scored):,}")
    k2.metric("High-risk customers", f"{len(high):,}")
    k3.metric("Average churn probability", pct(scored["churn_probability"].mean()))
    k4.metric("Monthly revenue at risk", inr(scored["monthly_revenue_at_risk"].sum()))

    plot_df = scored.assign(probability_pct=scored["churn_probability"] * 100)
    fig = px.histogram(plot_df, x="probability_pct", nbins=20, color="risk_band",
                       color_discrete_map=config.RISK_COLORS, category_orders={"risk_band": config.RISK_ORDER},
                       title="Distribution of churn probability")
    fig.update_layout(xaxis_title="Churn probability (%)", yaxis_title="Customers", legend_title="Risk band")
    st.plotly_chart(style_fig(fig))

    chosen = st.multiselect("Show risk bands", config.RISK_ORDER, default=config.RISK_ORDER)
    shown = scored[scored["risk_band"].isin(chosen)].copy()
    shown["churn_probability"] = (shown["churn_probability"] * 100).round(1)
    st.dataframe(
        shown, hide_index=True,
        column_config={"churn_probability": st.column_config.ProgressColumn(
            "Churn risk (%)", min_value=0, max_value=100, format="%.1f")},
    )
    st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"),
                       file_name="scored_customers.csv", mime="text/csv")
