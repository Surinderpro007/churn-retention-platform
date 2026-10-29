"""Dashboard tab: portfolio overview with filters, KPIs, charts and the top at-risk customers."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.views.common import inr, pct, style_fig

TENURE_BINS = [0, 6, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"]


def _churn_bar(view: pd.DataFrame, col: str, title: str):
    g = view.groupby(col, as_index=False)[config.TARGET].mean()
    g["churn_rate"] = g[config.TARGET] * 100
    fig = px.bar(g, x=col, y="churn_rate", title=title, labels={col: "", "churn_rate": "Churn rate (%)"})
    fig.update_traces(marker_color=config.ACCENT, texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_yaxes(range=[0, g["churn_rate"].max() * 1.2])
    return style_fig(fig)


def render(df: pd.DataFrame):
    """``df`` is the customer table plus churn_probability and risk_band columns."""
    f1, f2, f3 = st.columns(3)
    contracts = f1.multiselect("Contract", config.CATEGORY_OPTIONS["contract"],
                               default=config.CATEGORY_OPTIONS["contract"])
    services = f2.multiselect("Internet service", config.CATEGORY_OPTIONS["internet_service"],
                              default=config.CATEGORY_OPTIONS["internet_service"])
    bands = f3.multiselect("Risk band", config.RISK_ORDER, default=config.RISK_ORDER)

    view = df[df["contract"].isin(contracts) & df["internet_service"].isin(services)
              & df["risk_band"].isin(bands)]
    if view.empty:
        st.info("No customers match the selected filters.")
        return

    # Customers that have not churned yet are the ones we can still save.
    active = view[view[config.TARGET] == 0]
    at_risk = active["churn_probability"] * active["monthly_charges"]
    high = active[active["risk_band"] == "High"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers", f"{len(view):,}")
    k2.metric("Churn rate", pct(view[config.TARGET].mean()))
    k3.metric("Avg monthly charges", inr(view["monthly_charges"].mean()))
    k4.metric("High-risk active customers", f"{len(high):,}",
              help="Customers who have not churned yet and have a churn probability of "
                   f"{config.HIGH_RISK:.0%} or more.")
    k5.metric("Monthly revenue at risk", inr(at_risk.sum()),
              help="Sum of (monthly charges x churn probability) over active customers.")

    c1, c2 = st.columns(2)
    c1.plotly_chart(_churn_bar(view, "contract", "Churn rate by contract"))
    c2.plotly_chart(_churn_bar(view, "internet_service", "Churn rate by internet service"))

    c3, c4 = st.columns(2)
    if active.empty:
        c3.info("No active customers in this selection.")
    else:
        counts = active["risk_band"].value_counts().reindex(config.RISK_ORDER, fill_value=0).reset_index()
        counts.columns = ["risk_band", "customers"]
        fig = px.pie(counts, names="risk_band", values="customers", hole=0.55, color="risk_band",
                     color_discrete_map=config.RISK_COLORS, title="Active customers by risk band",
                     category_orders={"risk_band": config.RISK_ORDER})
        fig.update_traces(textinfo="label+percent", sort=False)
        c3.plotly_chart(style_fig(fig))

    bucket = pd.cut(view["tenure_months"], bins=TENURE_BINS, labels=TENURE_LABELS)
    t = view.groupby(bucket, observed=True)[config.TARGET].mean().mul(100).reset_index()
    t.columns = ["tenure", "churn_rate"]
    fig = px.line(t, x="tenure", y="churn_rate", markers=True, title="Churn rate by tenure (months)",
                  labels={"tenure": "", "churn_rate": "Churn rate (%)"})
    fig.update_traces(line_color=config.ACCENT)
    c4.plotly_chart(style_fig(fig))

    st.subheader("Top 10 customers to save first")
    st.caption("Active customers (not churned yet) with the highest churn probability in the current selection.")
    top = active.sort_values("churn_probability", ascending=False).head(10)
    table = pd.DataFrame({
        "Customer": top[config.ID_COL],
        "Contract": top["contract"],
        "Tenure (months)": top["tenure_months"],
        "Monthly charges": top["monthly_charges"],
        "Support calls": top["support_calls"],
        "Churn risk (%)": (top["churn_probability"] * 100).round(1),
    })
    st.dataframe(
        table, hide_index=True,
        column_config={"Churn risk (%)": st.column_config.ProgressColumn(
            "Churn risk (%)", min_value=0, max_value=100, format="%.1f")},
    )
