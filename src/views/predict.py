"""Predict tab: score one customer live, explain the score and show what-if scenarios."""
import math

import plotly.graph_objects as go
import streamlit as st

from src import config
from src.core import predict_risk, recommend, revenue_at_risk, risk_band
from src.explain import FEATURE_LABELS, explain_prediction, what_if_contracts
from src.views.common import badge, inr, pct, style_fig


def _gauge(p: float, band: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": config.RISK_COLORS[band]},
               "steps": [{"range": [0, config.MEDIUM_RISK * 100], "color": "#e3f4ec"},
                         {"range": [config.MEDIUM_RISK * 100, config.HIGH_RISK * 100], "color": "#fbf0d0"},
                         {"range": [config.HIGH_RISK * 100, 100], "color": "#f8dcdc"}]},
        title={"text": "Churn probability"}))
    return style_fig(fig, height=260)


def _impact_chart(impacts):
    top = impacts.copy()
    top["text"] = top["label"] + ": " + top["value"].astype(str)
    colors = [config.RISK_COLORS["High"] if v > 0 else config.RISK_COLORS["Low"] for v in top["impact"]]
    fig = go.Figure(go.Bar(x=top["impact"] * 100, y=top["text"], orientation="h", marker_color=colors,
                           text=[f"{v * 100:+.1f}" for v in top["impact"]], textposition="outside"))
    fig.update_layout(title="What drives this score? (points vs a typical customer)",
                      xaxis_title="Red = raises risk, green = lowers risk")
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, height=300)


def render(model, df, reference: dict):
    left, right = st.columns([1, 1.5], gap="large")

    with left:
        st.subheader("Customer details")
        tmin, tmax = int(df["tenure_months"].min()), int(df["tenure_months"].max())
        cmin = float(math.floor(df["monthly_charges"].min()))
        cmax = float(math.ceil(df["monthly_charges"].max()))
        row = {
            "tenure_months": st.slider("Tenure (months)", tmin, tmax, min(6, tmax)),
            "monthly_charges": st.number_input("Monthly charges", cmin, cmax, min(max(85.0, cmin), cmax), step=1.0),
            "support_calls": st.slider("Support calls", 0, max(10, int(df["support_calls"].max())), 3),
            "contract": st.selectbox("Contract", config.CATEGORY_OPTIONS["contract"]),
            "internet_service": st.selectbox("Internet service", config.CATEGORY_OPTIONS["internet_service"]),
            "payment_method": st.selectbox("Payment method", config.CATEGORY_OPTIONS["payment_method"]),
        }

    p = predict_risk(model, row)
    band = risk_band(p)
    impacts = explain_prediction(model, row, reference)

    with right:
        st.plotly_chart(_gauge(p, band))
        st.markdown(badge(band), unsafe_allow_html=True)
        st.metric("Expected monthly revenue at risk", inr(revenue_at_risk(row["monthly_charges"], p)))

        driver = impacts.iloc[0]
        if abs(driver["impact"]) >= 0.005:
            direction = "raises" if driver["impact"] > 0 else "lowers"
            st.info(f"Biggest driver: **{FEATURE_LABELS[driver['feature']]} = {driver['value']}** "
                    f"{direction} the risk by {abs(driver['impact']) * 100:.1f} points vs a typical customer.")
        st.plotly_chart(_impact_chart(impacts))

        st.subheader("Recommended actions")
        for tip in recommend(row, p):
            st.write("- " + tip)

        st.subheader("What if the contract changes?")
        scenarios = what_if_contracts(model, row)
        cols = st.columns(len(scenarios))
        for col, (contract, prob) in zip(cols, scenarios.items()):
            if contract == row["contract"]:
                col.metric(f"{contract} (current)", pct(prob))
            else:
                col.metric(contract, pct(prob), f"{(prob - p) * 100:+.1f} pts", delta_color="inverse")
