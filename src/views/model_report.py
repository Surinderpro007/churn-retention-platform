"""Model Report tab: metrics at an adjustable decision threshold, curves and model comparison."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from src import config
from src.metrics import best_f1_threshold, metrics_at_threshold
from src.views.common import pct, style_fig


def _confusion_fig(cm):
    fig = go.Figure(go.Heatmap(z=cm, x=["Pred: Stay", "Pred: Churn"], y=["Actual: Stay", "Actual: Churn"],
                               text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(title="Confusion matrix")
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig)


def _roc_fig(y, prob, res):
    fpr, tpr, _ = roc_curve(y, prob)
    (tn, fp), _ = res["confusion_matrix"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color=config.ACCENT)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                             line=dict(color="#9aa0aa", dash="dash")))
    fig.add_trace(go.Scatter(x=[fp / max(fp + tn, 1)], y=[res["recall"]], mode="markers", name="Current threshold",
                             marker=dict(color=config.RISK_COLORS["High"], size=12)))
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    return style_fig(fig)


def _pr_fig(y, prob, res):
    precision, recall, _ = precision_recall_curve(y, prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Precision-Recall",
                             line=dict(color=config.ACCENT)))
    fig.add_trace(go.Scatter(x=[res["recall"]], y=[res["precision"]], mode="markers", name="Current threshold",
                             marker=dict(color=config.RISK_COLORS["High"], size=12)))
    fig.update_layout(title="Precision-Recall curve", xaxis_title="Recall", yaxis_title="Precision")
    return style_fig(fig)


def render(metrics: dict, eval_df):
    st.caption(f"Model: **{metrics.get('model_name', 'Logistic Regression')}** | "
               f"trained {metrics.get('trained_at', 'n/a')} | "
               f"train rows {metrics.get('n_train', 'n/a')} | test rows {metrics.get('n_test', 'n/a')}")

    if eval_df is None:
        # Older model files have no saved test predictions: show the static numbers only.
        st.warning("Interactive charts need test predictions. Run `python -m scripts.train_model` to create them.")
        st.json({k: v for k, v in metrics.items() if k != "cv_results"})
        return

    y, prob = eval_df["y_true"].to_numpy(), eval_df["y_prob"].to_numpy()
    best_t, best_f1 = best_f1_threshold(y, prob)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01,
                          help="A customer is predicted to churn when the probability is at or above this value. "
                               "Lower = catch more churners (higher recall) but more false alarms.")
    st.caption(f"Best F1 on the test set is {best_f1:.2f} at threshold {best_t:.2f}.")
    res = metrics_at_threshold(y, prob, threshold)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", pct(res["accuracy"]))
    m2.metric("Precision", pct(res["precision"]), help="Of customers flagged as churners, how many really churn.")
    m3.metric("Recall", pct(res["recall"]), help="Of customers who really churn, how many we catch.")
    m4.metric("F1 score", f"{res['f1']:.2f}")
    m5.metric("ROC-AUC", f"{roc_auc_score(y, prob):.2f}", help="Ranking quality, independent of the threshold.")

    c1, c2 = st.columns(2)
    c1.plotly_chart(_confusion_fig(res["confusion_matrix"]))
    c2.plotly_chart(_roc_fig(y, prob, res))
    c3, c4 = st.columns(2)
    c3.plotly_chart(_pr_fig(y, prob, res))
    with c4:
        st.markdown("**Model comparison (5-fold cross-validation on training data)**")
        cv = pd.DataFrame(metrics.get("cv_results", []))
        if cv.empty:
            st.info("No comparison saved. Run `python -m scripts.train_model`.")
        else:
            cv = cv.rename(columns={"model": "Model", "cv_roc_auc": "ROC-AUC", "cv_f1": "F1", "cv_recall": "Recall"})
            st.dataframe(cv.round(3), hide_index=True)

    st.caption("A false negative (missing a customer who will churn) is usually costlier than a false alarm, "
               "so recall matters. Lower the threshold to catch more churners.")
