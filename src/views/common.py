"""Small UI helpers shared by all views."""
import streamlit as st

from src import config


def inr(x: float) -> str:
    return f"{config.CURRENCY} {x:,.0f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def style_fig(fig, height: int = 320):
    """Apply one consistent look to every plotly figure."""
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
                      title_font_size=15, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def badge(band: str) -> str:
    """HTML pill coloured by risk band (render with unsafe_allow_html=True)."""
    return f'<span class="badge" style="background:{config.RISK_COLORS[band]}">{band} risk</span>'


def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e6e8ee; border-radius: 12px;
            padding: 14px 16px; box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
        }
        .badge {display: inline-block; padding: 4px 16px; border-radius: 999px;
                color: #fff; font-weight: 600; font-size: .95rem;}
        .subtitle {color: #667085; margin-top: -0.6rem; margin-bottom: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
