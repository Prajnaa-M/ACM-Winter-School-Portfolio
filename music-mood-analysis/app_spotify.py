"""Predictive Spotify Mood Analytics Portal.

Empirical Sound Pattern Telemetry & Multiple Linear Regression Mapping.
Principal Investigator: Prajnaa M. (ACM Selection Portfolio)

Run with:  streamlit run app_spotify.py
"""

from dataclasses import dataclass
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Model definition (fitted on 114,000 Spotify tracks)
# ---------------------------------------------------------------------------

APP_TITLE = "Predictive Spotify Mood Analytics Portal"
APP_SUBTITLE = "Empirical Sound Pattern Telemetry & Multiple Linear Regression Mapping"
INVESTIGATOR = "Principal Investigator: Prajnaa M. (ACM Selection Portfolio)"

INTERCEPT = -0.09199
COEF_TEMPO = 0.000481
COEF_ENERGY = 0.1896
COEF_DANCEABILITY = 0.6804

FEATURE_COLORS: Dict[str, str] = {
    "Danceability": "#1DB954",
    "Energy": "#38BDF8",
    "Tempo": "#F59E0B",
}

CHART_BACKGROUND = "#0F172A"
CHART_GRID = "#1E293B"
CHART_TEXT = "#E2E8F0"


@dataclass(frozen=True)
class Prediction:
    """Container for one model evaluation."""

    raw_valence: float
    valence: float
    tempo_term: float
    energy_term: float
    danceability_term: float

    @property
    def was_clamped(self) -> bool:
        return self.raw_valence != self.valence


def predict_valence(tempo: float, energy: float, danceability: float) -> Prediction:
    """Evaluate Valence = -0.09199 + 0.000481*Tempo + 0.1896*Energy + 0.6804*Danceability."""
    tempo_term = COEF_TEMPO * tempo
    energy_term = COEF_ENERGY * energy
    danceability_term = COEF_DANCEABILITY * danceability
    raw_valence = INTERCEPT + tempo_term + energy_term + danceability_term
    clamped = max(0.0, min(1.0, raw_valence))
    return Prediction(raw_valence, clamped, tempo_term, energy_term, danceability_term)


def describe_mood(valence: float) -> str:
    """Translate a valence score into a plain-language mood label."""
    if valence >= 0.75:
        return "🎉 Euphoric and highly positive"
    if valence >= 0.55:
        return "😊 Upbeat and positive"
    if valence >= 0.40:
        return "😐 Balanced and neutral"
    if valence >= 0.20:
        return "😔 Subdued and melancholic"
    return "🌧️ Dark and strongly negative"


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

APP_STYLES = """
<style>
.portal-banner {background: linear-gradient(120deg, #052e16 0%, #0f172a 55%, #1e1b4b 100%); border: 1px solid rgba(29, 185, 84, 0.45); border-radius: 16px; padding: 1.7rem 2rem; margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);}
.portal-title {font-size: 2.2rem; font-weight: 800; color: #ffffff; line-height: 1.2;}
.portal-subtitle {font-size: 1.02rem; color: #86efac; margin-top: 0.45rem;}
.author-block {display: inline-block; margin-top: 1rem; background: rgba(29, 185, 84, 0.14); border: 1px solid rgba(29, 185, 84, 0.55); border-radius: 999px; padding: 0.45rem 1.1rem; color: #ffffff; font-weight: 600; font-size: 0.95rem;}
.equation-box {background: rgba(15, 23, 42, 0.85); border-left: 4px solid #1DB954; border-radius: 8px; padding: 0.8rem 1rem; color: #e2e8f0; font-family: monospace; font-size: 0.88rem; margin-bottom: 1rem;}
</style>
"""


def render_banner() -> None:
    """Render the branding banner with the author profile block."""
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    st.markdown(
        '<div class="portal-banner">'
        f'<div class="portal-title">🎧 {APP_TITLE}</div>'
        f'<div class="portal-subtitle">{APP_SUBTITLE}</div>'
        f'<div class="author-block">👩‍🔬 {INVESTIGATOR}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="equation-box">Valence = -0.09199 + (0.000481 × Tempo) + (0.1896 × Energy) '
        "+ (0.6804 × Danceability)<br>Fitted on a sample space of 114,000 distinct Spotify tracks</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def build_weight_chart(prediction: Prediction, view: str) -> go.Figure:
    """Build the horizontal bar chart of coefficient weights or live contributions."""
    if view == "Model coefficients (beta weights)":
        values = [COEF_TEMPO, COEF_ENERGY, COEF_DANCEABILITY]
        labels = ["0.000481", "0.1896", "0.6804"]
        axis_title = "Regression coefficient"
        chart_title = "Linear Beta Weights by Feature"
    else:
        values = [prediction.tempo_term, prediction.energy_term, prediction.danceability_term]
        labels = [f"{value:.4f}" for value in values]
        axis_title = "Contribution to Valence (coefficient × input)"
        chart_title = "Live Feature Contributions at Current Slider Values"

    frame = pd.DataFrame(
        {
            "Feature": ["Tempo", "Energy", "Danceability"],
            "Value": values,
            "Label": labels,
        }
    )
    figure = px.bar(
        frame,
        x="Value",
        y="Feature",
        orientation="h",
        color="Feature",
        color_discrete_map=FEATURE_COLORS,
        text="Label",
        template="plotly_dark",
        title=chart_title,
    )
    figure.update_traces(
        textposition="outside",
        textfont=dict(color=CHART_TEXT, size=14),
        marker_line_width=0,
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:.6f}<extra></extra>",
    )
    figure.update_layout(
        paper_bgcolor=CHART_BACKGROUND,
        plot_bgcolor=CHART_BACKGROUND,
        font=dict(color=CHART_TEXT, family="Inter, Segoe UI, sans-serif"),
        title=dict(font=dict(size=16)),
        showlegend=False,
        height=340,
        margin=dict(l=20, r=60, t=60, b=40),
        xaxis=dict(
            title=axis_title,
            gridcolor=CHART_GRID,
            zeroline=True,
            zerolinecolor="#475569",
            range=[0, max(values) * 1.25],
        ),
        yaxis=dict(title="", gridcolor=CHART_GRID),
    )
    return figure


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


def render_controls() -> Dict[str, float]:
    """Render the three parameter sliders and return their values."""
    st.markdown("### 🎚️ Acoustic Parameter Controls")
    tempo = st.slider(
        "Tempo (Beats Per Minute)",
        min_value=0.0,
        max_value=250.0,
        value=122.02,
        step=0.01,
        format="%.2f",
        help="Dataset median: 122.02 BPM",
    )
    energy = st.slider(
        "Energy",
        min_value=0.000,
        max_value=1.000,
        value=0.6414,
        step=0.001,
        format="%.3f",
        help="Dataset mean: 0.6414",
    )
    danceability = st.slider(
        "Danceability",
        min_value=0.000,
        max_value=1.000,
        value=0.5668,
        step=0.001,
        format="%.3f",
        help="Dataset mean: 0.5668",
    )
    st.caption("Defaults are the empirical dataset anchors: median tempo 122.02 BPM, mean energy 0.6414, mean danceability 0.5668.")
    return {"tempo": tempo, "energy": energy, "danceability": danceability}


def render_prediction(prediction: Prediction) -> None:
    """Render the predicted valence metric, progress indicator, and weight chart."""
    st.markdown("### 🔮 Real-Time Valence Prediction")
    st.metric(
        "Predicted Valence Score",
        f"{prediction.valence:.4f}",
        help="Clamped strictly between 0.0000 and 1.0000.",
    )
    st.progress(prediction.valence, text=describe_mood(prediction.valence))
    if prediction.was_clamped:
        st.warning(
            f"The unclamped model output was {prediction.raw_valence:.4f}; "
            "the displayed score is clamped to the valid valence range of 0 to 1."
        )

    view = st.radio(
        "Chart view",
        ["Model coefficients (beta weights)", "Live contributions at current inputs"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.plotly_chart(build_weight_chart(prediction, view), use_container_width=True)
    st.caption(
        "Danceability carries the largest weight in both views. Raw coefficients are expressed in each "
        "feature's own units (tempo is measured in BPM), so the live-contribution view gives the "
        "scale-aware comparison."
    )


def render_diagnostics() -> None:
    """Render the model diagnostics expander."""
    with st.expander("📊 View Model Diagnostics & Dataset Fit (114,000 Tracks)"):
        st.markdown(
            """
| Verification Metric | Value |
|---|---|
| Dataset Observations | 114,000 records |
| Multiple R-squared | 0.2694 |
| Adjusted R-squared | 0.2694 |
| Residual Standard Error (RSE) | 0.2216 |
| F-Statistic | 1.401e4 (p-value < 2.2e-16) |
"""
        )
        st.markdown(
            "**Analytical summary.** The F-statistic and its p-value below 2.2e-16 show definitively that "
            "the model explains valence better than an intercept-only baseline, and the three predictors "
            "are highly significant. However, a multiple R-squared of 0.2694 means roughly 73% of the "
            "variance in valence remains unexplained, and the residual standard error of 0.2216 is large "
            "on a 0 to 1 scale. Emotional perception in music is shaped by many interacting acoustic and "
            "contextual signals, so a multi-feature lens beyond tempo, energy, and danceability is "
            "required for sharper mood prediction."
        )


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_banner()

    left, right = st.columns([1, 1])
    with left:
        inputs = render_controls()
    prediction = predict_valence(inputs["tempo"], inputs["energy"], inputs["danceability"])
    with right:
        render_prediction(prediction)

    st.divider()
    render_diagnostics()
    st.caption(f"{INVESTIGATOR} · Multiple linear regression on 114,000 Spotify tracks")


if __name__ == "__main__":
    main()