"""Streamlit frontend for cardiovascular disease screening.

This app collects patient details, sends them to the FastAPI backend, and
renders the returned prediction with a clean clinical-style UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
import time
from urllib import error, request

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENDER_OPTIONS = {
    "Female": 1,
    "Male": 2,
}

LEVEL_OPTIONS = {
    "Normal": 1,
    "Above Normal": 2,
    "Well Above Normal": 3,
}

ACTIVE_OPTIONS = {
    "No": 0,
    "Yes": 1,
}


st.set_page_config(
    page_title="CardioRisk Screening",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    """Add the visual layer for the app."""

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
            :root {
                --page: #f5f7f6;
                --panel: #ffffff;
                --panel-soft: #fbfcfb;
                --ink: #121a17;
                --muted: #66736d;
                --line: #dfe7e2;
                --line-strong: #c6d2cb;
                --green: #1f8a54;
                --green-dark: #176845;
                --amber: #c98c00;
                --red: #bf3e35;
                --shadow: 0 8px 24px rgba(18, 26, 23, 0.06);
            }

            .stApp {
                background: var(--page);
                color: var(--ink);
            }

            .block-container {
                max-width: 1240px;
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }

            section[data-testid="stSidebar"] {
                border-right: 1px solid var(--line);
                background: #f8faf8;
            }

            .hero {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                padding: 1.1rem 1.25rem;
                margin-bottom: 1rem;
                border: 1px solid var(--line);
                border-radius: 18px;
                background: linear-gradient(180deg, #ffffff 0%, #f9fbf9 100%);
                box-shadow: var(--shadow);
                animation: floatIn 220ms ease-out both;
            }

            .hero-left {
                display: flex;
                align-items: center;
                gap: 0.9rem;
            }

            .hero-badge {
                width: 48px;
                height: 48px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 14px;
                background: #eaf6ef;
                border: 1px solid #cae7d6;
                color: var(--green);
                font-weight: 900;
                font-size: 1.05rem;
            }

            .hero-title {
                margin: 0;
                font-size: 1.7rem;
                line-height: 1.1;
                font-weight: 900;
                color: var(--ink);
            }

            .hero-subtitle {
                margin: 0.25rem 0 0;
                color: var(--muted);
                font-size: 0.93rem;
                line-height: 1.45;
            }

            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.55rem 0.8rem;
                border-radius: 999px;
                border: 1px solid var(--line);
                background: #f3f6f4;
                color: #314038;
                font-size: 0.82rem;
                font-weight: 800;
                white-space: nowrap;
            }

            .section-label {
                margin: 0 0 0.3rem;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.06em;
                font-size: 0.72rem;
                font-weight: 900;
            }

            .section-title {
                margin: 0 0 0.75rem;
                color: var(--ink);
                font-size: 1.15rem;
                line-height: 1.2;
                font-weight: 900;
            }

            .section-help {
                margin: -0.35rem 0 1rem;
                color: var(--muted);
                font-size: 0.92rem;
                line-height: 1.45;
            }

            .panel {
                padding: 1rem;
                border: 1px solid var(--line);
                border-radius: 16px;
                background: var(--panel);
                box-shadow: var(--shadow);
            }

            .result-wrap {
                display: grid;
                grid-template-columns: 1fr;
                gap: 0.9rem;
            }

            .result-head {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: flex-start;
                padding-bottom: 0.85rem;
                border-bottom: 1px solid var(--line);
            }

            .risk-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .risk-value {
                margin-top: 0.25rem;
                font-size: 2.05rem;
                line-height: 1;
                font-weight: 950;
            }

            .risk-high {
                color: var(--red);
            }

            .risk-low {
                color: var(--green);
            }

            .confidence {
                padding: 0.55rem 0.8rem;
                border-radius: 999px;
                color: #fff;
                font-size: 0.82rem;
                font-weight: 900;
                white-space: nowrap;
            }

            .confidence-high {
                background: var(--red);
            }

            .confidence-low {
                background: var(--green);
            }

            .summary-box {
                padding: 0.9rem;
                border-radius: 14px;
                border: 1px solid var(--line);
                background: var(--panel-soft);
            }

            .summary-box strong {
                display: block;
                margin-bottom: 0.3rem;
                color: var(--ink);
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .summary-box p {
                margin: 0;
                color: var(--muted);
                line-height: 1.5;
                font-size: 0.95rem;
            }

            .gauge-grid {
                display: grid;
                grid-template-columns: 220px 1fr;
                gap: 1rem;
                align-items: center;
            }

            .gauge-card {
                border: 1px solid var(--line);
                border-radius: 16px;
                background: var(--panel-soft);
                padding: 0.9rem;
            }

            .gauge {
                position: relative;
                width: 190px;
                height: 95px;
                margin: 0 auto;
                overflow: hidden;
            }

            .gauge-arc {
                width: 190px;
                height: 190px;
                border-radius: 50%;
                background: conic-gradient(
                    from 270deg,
                    var(--green) 0deg 60deg,
                    var(--amber) 60deg 120deg,
                    var(--red) 120deg 180deg,
                    transparent 180deg 360deg
                );
            }

            .gauge-inner {
                position: absolute;
                inset: 20px 18px 0;
                height: 152px;
                border-radius: 50%;
                background: var(--panel-soft);
            }

            .gauge-needle {
                position: absolute;
                left: 94px;
                top: 24px;
                width: 2px;
                height: 68px;
                border-radius: 999px;
                background: var(--ink);
                transform-origin: 50% 68px;
                transform: rotate(var(--needle-angle));
                transition: transform 900ms cubic-bezier(.22,.9,.35,1);
            }

            .gauge-hub {
                position: absolute;
                left: 88px;
                top: 83px;
                width: 14px;
                height: 14px;
                border-radius: 50%;
                background: var(--ink);
                border: 3px solid #fff;
            }

            .gauge-score {
                position: absolute;
                left: 0;
                right: 0;
                top: 46px;
                text-align: center;
                font-size: 1.5rem;
                font-weight: 950;
                color: var(--ink);
            }

            .gauge-scale {
                display: flex;
                justify-content: space-between;
                margin-top: 0.3rem;
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 900;
            }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.8rem;
            }

            .metric {
                border: 1px solid var(--line);
                border-radius: 14px;
                background: var(--panel-soft);
                padding: 0.85rem;
                min-height: 84px;
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .metric-value {
                margin-top: 0.35rem;
                font-size: 1.35rem;
                line-height: 1.1;
                font-weight: 950;
                color: var(--ink);
            }

            .metric-note {
                margin-top: 0.3rem;
                color: var(--muted);
                font-size: 0.85rem;
            }

            .factor-list {
                display: grid;
                gap: 0.8rem;
            }

            .factor-row {
                display: grid;
                grid-template-columns: 120px 1fr 42px;
                gap: 0.75rem;
                align-items: center;
                padding: 0.1rem 0;
            }

            .factor-name {
                color: var(--ink);
                font-size: 0.88rem;
                font-weight: 800;
            }

            .factor-track {
                height: 8px;
                border-radius: 999px;
                overflow: hidden;
                background: #e9efeb;
            }

            .factor-fill {
                width: var(--factor-width);
                height: 100%;
                border-radius: 999px;
                background: var(--factor-color);
            }

            .factor-score {
                color: var(--muted);
                text-align: right;
                font-size: 0.82rem;
                font-weight: 900;
            }

            .trace-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.86rem;
            }

            .trace-table td {
                padding: 0.6rem 0.25rem;
                border-bottom: 1px solid var(--line);
            }

            .trace-table td:first-child {
                color: var(--muted);
                font-weight: 800;
            }

            .trace-table td:last-child {
                color: var(--ink);
                text-align: right;
                font-weight: 900;
            }

            .notice {
                margin-top: 0.9rem;
                padding: 0.85rem 0.9rem;
                border-radius: 14px;
                border: 1px solid #f1dfb0;
                background: #fffaf0;
                color: #855300;
                font-size: 0.84rem;
                line-height: 1.45;
            }

            .backend-box {
                padding: 0.85rem;
                border-radius: 14px;
                border: 1px solid var(--line);
                background: #f9fbfa;
            }

            .backend-title {
                margin: 0 0 0.45rem;
                color: var(--ink);
                font-size: 0.88rem;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .backend-meta {
                color: var(--muted);
                font-size: 0.85rem;
                line-height: 1.45;
            }

            div.stButton > button {
                width: 100%;
                min-height: 48px;
                border-radius: 12px;
                border: 1px solid var(--green);
                background: var(--green);
                color: #fff;
                font-weight: 900;
                transition: transform 150ms ease, background-color 150ms ease, box-shadow 150ms ease;
                box-shadow: 0 6px 18px rgba(31,138,84,0.12);
            }

            div.stButton > button:hover {
                background: var(--green-dark);
                border-color: var(--green-dark);
                color: #fff;
                transform: translateY(-1px);
                box-shadow: 0 10px 30px rgba(23,104,69,0.14);
            }

            .footer {
                text-align: center;
                color: var(--muted);
                font-size: 0.82rem;
                margin-top: 1.5rem;
                opacity: 0.9;
            }

            div[data-baseweb="select"] > div,
            div[data-testid="stNumberInput"] input,
            div[data-testid="stSlider"] [role="slider"] {
                border-radius: 12px;
            }

            .fade-in {
                animation: floatIn 220ms ease-out both;
            }

            @keyframes floatIn {
                from {
                    opacity: 0;
                    transform: translateY(8px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @media (max-width: 820px) {
                .hero,
                .result-head,
                .gauge-grid,
                .metric-grid,
                .factor-row {
                    grid-template-columns: 1fr;
                }

                .hero {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .gauge {
                    width: 175px;
                }

                .risk-value {
                    font-size: 1.7rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Show the page heading."""

    st.markdown(
        """
        <div class="hero fade-in">
            <div class="hero-left">
                <div class="hero-badge">CR</div>
                <div>
                    <h1 class="hero-title">Cardiovascular Risk Screening</h1>
                    <p class="hero-subtitle">Streamlit front end that sends patient data to the FastAPI backend and renders the returned score.</p>
                </div>
            </div>
            <div class="status-pill">Frontend only · Backend via API</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(eyebrow: str, title: str, help_text: str | None = None) -> None:
    """Render a consistent section header."""

    st.markdown(f'<p class="section-label">{eyebrow}</p>', unsafe_allow_html=True)
    st.markdown(f'<h3 class="section-title">{title}</h3>', unsafe_allow_html=True)
    if help_text:
        st.markdown(f'<p class="section-help">{help_text}</p>', unsafe_allow_html=True)


def classify_bmi(bmi: float) -> str:
    """Return a simple BMI category."""

    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def classify_bp(ap_hi: int, ap_lo: int) -> str:
    """Return a blood-pressure summary."""

    if ap_hi >= 140 or ap_lo >= 90:
        return "Hypertensive range"
    if ap_hi >= 120 or ap_lo >= 80:
        return "Elevated range"
    return "Normal range"


def classify_pulse_pressure(pulse_pressure: int) -> str:
    """Return a pulse-pressure summary."""

    if pulse_pressure < 40:
        return "Low"
    if pulse_pressure <= 60:
        return "Normal"
    return "Elevated"


def color_for_score(score: float) -> str:
    """Map a score to a display color."""

    if score >= 70:
        return "#bf3e35"
    if score >= 45:
        return "#c98c00"
    return "#1f8a54"


def clamp_score(score: float) -> float:
    """Keep the visual score inside a 0-100 range."""

    return max(0.0, min(100.0, score))


def build_client_payload(
    age_years: float,
    gender_label: str,
    height_cm: float,
    weight_kg: float,
    ap_hi: int,
    ap_lo: int,
    cholesterol_label: str,
    glucose_label: str,
    active_label: str,
) -> dict[str, float | int]:
    """Create the request body for the API."""

    return {
        "age_years": age_years,
        "gender": GENDER_OPTIONS[gender_label],
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": LEVEL_OPTIONS[cholesterol_label],
        "gluc": LEVEL_OPTIONS[glucose_label],
        "active": ACTIVE_OPTIONS[active_label],
    }


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Calculate BMI for the local preview cards."""

    height_in_meters = height_cm / 100
    return round(weight_kg / (height_in_meters ** 2), 1)


def calculate_pulse_pressure(ap_hi: int, ap_lo: int) -> int:
    """Calculate pulse pressure for the local preview cards."""

    return ap_hi - ap_lo


def make_factor_profile(
    age_years: float,
    ap_hi: int,
    ap_lo: int,
    bmi: float,
    cholesterol_label: str,
    glucose_label: str,
    active_label: str,
) -> list[tuple[str, float]]:
    """Build a quick visual profile for the submitted case."""

    cholesterol_score = {"Normal": 22, "Above Normal": 58, "Well Above Normal": 88}[cholesterol_label]
    glucose_score = {"Normal": 20, "Above Normal": 55, "Well Above Normal": 84}[glucose_label]
    activity_score = 25 if active_label == "Yes" else 68

    return [
        ("Age", clamp_score((age_years - 35) * 2.0)),
        ("Systolic BP", clamp_score((ap_hi - 105) * 1.45)),
        ("Diastolic BP", clamp_score((ap_lo - 65) * 2.0)),
        ("BMI", clamp_score((bmi - 20) * 5.0)),
        ("Cholesterol", cholesterol_score),
        ("Glucose", glucose_score),
        ("Activity", activity_score),
    ]


def render_metric(label: str, value: str, note: str) -> None:
    """Render one compact metric card."""

    st.markdown(
        f"""
        <div class="metric fade-in">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_factor_rows(factors: list[tuple[str, float]]) -> None:
    """Render the factor profile bars."""

    for label, score in factors:
        st.markdown(
            f"""
            <div class="factor-row fade-in">
                <div class="factor-name">{label}</div>
                <div class="factor-track">
                    <div class="factor-fill" style="--factor-width: {score:.0f}%; --factor-color: {color_for_score(score)};"></div>
                </div>
                <div class="factor-score">{score:.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_trace(result: dict[str, Any] | None) -> None:
    """Render the backend response payload."""

    if result is None:
        rows = [
            ("Result", "Waiting"),
            ("Connection", "API required"),
            ("Derived values", "BMI, pulse pressure"),
        ]
    else:
        rows = [
            ("Prediction", result.get("prediction", "-")),
            ("Risk label", result.get("risk_label", "-")),
            ("Probability", f"{float(result.get('probability') or 0.0):.4f}"),
            ("Probability %", result.get("probability_percent", "-")),
            ("BMI", result.get("model_features", {}).get("bmi", "-")),
            ("Pulse pressure", result.get("model_features", {}).get("pulse_pressure", "-")),
        ]

    table = "".join(f"<tr><td>{left}</td><td>{right}</td></tr>" for left, right in rows)

    st.markdown(
        f"""
        <table class="trace-table fade-in">
            {table}
        </table>
        """,
        unsafe_allow_html=True,
    )


def get_backend_health(base_url: str) -> tuple[bool, str]:
    """Check whether the FastAPI service is available."""

    health_url = f"{base_url.rstrip('/')}/health"
    health_request = request.Request(health_url, method="GET")

    try:
        with request.urlopen(health_request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            service_name = str(payload.get("service", "FastAPI backend"))
            return True, service_name
    except Exception:
        return False, "FastAPI backend"


def send_prediction(base_url: str, payload: dict[str, float | int]) -> dict[str, Any]:
    """Send one prediction request to the API."""

    predict_url = f"{base_url.rstrip('/')}/predict"
    http_request = request.Request(
        predict_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API request failed ({exc.code}): {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError("Could not connect to the FastAPI backend.") from exc


def render_backend_panel(api_base_url: str) -> None:
    """Show backend status in the sidebar."""

    is_online, service_name = get_backend_health(api_base_url)

    st.markdown('<div class="backend-box">', unsafe_allow_html=True)
    st.markdown('<div class="backend-title">Backend status</div>', unsafe_allow_html=True)

    if is_online:
        st.success(f"{service_name} is reachable")
        st.caption(f"API base URL: {api_base_url}")
    else:
        st.warning("FastAPI backend is not reachable yet.")
        st.caption("Start the API server first, then submit a case.")

    st.markdown('</div>', unsafe_allow_html=True)


def render_result_panel(
    result: dict[str, Any] | None,
    bmi: float,
    pulse_pressure: int,
    bp_status: str,
) -> None:
    """Render the summary pane on the right side."""

    if not result:
        st.markdown(
            """
            <div class="panel fade-in">
                <div class="result-wrap">
                    <div class="result-head">
                        <div>
                            <div class="risk-label">Risk level</div>
                            <div class="risk-value" style="color:#6b7280;">Awaiting score</div>
                        </div>
                        <div class="status-pill">Ready</div>
                    </div>
                    <div class="summary-box">
                        <strong>Assessment pending</strong>
                        <p>Fill in the intake fields and run the assessment to receive the backend prediction.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="panel" style="margin-top:0.85rem;">',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Quick preview</div>', unsafe_allow_html=True)
        render_metric("BMI", f"{bmi:.1f}", classify_bmi(bmi))
        render_metric("Blood pressure", bp_status, "Local preview")
        render_metric("Pulse pressure", str(pulse_pressure), classify_pulse_pressure(pulse_pressure))
        st.markdown("</div>", unsafe_allow_html=True)
        return

    risk_label = str(result.get("risk_label", "Unknown"))
    probability = float(result.get("probability") or 0.0)
    confidence = probability * 100
    high_risk = risk_label == "High Risk"
    needle_angle = int(probability * 180 - 90)

    summary_text = (
        "The backend scored this case in a higher risk range. Review the submitted values carefully before next action."
        if high_risk
        else "The backend scored this case in a lower risk range. Continue routine monitoring and preventive guidance."
    )

    st.markdown(
        f"""
        <div class="panel fade-in">
            <div class="result-wrap">
                <div class="result-head">
                    <div>
                        <div class="risk-label">Risk level</div>
                        <div class="risk-value {'risk-high' if high_risk else 'risk-low'}">{risk_label}</div>
                    </div>
                    <div class="confidence {'confidence-high' if high_risk else 'confidence-low'}">{confidence:.1f}% confidence</div>
                </div>
                <div class="gauge-grid">
                    <div class="gauge-card">
                        <div class="gauge" style="--needle-angle: {needle_angle}deg;">
                            <div class="gauge-arc"></div>
                            <div class="gauge-inner"></div>
                            <div class="gauge-needle"></div>
                            <div class="gauge-hub"></div>
                            <div class="gauge-score">{confidence:.1f}%</div>
                        </div>
                        <div class="gauge-scale">
                            <span>Low</span>
                            <span>Medium</span>
                            <span>High</span>
                        </div>
                    </div>
                    <div class="summary-box">
                        <strong>Backend summary</strong>
                        <p>{summary_text}</p>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="panel" style="margin-top:0.85rem;">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Clinical preview</div>', unsafe_allow_html=True)
    render_metric("BMI", f"{bmi:.1f}", classify_bmi(bmi))
    render_metric("Blood pressure", bp_status, "Local preview")
    render_metric("Pulse pressure", str(pulse_pressure), classify_pulse_pressure(pulse_pressure))
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    """Run the Streamlit experience."""

    inject_styles()
    render_header()

    st.sidebar.markdown("## Backend")
    api_base_url = st.sidebar.text_input(
        "FastAPI URL",
        value="https://cardiorisk-ml-pipline.onrender.com",
        help="This is the API that handles prediction.",
    )
    render_backend_panel(api_base_url)

    left_col, right_col = st.columns([0.95, 1.05], gap="large")
    submitted_result: dict[str, Any] | None = None

    with left_col:
        with st.container(border=True):
            render_section(
                "Patient intake",
                "Profile details",
                "Collect the user-facing fields before sending the case to the backend.",
            )
            with st.form("patient_form", clear_on_submit=False):
                form_left, form_right = st.columns(2)
                with form_left:
                    age_years = st.number_input("Age", min_value=18, max_value=100, value=52, step=1)
                    gender_label = st.selectbox("Gender", list(GENDER_OPTIONS.keys()), index=1)
                    height_cm = st.number_input("Height (cm)", min_value=120, max_value=220, value=170, step=1)
                with form_right:
                    weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=250, value=82, step=1)
                    ap_hi = st.slider("Systolic blood pressure", 90, 250, 135)
                    ap_lo = st.slider("Diastolic blood pressure", 30, 150, 88)

                st.markdown("---")

                clinical_left, clinical_right = st.columns(2)
                with clinical_left:
                    cholesterol_label = st.selectbox(
                        "Cholesterol",
                        list(LEVEL_OPTIONS.keys()),
                        index=1,
                    )
                    active_label = st.radio("Physical activity", list(ACTIVE_OPTIONS.keys()), horizontal=True, index=0)
                with clinical_right:
                    glucose_label = st.selectbox(
                        "Glucose",
                        list(LEVEL_OPTIONS.keys()),
                        index=0,
                    )

                predict_clicked = st.form_submit_button("Run Risk Assessment")

    bmi = calculate_bmi(height_cm, weight_kg)
    pulse_pressure = calculate_pulse_pressure(ap_hi, ap_lo)
    bp_status = classify_bp(ap_hi, ap_lo)

    with right_col:
        with st.container(border=True):
            render_section(
                "Assessment",
                "Prediction result",
                "The result below comes from the FastAPI backend.",
            )
            render_result_panel(submitted_result, bmi, pulse_pressure, bp_status)

    if predict_clicked:
        payload = build_client_payload(
            age_years=age_years,
            gender_label=gender_label,
            height_cm=height_cm,
            weight_kg=weight_kg,
            ap_hi=ap_hi,
            ap_lo=ap_lo,
            cholesterol_label=cholesterol_label,
            glucose_label=glucose_label,
            active_label=active_label,
        )
        try:
            with st.spinner("Running risk assessment..."):
                submitted_result = send_prediction(api_base_url, payload)
                time.sleep(0.25)

            st.session_state["latest_result"] = submitted_result
            st.session_state["latest_payload"] = payload
            st.success("Assessment completed")
            st.rerun()

        except RuntimeError:
            st.error("Could not reach the backend. Make sure the FastAPI server is running.")
        except Exception as exc:
            # Keep user-facing message friendly; print details to console for debugging
            st.error("Unexpected error during assessment — please try again later.")
            print("Assessment error:", exc)

    submitted_result = st.session_state.get("latest_result") if "latest_result" in st.session_state else None

    review_col, trace_col = st.columns([0.95, 1.05], gap="large")

    with review_col:
        with st.container(border=True):
            render_section(
                "Case review",
                "Input factor profile",
                "A compact visual summary of the submitted values.",
            )
            factors = make_factor_profile(
                age_years=age_years,
                ap_hi=ap_hi,
                ap_lo=ap_lo,
                bmi=bmi,
                cholesterol_label=cholesterol_label,
                glucose_label=glucose_label,
                active_label=active_label,
            )
            render_factor_rows(factors)

    with trace_col:
        with st.container(border=True):
            render_section(
                "Trace",
                "Backend payload",
                "Shows the response returned by the API after assessment.",
            )
            render_trace(submitted_result)
            st.markdown(
                """
                <div class="notice fade-in">
                    This app is a screening demo. It supports review and triage,
                    but it is not a diagnosis tool and does not replace clinical judgement.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Footer / credits
    render_footer()


def render_footer() -> None:
    """Render a small footer with credits and a friendly note."""

    st.markdown(
        '<div class="footer">Built with ❤️ — CardioRisk demo for internship/portfolio. Not for clinical use.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
