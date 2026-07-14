"""
utils.py
================================================================
Shared helper functions for the Streamlit app: caching wrappers,
CSS injection, dataset loading, and small formatting helpers.
================================================================
"""

import json
import os

import pandas as pd
import streamlit as st

from predict import load_artifacts
from preprocessing import load_and_clean

ARTIFACTS_DIR = "artifacts"
DATA_PATH = "ai_student_impact_dataset.csv"


@st.cache_resource(show_spinner=False)
def get_model_and_encoders():
    """Cached loader for the trained model + encoders (loaded once per session)."""
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    """Cached loader for the cleaned dataset (used for EDA / visualizations)."""
    return load_and_clean(DATA_PATH)


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    """Cached loader for the metrics.json produced by train_model.py."""
    path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_feature_importance() -> dict:
    path = os.path.join(ARTIFACTS_DIR, "feature_importance.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def artifact_image_path(filename: str) -> str:
    return os.path.join(ARTIFACTS_DIR, filename)


def inject_custom_css():
    """Injects the global glassmorphism / gradient theme CSS used across all pages."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        :root {
            --bg-gradient-start: #0f0c29;
            --bg-gradient-mid: #302b63;
            --bg-gradient-end: #24243e;
            --accent-1: #7F5AF0;
            --accent-2: #2CB1BC;
            --accent-3: #FF6EC7;
            --card-bg: rgba(255, 255, 255, 0.06);
            --card-border: rgba(255, 255, 255, 0.12);
            --text-light: #F1F5F9;
            --text-muted: #A1A1AA;
        }

        .stApp {
            background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-mid) 50%, var(--bg-gradient-end) 100%);
            background-attachment: fixed;
        }

        h1, h2, h3, h4 {
            font-family: 'Poppins', sans-serif !important;
            color: var(--text-light) !important;
            font-weight: 700 !important;
        }

        p, span, label, li, div {
            color: var(--text-light);
        }

        /* Glass card */
        .glass-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.28);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            margin-bottom: 1.2rem;
        }
        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 14px 40px rgba(127, 90, 240, 0.28);
        }

        /* Hero banner */
        .hero-banner {
            background: linear-gradient(120deg, var(--accent-1), var(--accent-2));
            border-radius: 24px;
            padding: 2.6rem 2.2rem;
            text-align: center;
            box-shadow: 0 12px 40px rgba(127, 90, 240, 0.35);
            margin-bottom: 1.6rem;
            animation: fadeIn 1s ease;
        }
        .hero-banner h1 {
            color: white !important;
            font-size: 2.4rem !important;
            margin-bottom: 0.4rem !important;
        }
        .hero-banner p {
            color: rgba(255,255,255,0.92) !important;
            font-size: 1.05rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Badge pills */
        .badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.18);
            color: var(--text-light);
            font-size: 0.82rem;
            font-weight: 500;
            margin: 0.2rem 0.3rem 0.2rem 0;
        }

        /* Risk result banners */
        .risk-banner {
            border-radius: 18px;
            padding: 1.6rem;
            text-align: center;
            font-weight: 700;
            font-size: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 28px rgba(0,0,0,0.3);
        }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(120deg, var(--accent-1), var(--accent-3));
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.6rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 22px rgba(127, 90, 240, 0.45);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        /* Footer */
        .app-footer {
            text-align: center;
            padding: 1.4rem 0 0.6rem 0;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid rgba(255,255,255,0.08);
            margin-top: 2.4rem;
        }
        .app-footer a {
            color: var(--accent-2);
            text-decoration: none;
            margin: 0 0.4rem;
            font-weight: 600;
        }

        /* DataFrame / table tweaks */
        [data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            Built with ❤️ by <b>Sandeep</b> &nbsp;|&nbsp;
            <a href="https://github.com/Sahilberwer" target="_blank">GitHub</a>
            <a href="https://linkedin.com/in/sandeep-berwer" target="_blank">LinkedIn</a>
            &nbsp;|&nbsp; Honeywell AI Internship Project — 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge_row(items):
    html = "".join(f'<span class="badge">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)
