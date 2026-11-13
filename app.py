import base64
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st

# --- 0. PAGE CONFIG (Must be first Streamlit command) ---
st.set_page_config(
    page_title="Personality Calligraphy Matcher",
    page_icon="🖌️",
    layout="centered",
)


# --- HELPER: BACKGROUND, TIMES NEW ROMAN, GREEN-PURPLE BUBBLES & ANIMATIONS ---
def apply_custom_styling():
    local_bg_path = "asset/ink_mountains.jpg"
    if os.path.exists(local_bg_path):
        with open(local_bg_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        bg_css = f'url("data:image/jpeg;base64,{encoded}")'
    else:
        bg_css = 'url("https://images.unsplash.com/photo-1518709268805-4e9042af9f23?auto=format&fit=crop&w=1920&q=80")'

    st.markdown(
        f"""
        <style>
        /* Global Font Override to Times New Roman */
        html, body, [class*="st-"] {{
            font-family: 'Times New Roman', Times, serif !important;
        }}

        /* Ink Wash Mountains Background */
        .stApp {{
            background-image: linear-gradient(rgba(245, 243, 238, 0.78), rgba(245, 243, 238, 0.85)), {bg_css};
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Frosted Rice-Paper Container for Readability */
        .block-container {{
            background-color: rgba(255, 253, 249, 0.92);
            padding: 2.5rem 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
            margin-top: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(180, 175, 165, 0.4);
        }}

        /* 7-Point Horizontal Radio Bubbles with Green-to-Purple Gradient */
        div[role="radiogroup"] {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: rgba(242, 239, 233, 0.75);
            padding: 12px 14px;
            border-radius: 12px;
            border: 1px solid rgba(180, 175, 165, 0.35);
            margin-bottom: 16px;
            gap: 4px;
        }}
        div[role="radiogroup"] label {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            cursor: pointer;
            font-size: 0.78rem;
            padding: 4px 4px;
            border-radius: 8px;
            transition: transform 0.15s ease;
        }}
        div[role="radiogroup"] label:hover {{
            transform: scale(1.05);
        }}

        /* Color Gradient mapping for the 7 bubbles (Green -> Teal/Neutral -> Purple) */
        div[role="radiogroup"] label:nth-child(1) p {{ color: #1b4332; font-weight: bold; }}
        div[role="radiogroup"] label:nth-child(2) p {{ color: #2d6a4f; }}
        div[role="radiogroup"] label:nth-child(3) p {{ color: #52b788; }}
        div[role="radiogroup"] label:nth-child(4) p {{ color: #6c757d; }}
        div[role="radiogroup"] label:nth-child(5) p {{ color: #7b2cbf; }}
        div[role="radiogroup"] label:nth-child(6) p {{ color: #560bad; }}
        div[role="radiogroup"] label:nth-child(7) p {{ color: #3c096c; font-weight: bold; }}

        /* Falling Hearts Animation */
        @keyframes fall {{
            0% {{
                transform: translateY(-10vh) rotate(0deg);
                opacity: 1;
            }}
            100% {{
                transform: translateY(105vh) rotate(360deg);
                opacity: 0;
            }}
        }}
        .heart-particle {{
            position: fixed;
            top: -5vh;
            font-size: 26px;
            user-select: none;
            pointer-events: none;
            z-index: 9999;
            animation: fall linear forwards;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def trigger_hearts():
    """Renders floating/falling hearts across the screen."""
    hearts_html = ""
    heart_icons = ["❤️", "🖤", "♥️", "💖"]
    np.random.seed(None)
    for i in range(30):
        left = np.random.randint(2, 98)
        duration = np.random.uniform(2.5, 4.8)
        delay = np.random.uniform(0.0, 1.5)
        size = np.random.randint(20, 34)
        icon = heart_icons[i % len(heart_icons)]
        hearts_html += (
            f'<div class="heart-particle" style="left:{left}%; '
            f"animation-duration:{duration:.2f}s; animation-delay:{delay:.2f}s; "
            f'font-size:{size}px;">{icon}</div>'
        )
    st.markdown(hearts_html, unsafe_allow_html=True)


apply_custom_styling()

# --- 1. CONFIGURATION & FONT MAPPINGS ---
FONT_MAPPINGS = {
    0: {
        "name": "Small Seal Script (小篆)",
        "desc": (
            "Rooted in the Qin Dynasty unification of Chinese writing, Small Seal Script is celebrated for its "
            "elongated proportions, symmetrical balance, and deliberate, unhurried brushwork. "
            "You are a thoughtful traditionalist who values heritage, precision, and enduring principles. "
