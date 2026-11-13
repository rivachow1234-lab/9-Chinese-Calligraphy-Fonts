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

# --- HELPER: BACKGROUND, 16PERSONALITIES BUBBLES & CUSTOM ANIMATIONS ---
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
        /* Ink Wash Mountains Background */
        .stApp {{
            background-image: linear-gradient(rgba(245, 243, 238, 0.78), rgba(245, 243, 238, 0.85)), {bg_css};
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Frosted Rice-Paper Container */
        .block-container {{
            background-color: rgba(255, 253, 249, 0.90);
            padding: 2.5rem 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
            margin-top: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(180, 175, 165, 0.4);
            color: #2C2A29;
            overflow-x: hidden;
        }}

        /* Force dark text color */
        h1, h2, h3, h4, h5, h6, p, span, label, div[data-testid="stMarkdownContainer"] {{
            color: #2C2A29 !important;
        }}

        /* Scale Row Container */
        div[class*="st-key-scale_row_"] {{
            background-color: rgba(242, 239, 233, 0.7) !important;
            padding: 8px 12px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(180, 175, 165, 0.35) !important;
            margin-bottom: 16px !important;
            width: 100% !important;
        }}

        .scale-text-agree {{
            font-weight: 600; color: #2C2A29; font-size: 0.95rem;
            text-align: right; padding-top: 6px; user-select: none;
        }}

        .scale-text-disagree {{
            font-weight: 600; color: #2C2A29; font-size: 0.95rem;
            text-align: left; padding-top: 6px; user-select: none;
        }}

        /* Radio Group Alignment */
        div[class*="st-key-scale_row_"] div[role="radiogroup"] {{
            display: flex !important;
            flex-direction: row !important;
            justify-content: center !important;
            align-items: center !important;
            background-color: transparent !important;
            padding: 0 !important;
            border: none !important;
            margin: 0 !important;
            gap: 14px !important;
        }}
        div[class*="st-key-scale_row_"] div[role="radiogroup"] > label {{
            cursor: pointer; display: flex !important;
            align-items: center !important; justify-content: center !important;
            transition: transform 0.15s ease;
            background-color: transparent !important; border: none !important;
            padding: 0 !important; margin: 0 !important;
        }}
        div[class*="st-key-scale_row_"] div[role="radiogroup"] > label:hover {{
            transform: scale(1.15);
        }}

        /* Varying bubble sizes */
        div[class*="st-key-scale_row_"] div[role="radiogroup"] > label:nth-child(1) div[role="radio"] > div:first-child,
        div[class*="st-key-scale_row_"] div[role="radiogroup"] > label:nth-child(7) div[role="radio"] > div:first-child {{
            transform: scale(1.6) !important;
        }}
        div[class*="st-key-scale_row_"] div[role="radiogroup"] > label:nth-child(2) div[role="radio"] > div:first-child,
        div[class*="st-key-scale_row_"] div[role="radiogroup"] > label:nth-child(6) div[role="radio"] > div:first-child {{
            transform: scale(1.3) !important;
        }}
        div[class*="st-key-scale_row_"] div[role="radiogroup"] > label:nth-child(3) div[role="radio"] > div:first-child,
        div[class*="st-key-scale_row_"] div[role="radiogroup"] > label:nth-child(5) div[role="radio"] > div:first-child {{
            transform: scale(1.0) !important;
        }}
        div[class*="st-key-scale_row_"] div[role="radiogroup"] > label:nth-child(4) div[role="radio"] > div:first-child {{
            transform: scale(0.75) !important;
        }}

        /* Submit Button Styling */
        div[data-testid="stFormSubmitButton"] button {{
            background-color: #EBE7DE !important;
            color: #2C2A29 !important;
            border: 1px solid rgba(180, 175, 165, 0.6) !important;
            font-size: 0.85rem !important;
            padding: 0.5rem 1rem !important;
            width: 100% !important;
            border-radius: 8px !important;
            transition: background-color 0.2s ease !important;
        }}
        div[data-testid="stFormSubmitButton"] button:hover {{
            background-color: #DCD6CC !important;
            border-color: rgba(180, 175, 165, 0.9) !important;
        }}
        div[data-testid="stFormSubmitButton"] button p {{
            font-size: 0.85rem !important;
            font-weight: 600 !important;
        }}

        /* --- MOBILE FIXES --- */
        @media (max-width: 640px) {{
            div[class*="st-key-scale_row_"] div[data-testid="stHorizontalBlock"] {{
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 0px !important;
                width: 100% !important;
            }}
            div[class*="st-key-scale_row_"] div[data-testid="column"] {{
                padding: 0 !important;
                margin: 0 !important;
                width: auto !important;
                min-width: 0 !important;
            }}
            div[class*="st-key-scale_row_"] div[data-testid="column"]:nth-of-type(2) {{
                flex: 1 1 auto !important;
            }}
            .scale-text-agree, .scale-text-disagree {{
                font-size: 0.75rem !important;
                padding-top: 8px !important;
                white-space: nowrap !important;
            }}
            div[class*="st-key-scale_row_"] div[role="radiogroup"] {{
                gap: 4px !important; 
                justify-content: space-evenly !important;
            }}
            div[class*="st-key-scale_row_"] div[role="radio"] > div:first-child {{
                transform: scale(0.85) !important;
            }}
        }}

        /* Falling Hearts Animation */
        @keyframes fall {{
            0% {{ transform: translateY(-10vh) rotate(0deg); opacity: 1; }}
            100% {{ transform: translateY(105vh) rotate(360deg); opacity: 0; }}
        }}
        .heart-particle {{
            position: fixed; top: -5vh; font-size: 26px; user-select: none;
            pointer-events: none; z-index: 9999; animation: fall linear forwards;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def trigger_hearts():
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
    0: {"name": "Small Seal Script (小篆)", "desc": "Rooted in the Qin Dynasty unification of Chinese writing, Small Seal Script is celebrated for its elongated proportions, symmetrical balance, and deliberate, unhurried brushwork. You are a thoughtful traditionalist who values heritage, precision, and enduring principles.", "traits": "High Structure, High History", "image": "asset/font_0_seal.png"},
    1: {"name": "Clerical Script (隸書)", "desc": "Born during the Han Dynasty to bring clarity and efficiency to official records, Clerical Script features broad, grounded horizontal strokes and a calm, rhythmic cadence. You possess a pragmatic, steady, and grounded temperament.", "traits": "High Structure, Moderate Tempo", "image": "asset/font_1_clerical.png"},
    2: {"name": "Yan Style Standard Script (顏體楷書)", "desc": "Created by Tang Dynasty master Yan Zhenqing, this iconic script is muscular, upright, and monumental. You are a principled leader with strong moral conviction and quiet authority.", "traits": "High Structure, Low Impulsivity", "image": "asset/font_2_yan.png"},
    3: {"name": "Liu Style Standard Script (柳體楷書)", "desc": "Pioneered by Liu Gongquan, Liu Style is famed for its 'bony' strength. You are a composed, versatile, and well-rounded individual who thrives in both structured and collaborative settings.", "traits": "Balanced Structure & Flow", "image": "asset/font_3_liu.png"},
    4: {"name": "Smooth Running Script (淡雅行書)", "desc": "Bridging the legibility of Standard Script with the grace of Cursive, Smooth Running Script flows naturally. You are an adaptable diplomat who values harmony, grace, and practical ease.", "traits": "Moderate Flow, High Adaptability", "image": "asset/font_4_smooth_running.png"},
    5: {"name": "Expressive Running Script (狂行書)", "desc": "Dynamic, sweeping, and alive with momentum, Expressive Running Script captures thought in rapid motion. You are a spirited creative who thrives on inspiration, novelty, and fast-paced exploration.", "traits": "High Flow, High Expression", "image": "asset/font_5_expressive_running.png"},
    6: {"name": "Cursive Script (草書)", "desc": "Unbound by strict stroke-by-stroke conventions, Cursive Script transforms characters into pure rhythm, velocity, and emotional release. You are a passionate, intuitive visionary driven by raw emotion.", "traits": "High Energy, High Extroversion", "image": "asset/font_6_cursive.png"},
    7: {"name": "Slender Gold Calligraphy (瘦金體書法)", "desc": "Invented by Emperor Huizong of Song, Slender Gold is unmistakable for its razor-sharp, metallic, elongated strokes. You are an aesthete and modern minimalist with a keen eye for elegance.", "traits": "Low Complexity, High Flow", "image": "asset/font_7_slender.png"},
    8: {"name": "Heavy Splash Ink Style (重墨書法)", "desc": "Characterized by saturated, heavyweight brushloads and bold contrasts. You are a fearless maverick with a commanding, unforgettable presence.", "traits": "High Boldness, Unconventional", "image": "asset/font_8_splash_ink.png"},
}


# --- 2. INITIALIZE MODEL WITH SYNTHETIC DATA ---
@st.cache_resource
def train_kmeans_model():
    np.random.seed(42)
    centers = np.array([
        [4.5, 1.5, 2.0, 4.8, 2.0], [4.0, 2.5, 2.5, 3.5, 2.5], [4.5, 2.0, 3.0, 4.0, 2.0],
        [3.0, 3.0, 3.0, 3.0, 3.0], [2.5, 3.5, 3.5, 2.5, 4.0], [2.0, 4.0, 4.0, 2.0, 4.5],
        [1.5, 4.8, 4.8, 1.5, 4.5], [2.0, 3.0, 2.0, 1.2, 3.5], [3.5, 4.5, 4.5, 3.0, 3.5],
    ])
    data = []
    for center in centers:
        data.append(np.random.normal(loc=center, scale=0.4, size=(30, 5)))
    X = np.vstack(data)
    X = np.clip(X, 1.0, 5.0)
    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans

model = train_kmeans_model()

# --- 3. 16PERSONALITIES SCALE MAPPING ---
LIKERT_MAPPING = {1: 5.0, 2: 4.33, 3: 3.67, 4: 3.0, 5: 2.33, 6: 1.67, 7: 1.0}


def mbti_bubbles(question_text, key_name):
    st.markdown(f"**{question_text}**")
    
    # Static container key keeps your CSS styling intact without caching widget state
    with st.container(border=True, key=f"scale_row_{key_name}"):
        col_l, col_m, col_r = st.columns([0.8, 5, 0.8])
        with col_l:
            st.markdown('<div class="scale-text-agree">Agree</div>', unsafe_allow_html=True)
        with col_m:
            choice = st.radio(
                label=question_text,
                options=[1, 2, 3, 4, 5, 6, 7],
                index=3,  # <--- Forces default to index 3 (option 4, middle neutral)
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda x: "",
                # NO 'key' parameter here, preventing Streamlit from locking/storing old states
            )
        with col_r:
            st.markdown('<div class="scale-text-disagree">Disagree</div>', unsafe_allow_html=True)
    
    return LIKERT_MAPPING[choice]


# --- 4. STREAMLIT UI ---
st.title("🖌️ Chinese Calligraphy Personality Matcher")
st.markdown(
    "Select the bubble that best reflects how much you agree or disagree with each of the **10 statements** below. "
    "Our K-Means clustering model will analyze your responses across 5 psychological dimensions to match you with **1 of 9 Chinese Calligraphy Styles**."
)

# Reset Button placement right above the form
col_title_space, col_reset_btn = st.columns([4, 1])
with col_reset_btn:
    if st.button("🔄 Reset Form", use_container_width=True):
        st.rerun()

st.markdown("---")

with st.form("calligraphy_form"):
    st.subheader("Your Personality Assessment")

    q1 = mbti_bubbles("1. I prefer clear rules, neat organization, and structured plans over creative chaos.", "q1")
    q2 = mbti_bubbles("2. I feel most comfortable when my daily schedule and workspace are well-ordered.", "q2")

    q3 = mbti_bubbles("3. I move, think, and make decisions quickly, often acting on instinct.", "q3")
    q4 = mbti_bubbles("4. I thrive in fast-paced environments rather than taking a slow, methodical approach.", "q4")

    q5 = mbti_bubbles("5. I express my emotions outwardly and passionately rather than keeping them bottled up.", "q5")
    q6 = mbti_bubbles("6. People around me can easily tell how I am feeling from my energy and tone.", "q6")

    q7 = mbti_bubbles("7. I am deeply fascinated by history, rich tradition, and intricate details.", "q7")
    q8 = mbti_bubbles("8. I appreciate layered, ornate craftsmanship more than sleek, bare minimalism.", "q8")

    q9 = mbti_bubbles("9. I adapt smoothly to sudden changes rather than sticking rigidly to my original plan.", "q9")
    q10 = mbti_bubbles("10. I enjoy improvising in the moment when unexpected opportunities arise.", "q10")

    submitted = st.form_submit_button("Find My Calligraphy Font!")

# --- 5. PREDICTION & RESULT DISPLAY ---
if submitted:
    dim_structure = (q1 + q2) / 2.0
    dim_tempo = (q3 + q4) / 2.0
    dim_expression = (q5 + q6) / 2.0
    dim_complexity = (q7 + q8) / 2.0
    dim_adaptability = (q9 + q10) / 2.0

    user_vector = np.array([[dim_structure, dim_tempo, dim_expression, dim_complexity, dim_adaptability]])

    cluster_idx = int(model.predict(user_vector)[0])
    display_cluster_num = cluster_idx + 1
    font_info = FONT_MAPPINGS[cluster_idx]

    trigger_hearts()

    st.markdown("---")
    st.subheader("✨ Your Calligraphy Match Result")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Assigned Cluster ID", value=f"Cluster #{display_cluster_num}")
    with col2:
        st.success(f"**Recommended Style:** {font_info['name']}")

    st.markdown(f"### **Core Profile Vibe:** _{font_info['traits']}_")
    st.write(font_info["desc"])

    with col2:
        if os.path.exists(font_info["image"]):
            st.image(font_info["image"], caption=f"Preview: {font_info['name']}", use_container_width=True)
        else:
            st.info(f"Preview image placeholder: `{font_info['image']}`")

    with st.expander("View K-Means Distance Breakdown"):
        distances = model.transform(user_vector)[0]
        dist_df = pd.DataFrame({
            "Cluster": [k + 1 for k in FONT_MAPPINGS.keys()],
            "Font Style": [f["name"] for f in FONT_MAPPINGS.values()],
            "Distance Score (Lower is Closer)": distances,
        }).sort_values("Distance Score (Lower is Closer)")
        st.dataframe(dist_df, hide_index=True)
