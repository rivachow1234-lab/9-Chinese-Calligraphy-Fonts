import base64
import io
import os
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import streamlit as st


# --- 0. PAGE CONFIG (Must be first Streamlit command) ---
st.set_page_config(
    page_title="Personality Calligraphy Matcher",
    page_icon="🖌️",
    layout="centered",
)


# --- HELPER: BACKGROUND, CUSTOM STYLING & ANIMATIONS ---
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
        /* Force light color scheme on mobile browsers */
        :root {{
            color-scheme: light !important;
        }}

        /* Global font: Times New Roman, dark text */
        html, body, [class*="css"], .stApp, .stMarkdown, .stText,
        p, h1, h2, h3, h4, h5, h6, label, .stSlider label,
        .stMetric, .stSuccess, .stInfo, .stExpander {{
            font-family: "Times New Roman", Times, serif !important;
            color: #1a1a1a !important;
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
            background-color: rgba(255, 253, 249, 0.92) !important;
            padding: 2.5rem 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
            margin-top: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(180, 175, 165, 0.4);
        }}

        /* --- FIX FOR MOBILE DARK MODE ON RESULT CARDS --- */
        /* Assigned Cluster ID Metric Box */
        div[data-testid="stMetric"] {{
            background-color: #f7f3ea !important;
            border: 1px solid #d4cbb8 !important;
            border-radius: 12px !important;
            padding: 0.9rem 1.1rem !important;
        }}
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricValue"] div {{
            color: #1a1a1a !important;
        }}

        /* Recommended Style Alert Box (st.success / st.info) */
        div[data-testid="stAlert"] {{
            background-color: #eef5ec !important;
            border: 1px solid #bfd4ba !important;
            border-radius: 12px !important;
            color: #1a1a1a !important;
        }}
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] svg {{
            color: #1a1a1a !important;
            fill: #1a1a1a !important;
        }}

        /* K-Means Distance Breakdown Expander */
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpanderDetails"] {{
            background-color: #faf7f0 !important;
            border-color: #d4cbb8 !important;
            color: #1a1a1a !important;
            border-radius: 10px !important;
        }}
        div[data-testid="stExpander"] summary svg {{
            fill: #1a1a1a !important;
        }}

        /* Light-colored submit button */
        div.stButton > button[kind="primary"],
        div.stButton > button,
        button[data-testid="baseButton-primary"],
        .stFormSubmitButton > button {{
            background-color: #f5f0e6 !important;
            color: #1a1a1a !important;
            border: 1px solid #d4cbb8 !important;
            border-radius: 10px !important;
            font-family: "Times New Roman", Times, serif !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.4rem !important;
            transition: background-color 0.2s ease, box-shadow 0.2s ease;
        }}
        div.stButton > button:hover,
        .stFormSubmitButton > button:hover {{
            background-color: #ebe4d6 !important;
            border-color: #c4b9a4 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            color: #1a1a1a !important;
        }}

        /* Slider track and thumb tweaks */
        div[data-testid="stSlider"] {{
            padding-bottom: 0.8rem;
        }}

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
# Keys 0..8 correspond to KMeans internal cluster indices and asset/font_0..8 filenames.
# Display cluster IDs are shifted to 1..9 in the UI.
FONT_MAPPINGS = {
    0: {
        "name": "Small Seal Script (小篆)",
        "desc": (
            "Rooted in the Qin Dynasty unification of Chinese writing, Small Seal Script is celebrated for its "
            "elongated proportions, symmetrical balance, and deliberate, unhurried brushwork. "
            "You are a thoughtful traditionalist who values heritage, precision, and enduring principles. "
            "Rather than chasing fleeting trends, you find deep satisfaction in careful craftsmanship, ritual, "
            "and building things meant to stand the test of time."
        ),
        "traits": "High Structure, High History",
        "image": "asset/font_0_seal.png",
    },
    1: {
        "name": "Clerical Script (隸書)",
        "desc": (
            "Born during the Han Dynasty to bring clarity and efficiency to official records, Clerical Script "
            "features broad, grounded horizontal strokes and a calm, rhythmic cadence. "
            "You possess a pragmatic, steady, and grounded temperament. People rely on you for your fairness, "
            "clarity of thought, and ability to bring order to complex situations without unnecessary drama."
        ),
        "traits": "High Structure, Moderate Tempo",
        "image": "asset/font_1_clerical.png",
    },
    2: {
        "name": "Yan Style Standard Script (顏體楷書)",
        "desc": (
            "Created by Tang Dynasty master Yan Zhenqing, this iconic script is muscular, upright, and monumental, "
            "with thick vertical pillars and firm, unwavering structure. "
            "You are a principled leader with strong moral conviction and quiet authority. You hold yourself "
            "and others to high standards, approaching responsibilities with integrity, resilience, and rock-solid dependability."
        ),
        "traits": "High Structure, Low Impulsivity",
        "image": "asset/font_2_yan.png",
    },
    3: {
        "name": "Liu Style Standard Script (柳體楷書)",
        "desc": (
            "Pioneered by Liu Gongquan, Liu Style is famed for its 'bony' strength—crisp, clean, well-proportioned "
            "strokes that balance discipline with agility. "
            "You are a composed, versatile, and well-rounded individual who thrives in both structured and collaborative settings. "
            "You value clarity, fairness, and poise, maintaining calm focus even when navigating competing priorities."
        ),
        "traits": "Balanced Structure & Flow",
        "image": "asset/font_3_liu.png",
    },
    4: {
        "name": "Smooth Running Script (淡雅行書)",
        "desc": (
            "Bridging the legibility of Standard Script with the grace of Cursive, Smooth Running Script flows naturally "
            "from stroke to stroke like water gliding over smooth stones. "
            "You are an adaptable diplomat who values harmony, grace, and practical ease. You read rooms effortlessly, "
            "adjust smoothly to shifting circumstances, and bring warmth and cohesion wherever you go."
        ),
        "traits": "Moderate Flow, High Adaptability",
        "image": "asset/font_4_smooth_running.png",
    },
    5: {
        "name": "Expressive Running Script (狂行書)",
        "desc": (
            "Dynamic, sweeping, and alive with momentum, Expressive Running Script captures thought in rapid motion, "
            "linking characters with energetic, spontaneous turns of the brush. "
            "You are a spirited creative who thrives on inspiration, novelty, and fast-paced exploration. "
            "Rules feel like starting points rather than limits to you, and your enthusiasm naturally energizes those around you."
        ),
        "traits": "High Flow, High Expression",
        "image": "asset/font_5_expressive_running.png",
    },
    6: {
        "name": "Cursive Script (草書)",
        "desc": (
            "Unbound by strict stroke-by-stroke conventions, Cursive Script transforms characters into pure rhythm, "
            "velocity, and emotional release—often written in a single breathless surge of ink. "
            "You are a passionate, intuitive visionary driven by raw emotion and creative instinct. "
            "You live intensely in the moment, value authenticity above conformity, and express your inner world boldly."
        ),
        "traits": "High Energy, High Extroversion",
        "image": "asset/font_6_cursive.png",
    },
    7: {
        "name": "Slender Gold Calligraphy (瘦金體書法)",
        "desc": (
            "Invented by Emperor Huizong of Song, Slender Gold is unmistakable for its razor-sharp, metallic, "
            "elongated strokes that resemble fine gold wire and bamboo leaves. "
            "You are an aesthete and modern minimalist with a keen eye for elegance, nuance, and clean design. "
            "Independent and discerning, you prefer refined simplicity over clutter and appreciate subtle beauty others might miss."
        ),
        "traits": "Low Complexity, High Flow",
        "image": "asset/font_7_slender.png",
    },
    8: {
        "name": "Heavy Splash Ink Style (重墨書法)",
        "desc": (
            "Characterized by saturated, heavyweight brushloads, bold contrasts, and unapologetic presence on the paper, "
            "Heavy Splash Ink turns every character into a dramatic visual statement. "
            "You are a fearless maverick with a commanding, unforgettable presence. Unafraid to stand apart from the crowd, "
            "you tackle challenges head-on with raw confidence, originality, and decisive impact."
        ),
        "traits": "High Boldness, Unconventional",
        "image": "asset/font_8_splash_ink.png",
    },
}


# --- 2. INITIALIZE MODEL WITH SYNTHETIC DATA ---
@st.cache_resource
def train_kmeans_model():
    """Generates synthetic training data representing 9 archetypes
    and fits a KMeans model with 9 clusters.
    """
    np.random.seed(42)
    # 5 dimensions: [Structure, Tempo, Expression, Complexity, Adaptability]
    centers = np.array(
        [
            [4.5, 1.5, 2.0, 4.8, 2.0],  # 0: Seal
            [4.0, 2.5, 2.5, 3.5, 2.5],  # 1: Clerical
            [4.5, 2.0, 3.0, 4.0, 2.0],  # 2: Yan Standard
            [3.0, 3.0, 3.0, 3.0, 3.0],  # 3: Liu Standard
            [2.5, 3.5, 3.5, 2.5, 4.0],  # 4: Smooth Running
            [2.0, 4.0, 4.0, 2.0, 4.5],  # 5: Expressive Running
            [1.5, 4.8, 4.8, 1.5, 4.5],  # 6: Cursive
            [2.0, 3.0, 2.0, 1.2, 3.5],  # 7: Slender Gold
            [3.5, 4.5, 4.5, 3.0, 3.5],  # 8: Heavy Splash Ink
        ]
    )

    data = []
    for center in centers:
        cluster_points = np.random.normal(loc=center, scale=0.4, size=(30, 5))
        data.append(cluster_points)

    X = np.vstack(data)
    X = np.clip(X, 1.0, 5.0)

    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans


model = train_kmeans_model()


# --- 3. SLIDER LIKERT SCALE HELPER ---
def likert_slider(question_text, key):
    """Renders a 1–5 Likert slider with labeled endpoints."""
    st.markdown(f"**{question_text}**")
    value = st.slider(
        label=question_text,
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=1.0,
        format="%d",
        label_visibility="collapsed",
        key=key,
        help="1 = Strongly Disagree · 3 = Neutral · 5 = Strongly Agree",
    )
    st.caption("1 Strongly Disagree · 2 Disagree · 3 Neutral · 4 Agree · 5 Strongly Agree")
    return float(value)


# --- 4. NAME CALLIGRAPHY CONFIG ---
# Small curated map of common English names -> Chinese. Extend as needed.
NAME_MAP = {
    "david": "大卫",
    "mary": "玛丽",
    "john": "约翰",
    "emily": "艾米丽",
    "michael": "迈克尔",
    "sophia": "索菲亚",
    "james": "詹姆斯",
    "olivia": "奥利维亚",
    "daniel": "丹尼尔",
    "grace": "格蕾丝",
    "anna": "安娜",
    "leo": "里奥",
}

# One TTF/OTF per style. Commit these under fonts/ in your repo.
# Keys MUST match the internal cluster indices 0..8 above.
FONT_FILES = {
    0: "fonts/seal.otf",
    1: "fonts/clerical.ttf",
    2: "fonts/yan.ttf",
    3: "fonts/liu.ttf",
    4: "fonts/smooth_running.ttf",
    5: "fonts/expressive_running.ttf",
    6: "fonts/cursive.ttf",
    7: "fonts/slender_gold.ttf",
    8: "fonts/splash_ink.ttf",
}


def render_name_image(chinese_name: str, font_path: str, font_size: int = 120) -> Image.Image:
    """Render Chinese text onto a clean white canvas with the given font."""
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font not found: {font_path}")

    font = ImageFont.truetype(font_path, font_size)

    # Measure text using textbbox (Pillow 10+ compatible)
    dummy_img = Image.new("RGB", (1, 1), "white")
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), chinese_name, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x, pad_y = 120, 100
    canvas_w = max(text_w + pad_x * 2, 600)
    canvas_h = max(text_h + pad_y * 2, 400)

    img = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(img)

    x = (canvas_w - text_w) / 2 - bbox[0]
    y = (canvas_h - text_h) / 2 - bbox[1]
    draw.text((x, y), chinese_name, font=font, fill="black")
    return img


# --- 5. STREAMLIT UI ---
st.title("🖌️ Chinese Calligraphy Personality Matcher")
st.markdown(
    "Move each slider to indicate how much you agree or disagree with the **10 statements** below. "
    "Our K-Means clustering model will analyze your responses across 5 psychological dimensions to match you with **1 of 9 Chinese Calligraphy Styles**."
)
st.markdown("---")

with st.form("calligraphy_form"):
    st.subheader("Your Personality Assessment")

    # Dimension 1: Structure & Order
    q1 = likert_slider(
        "1. I prefer clear rules, neat organization, and structured plans over creative chaos.",
        "q1",
    )
    q2 = likert_slider(
        "2. I feel most comfortable when my daily schedule and workspace are well-ordered.",
        "q2",
    )

    # Dimension 2: Tempo & Energy
    q3 = likert_slider(
        "3. I move, think, and make decisions quickly, often acting on instinct.",
        "q3",
    )
    q4 = likert_slider(
        "4. I thrive in fast-paced environments rather than taking a slow, methodical approach.",
        "q4",
    )

    # Dimension 3: Expression & Emotion
    q5 = likert_slider(
        "5. I express my emotions outwardly and passionately rather than keeping them bottled up.",
        "q5",
    )
    q6 = likert_slider(
        "6. People around me can easily tell how I am feeling from my energy and tone.",
        "q6",
    )

    # Dimension 4: Complexity vs. Minimalism
    q7 = likert_slider(
        "7. I am deeply fascinated by history, rich tradition, and intricate details.",
        "q7",
    )
    q8 = likert_slider(
        "8. I appreciate layered, ornate craftsmanship more than sleek, bare minimalism.",
        "q8",
    )

    # Dimension 5: Adaptability & Flow
    q9 = likert_slider(
        "9. I adapt smoothly to sudden changes rather than sticking rigidly to my original plan.",
        "q9",
    )
    q10 = likert_slider(
        "10. I enjoy improvising in the moment when unexpected opportunities arise.",
        "q10",
    )

    submitted = st.form_submit_button("Find My Calligraphy Font!")


# --- 6. PREDICTION (store in session_state so it survives reruns) ---
if submitted:
    dim_structure = (q1 + q2) / 2.0
    dim_tempo = (q3 + q4) / 2.0
    dim_expression = (q5 + q6) / 2.0
    dim_complexity = (q7 + q8) / 2.0
    dim_adaptability = (q9 + q10) / 2.0

    user_vector = np.array(
        [[dim_structure, dim_tempo, dim_expression, dim_complexity, dim_adaptability]]
    )

    cluster_idx = int(model.predict(user_vector)[0])
    font_info = FONT_MAPPINGS[cluster_idx]

    # Persist everything the result & name-generator sections need
    st.session_state.result_ready = True
    st.session_state.cluster_idx = cluster_idx
    st.session_state.font_info = font_info
    st.session_state.user_vector = user_vector
    st.session_state.hearts_triggered = False  # allow hearts to fire again
    # Clear any previously rendered name image
    st.session_state.name_image_bytes = None
    st.session_state.name_rendered_for = None


# --- 7. RESULT DISPLAY (persisted) ---
if st.session_state.get("result_ready"):
    cluster_idx = st.session_state.cluster_idx
    font_info = st.session_state.font_info
    user_vector = st.session_state.user_vector
    display_cluster_num = cluster_idx + 1

    # Fire the heart animation once per submission
    if not st.session_state.get("hearts_triggered"):
        trigger_hearts()
        st.session_state.hearts_triggered = True

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
            st.image(
                font_info["image"],
                caption=f"Preview: {font_info['name']}",
                use_container_width=True,
            )
        else:
            st.info(f"Preview image placeholder: `{font_info['image']}`")

    with st.expander("View K-Means Distance Breakdown"):
        distances = model.transform(user_vector)[0]
        dist_df = pd.DataFrame(
            {
                "Cluster": [k + 1 for k in FONT_MAPPINGS.keys()],
                "Font Style": [f["name"] for f in FONT_MAPPINGS.values()],
                "Distance Score (Lower is Closer)": distances,
            }
        ).sort_values("Distance Score (Lower is Closer)")
        st.dataframe(dist_df, hide_index=True)

    # --- 8. NAME CALLIGRAPHY GENERATOR ---
    st.markdown("---")
    st.subheader("🖋️ Generate Your Name in Calligraphy")
    st.caption(
        "Enter your name below. If you only give an English name, we'll try a small built-in mapping — "
        "for the best result please also enter your name in Chinese characters."
    )

    if "name_image_bytes" not in st.session_state:
        st.session_state.name_image_bytes = None
    if "name_rendered_for" not in st.session_state:
        st.session_state.name_rendered_for = None

    with st.form("name_calligraphy_form"):
        col_name_1, col_name_2 = st.columns(2)
        with col_name_1:
            english_name = st.text_input(
                "English Name (optional)",
                placeholder="e.g., David",
            )
        with col_name_2:
            chinese_name_input = st.text_input(
                "Chinese Name (recommended)",
                placeholder="e.g., 王小明",
            )
        generate_btn = st.form_submit_button("✍️ Render My Name")

    if generate_btn:
        final_cn_name = chinese_name_input.strip()

        if not final_cn_name and english_name.strip():
            key = english_name.strip().lower()
            final_cn_name = NAME_MAP.get(key, "")
            if not final_cn_name:
                st.warning(
                    "We don't have an automatic transliteration for that English name. "
                    "Please enter a Chinese name in the second field."
                )

        if final_cn_name:
            # Require at least one CJK character
            if not re.search(r"[\u4e00-\u9fff]", final_cn_name):
                st.warning("Please enter a name that contains Chinese characters.")
            else:
                font_path = FONT_FILES.get(cluster_idx)
                if font_path and os.path.exists(font_path):
                    try:
                        img = render_name_image(final_cn_name, font_path)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        st.session_state.name_image_bytes = buf.getvalue()
                        st.session_state.name_rendered_for = final_cn_name
                    except Exception as e:
                        st.error(f"Could not render the name: {e}")
                else:
                    st.error(
                        f"The font file for this style is missing: `{font_path}`. "
                        "Please check your `fonts/` folder and the `FONT_FILES` mapping."
                    )

    if st.session_state.name_image_bytes:
        st.success(
            f"Rendered name: **{st.session_state.name_rendered_for}** "
            f"in *{font_info['name']}*"
        )
        st.image(
            st.session_state.name_image_bytes,
            caption=f"Your name rendered in {font_info['name']}",
            use_container_width=True,
        )
        st.download_button(
            label="⬇️ Download Calligraphy Image (PNG)",
            data=st.session_state.name_image_bytes,
            file_name=f"calligraphy_{st.session_state.name_rendered_for}.png",
            mime="image/png",
        )
