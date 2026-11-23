import base64
from io import BytesIO
import os

from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
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

        div[data-testid="stSlider"] {{
            padding-bottom: 0.8rem;
        }}

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

STYLE_RENDER_CONFIG = {
    0: {"font_path": "asset/fonts/seal.ttf", "ink": (35, 32, 28), "accent": "#8b261d"},
    1: {"font_path": "asset/fonts/clerical.ttf", "ink": (30, 28, 25), "accent": "#7a2e22"},
    2: {"font_path": "asset/fonts/yan.ttf", "ink": (22, 22, 22), "accent": "#9e2a2b"},
    3: {"font_path": "asset/fonts/liu.ttf", "ink": (25, 27, 30), "accent": "#2b4c59"},
    4: {"font_path": "asset/fonts/running.ttf", "ink": (34, 34, 34), "accent": "#4a6b5d"},
    5: {"font_path": "asset/fonts/expressive.ttf", "ink": (20, 20, 20), "accent": "#a43820"},
    6: {"font_path": "asset/fonts/cursive.ttf", "ink": (18, 18, 18), "accent": "#8c1d18"},
    7: {"font_path": "asset/fonts/slender_gold.ttf", "ink": (65, 55, 35), "accent": "#9c7a2c"},
    8: {"font_path": "asset/fonts/splash_ink.ttf", "ink": (15, 15, 15), "accent": "#781d19"},
}

FALLBACK_CJK_FONTS = [
    "asset/fonts/NotoSerifTC-Regular.ttf",
    "asset/fonts/NotoSerifSC-Regular.ttf",
]


def load_calligraphy_font(preferred_path: str, size: int) -> ImageFont.ImageFont:
    """Loads the cluster's .ttf/.otf font, falling back to a CJK font or default."""
    candidates = [preferred_path] + FALLBACK_CJK_FONTS
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default(size=max(16, int(size * 0.75)))


def render_name_calligraphy_image(
    name_text: str, cluster_idx: int, style_name: str
) -> Image.Image:
    """Renders a personalized rice-paper calligraphy card using a local TTF/OTF font."""
    width, height = 900, 340
    paper_bg = (250, 246, 236)
    cfg = STYLE_RENDER_CONFIG.get(
        cluster_idx, {"font_path": "", "ink": (25, 25, 25), "accent": "#8b261d"}
    )

    img = Image.new("RGB", (width, height), color=paper_bg)
    draw = ImageDraw.Draw(img)

    # Traditional double border
    draw.rectangle([18, 18, width - 18, height - 18], outline=(195, 184, 162), width=2)
    draw.rectangle([26, 26, width - 26, height - 26], outline=(220, 212, 194), width=1)

    name_font = load_calligraphy_font(cfg["font_path"], size=84)
    sub_font = load_calligraphy_font(cfg["font_path"], size=24)

    bbox = draw.textbbox((0, 0), name_text, font=name_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = max(40, (width - text_w) / 2)
    text_y = (height - text_h) / 2 - 12

    draw.text((text_x, text_y), name_text, fill=cfg["ink"], font=name_font)

    footer_text = f"Style: {style_name}"
    fbbox = draw.textbbox((0, 0), footer_text, font=sub_font)
    fw = fbbox[2] - fbbox[0]
    draw.text(((width - fw) / 2, height - 62), footer_text, fill=(110, 102, 88), font=sub_font)

    # Decorative vermilion seal stamp
    seal_x0, seal_y0, seal_x1, seal_y1 = width - 115, 55, width - 55, 115
    draw.rectangle([seal_x0, seal_y0, seal_x1, seal_y1], outline=cfg["accent"], width=3)
    draw.text((seal_x0 + 14, seal_x0 + 16), "印", fill=cfg["accent"], font=sub_font)

    return img


class ChineseNameResult(BaseModel):
    chinese_name: str
    pinyin: str
    meaning: str


@st.cache_data(show_spinner=False)
def generate_chinese_name_llm(
    user_name: str,
    style_name: str,
    traits: str,
    character_script: str = "Traditional Chinese",
) -> dict | None:
    """Translates/adapts a user's name into a poetic Chinese name via Gemini."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    client = genai.Client(api_key=api_key)
    prompt = (
        f"Adapt the name '{user_name}' into a 2-to-3 character {character_script} name "
        f"that sounds natural, respectful, and evocative. "
        f"Match the aesthetic and vibe of the calligraphy style '{style_name}' "
        f"and personality traits '{traits}'. "
        f"Keep the phonetic feel of '{user_name}' when helpful, or blend phonetics with meaningful characters."
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ChineseNameResult,
            temperature=0.7,
        ),
    )

    if not response.text:
        return None
    return ChineseNameResult.model_validate_json(response.text).model_dump()


# --- 2. INITIALIZE MODEL WITH SYNTHETIC DATA ---
@st.cache_resource
def train_kmeans_model():
    """Generates synthetic training data representing 9 archetypes
    and fits a KMeans model with 9 clusters.
    """
    np.random.seed(42)
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
    st.caption("1 Strongly Disagree  ·  2 Disagree  ·  3 Neutral  ·  4 Agree  ·  5 Strongly Agree")
    return float(value)


# --- 4. STREAMLIT UI ---
st.title("🖌️ Chinese Calligraphy Personality Matcher")
st.markdown(
    "Move each slider to indicate how much you agree or disagree with the **10 statements** below. "
    "Our K-Means clustering model will analyze your responses across 5 psychological dimensions to match you with **1 of 9 Chinese Calligraphy Styles**."
)
st.markdown("---")

with st.form("calligraphy_form"):
    st.subheader("Your Personality Assessment")

    q1 = likert_slider(
        "1. I prefer clear rules, neat organization, and structured plans over creative chaos.",
        "q1",
    )
    q2 = likert_slider(
        "2. I feel most comfortable when my daily schedule and workspace are well-ordered.",
        "q2",
    )
    q3 = likert_slider(
        "3. I move, think, and make decisions quickly, often acting on instinct.",
        "q3",
    )
    q4 = likert_slider(
        "4. I thrive in fast-paced environments rather than taking a slow, methodical approach.",
        "q4",
    )
    q5 = likert_slider(
        "5. I express my emotions outwardly and passionately rather than keeping them bottled up.",
        "q5",
    )
    q6 = likert_slider(
        "6. People around me can easily tell how I am feeling from my energy and tone.",
        "q6",
    )
    q7 = likert_slider(
        "7. I am deeply fascinated by history, rich tradition, and intricate details.",
        "q7",
    )
    q8 = likert_slider(
        "8. I appreciate layered, ornate craftsmanship more than sleek, bare minimalism.",
        "q8",
    )
    q9 = likert_slider(
        "9. I adapt smoothly to sudden changes rather than sticking rigidly to my original plan.",
        "q9",
    )
    q10 = likert_slider(
        "10. I enjoy improvising in the moment when unexpected opportunities arise.",
        "q10",
    )

    submitted = st.form_submit_button("Find My Calligraphy Font!")

# --- 5. PREDICTION & RESULT DISPLAY ---
if submitted:
    dim_structure = (q1 + q2) / 2.0
    dim_tempo = (q3 + q4) / 2.0
    dim_expression = (q5 + q6) / 2.0
    dim_complexity = (q7 + q8) / 2.0
    dim_adaptability = (q9 + q10) / 2.0

    user_vector = np.array(
        [[dim_structure, dim_tempo, dim_expression, dim_complexity, dim_adaptability]]
    )
    st.session_state["cluster_idx"] = int(model.predict(user_vector)[0])
    st.session_state["user_vector"] = user_vector

if "cluster_idx" in st.session_state:
    cluster_idx = st.session_state["cluster_idx"]
    user_vector = st.session_state["user_vector"]
    display_cluster_num = cluster_idx + 1
    font_info = FONT_MAPPINGS[cluster_idx]

    if submitted:
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

    # --- 6. PERSONALIZED NAME CALLIGRAPHY GENERATOR ---
    st.markdown("---")
    st.subheader("✍️ Generate Your Name in This Calligraphy Style")
    st.caption(
        f"Matched with **{font_info['name']}**. Enter your name below to create a personalized calligraphy card."
    )

    has_api_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    with st.form("name_calligraphy_form"):
        user_name = st.text_input(
            "Enter your name",
            placeholder="e.g., Mei Ling or Alex Chen",
            max_chars=30,
        )
        use_llm = st.checkbox(
            "Generate poetic Chinese name via Gemini API",
            value=has_api_key,
            disabled=not has_api_key,
            help="Requires GEMINI_API_KEY or GOOGLE_API_KEY in your environment variables.",
        )
        script_mode = st.selectbox(
            "Chinese character set",
            options=["Traditional Chinese", "Simplified Chinese"],
            disabled=not use_llm,
        )
        render_choice = st.radio(
            "Text to print on the calligraphy card",
            options=["Chinese Name (if generated)", "Original Entered Name", "Both"],
            horizontal=True,
        )
        generate_name_btn = st.form_submit_button("Generate My Calligraphy Name")

    if generate_name_btn:
        cleaned_name = user_name.strip()
        if not cleaned_name:
            st.warning("Please enter a name before generating.")
        else:
            llm_data = None
            if use_llm:
                with st.spinner("Crafting your Chinese name..."):
                    try:
                        llm_data = generate_chinese_name_llm(
                            user_name=cleaned_name,
                            style_name=font_info["name"],
                            traits=font_info["traits"],
                            character_script=script_mode,
                        )
                    except Exception as exc:
                        st.error(f"Could not reach Gemini API: {exc}")

            if llm_data:
                st.info(
                    f"**Chinese Name:** {llm_data['chinese_name']} (*{llm_data['pinyin']}*) — {llm_data['meaning']}"
                )

            cn_name = llm_data["chinese_name"] if llm_data else cleaned_name
            if render_choice == "Original Entered Name" or not llm_data:
                card_text = cleaned_name
            elif render_choice == "Both":
                card_text = f"{cn_name} · {cleaned_name}"
            else:
                card_text = cn_name

            card_img = render_name_calligraphy_image(
                name_text=card_text,
                cluster_idx=cluster_idx,
                style_name=font_info["name"],
            )
            st.image(
                card_img,
                caption=f"{card_text} — {font_info['name']}",
                use_container_width=True,
            )

            buf = BytesIO()
            card_img.save(buf, format="PNG")
            st.download_button(
                label="Download Calligraphy Card (PNG)",
                data=buf.getvalue(),
                file_name=f"{cleaned_name.lower().replace(' ', '_')}_calligraphy.png",
                mime="image/png",
            )
