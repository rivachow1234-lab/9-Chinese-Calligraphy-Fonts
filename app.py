import base64
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st

# --- 0. PAGE CONFIG ---
st.set_page_config(
    page_title="Personality Calligraphy Matcher",
    page_icon="🖌️",
    layout="centered",
)


# --- HELPER: BACKGROUND, TIMES NEW ROMAN & STYLING ---
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

        /* Frosted Rice-Paper Container */
        .block-container {{
            background-color: rgba(255, 253, 249, 0.92);
            padding: 2.5rem 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
            margin-top: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(180, 175, 165, 0.4);
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


# --- 2. INITIALIZE K-MEANS MODEL ---
@st.cache_resource
def train_kmeans_model():
    np.random.seed(42)
    centers = np.array(
        [
            [6.5, 2.0, 2.5, 6.5, 2.5],
            [5.5, 3.5, 3.0, 5.0, 3.0],
            [6.5, 2.5, 4.0, 5.5, 2.5],
            [4.0, 4.0, 4.0, 4.0, 4.0],
            [3.0, 5.0, 5.0, 3.0, 6.0],
            [2.5, 6.0, 6.0, 2.5, 6.5],
            [1.5, 6.8, 6.8, 1.5, 6.5],
            [2.5, 4.0, 2.5, 1.8, 5.0],
            [5.0, 6.5, 6.5, 4.0, 5.0],
        ]
    )
    data = []
    for center in centers:
        cluster_points = np.random.normal(loc=center, scale=0.5, size=(30, 5))
        data.append(cluster_points)

    X = np.vstack(data)
    X = np.clip(X, 1.0, 7.0)
    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans


model = train_kmeans_model()


# --- 3. 16PERSONALITIES SCALE HELPER (Agree -> Bubbles -> Disagree) ---
def mbti_scale_question(question_text, key):
    st.markdown(f"**{question_text}**")

    # Render custom HTML container matching the 16Personalities layout exactly
    html_code = f"""
    <style>
      .scale-wrapper {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: rgba(242, 239, 233, 0.75);
        padding: 12px 18px;
        border-radius: 12px;
        border: 1px solid rgba(180, 175, 165, 0.35);
        margin-bottom: 16px;
      }}
      .lbl-agree {{
        font-family: 'Times New Roman', Times, serif;
        font-weight: bold;
        color: #1b4332;
        font-size: 1rem;
        width: 70px;
      }}
      .lbl-disagree {{
        font-family: 'Times New Roman', Times, serif;
        font-weight: bold;
        color: #3c096c;
        font-size: 1rem;
        width: 80px;
        text-align: right;
      }}
      .bubbles-row {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
        flex-grow: 1;
      }}
    </style>
    <div class="scale-wrapper">
      <div class="lbl-agree">Agree</div>
      <div class="bubbles-row">
        <!-- Bubbles will be injected here via Streamlit columns -->
      </div>
      <div class="lbl-disagree">Disagree</div>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    # 7 distinct radio options from Agree (left) to Disagree (right)
    # Sizes: 28px, 24px, 18px, 12px (neutral), 18px, 24px, 28px
    options = [
        ("Strongly Agree", 7.0, "28px", "#1b4332"),
        ("Agree", 6.0, "23px", "#2d6a4f"),
        ("Somewhat Agree", 5.0, "18px", "#52b788"),
        ("Neutral", 4.0, "13px", "#6c757d"),
        ("Somewhat Disagree", 3.0, "18px", "#7b2cbf"),
        ("Disagree", 2.0, "23px", "#560bad"),
        ("Strongly Disagree", 1.0, "28px", "#3c096c"),
    ]

    # Custom styling for radio buttons to arrange them horizontally with exact scaling & colors
    st.markdown(
        f"""
        <style>
        div[data-testid="stRadio"] div[role="radiogroup"] {{
            display: flex !important;
            flex-direction: row !important;
            justify-content: center !important;
            gap: 18px !important;
            margin-top: -55px !important;
            margin-bottom: 10px !important;
        }}
        div[data-testid="stRadio"] div[role="radiogroup"] label {{
            cursor: pointer !important;
        }}
        div[data-testid="stRadio"] div[role="radiogroup"] label p {{
            display: none !important;
        }}
        /* Sizing and colors for the 7 radio buttons */
        div[data-testid="stRadio"] div[role="radiogroup"] label:nth-child(1) input {{ width: 28px; height: 28px; accent-color: #1b4332; }}
        div[data-testid="stRadio"] div[role="radiogroup"] label:nth-child(2) input {{ width: 23px; height: 23px; accent-color: #2d6a4f; }}
        div[data-testid="stRadio"] div[role="radiogroup"] label:nth-child(3) input {{ width: 18px; height: 18px; accent-color: #52b788; }}
        div[data-testid="stRadio"] div[role="radiogroup"] label:nth-child(4) input {{ width: 13px; height: 13px; accent-color: #6c757d; }}
        div[data-testid="stRadio"] div[role="radiogroup"] label:nth-child(5) input {{ width: 18px; height: 18px; accent-color: #7b2cbf; }}
        div[data-testid="stRadio"] div[role="radiogroup"] label:nth-child(6) input {{ width: 23px; height: 23px; accent-color: #560bad; }}
        div[data-testid="stRadio"] div[role="radiogroup"] label:nth-child(7) input {{ width: 28px; height: 28px; accent-color: #3c096c; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    choice = st.radio(
        label=question_text,
        options=[opt[0] for opt in options],
        index=3,  # Default to Neutral
        horizontal=True,
        label_visibility="collapsed",
        key=key,
    )

    # Return score corresponding to chosen option
    score_dict = {opt[0]: opt[1] for opt in options}
    return score_dict[choice]


# --- 4. STREAMLIT UI ---
st.title("🖌️ Chinese Calligraphy Personality Matcher")
st.markdown(
    "Select the bubble that best reflects your stance on each of the **10 distinct statements** below. "
    "Our K-Means clustering model analyzes your profile across 5 psychological dimensions to match you with **1 of 9 Chinese Calligraphy Styles**."
)
st.markdown("---")

with st.form("calligraphy_form"):
    st.subheader("Your Personality Assessment")

    q1 = mbti_scale_question(
        "1. I strictly adhere to detailed daily checklists and rigid schedules to manage my time.",
        "q1",
    )
    q2 = mbti_scale_question(
        "2. When planning a new project, I prefer establishing an unchanging roadmap rather than adapting on the fly.",
        "q2",
    )
    q3 = mbti_scale_question(
        "3. I make important decisions instantly based on my immediate intuition rather than lengthy deliberation.",
        "q3",
    )
    q4 = mbti_scale_question(
        "4. I feel deeply impatient and frustrated when daily tasks or conversations move too slowly.",
        "q4",
    )
    q5 = mbti_scale_question(
        "5. I openly and passionately wear my heart on my sleeve, letting everyone around me know my exact emotional state.",
        "q5",
    )
    q6 = mbti_scale_question(
        "6. When strong emotions hit me, I express them outward with intense vocal and physical energy rather than keeping them internal.",
        "q6",
    )
    q7 = mbti_scale_question(
        "7. I am profoundly captivated by ancient historical archives, traditional philosophies, and intricate cultural heritage.",
        "q7",
    )
    q8 = mbti_scale_question(
        "8. I prefer ornate, highly detailed craftsmanship over sleek, bare, modern minimalist aesthetics.",
        "q8",
    )
    q9 = mbti_scale_question(
        "9. Sudden and radical changes in my environment excite and energize me instead of causing me stress.",
        "q9",
    )
    q10 = mbti_scale_question(
        "10. I effortlessly throw away my original plans and improvise the moment unexpected obstacles arise.",
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
