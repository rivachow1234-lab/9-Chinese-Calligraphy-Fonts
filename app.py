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


# --- HELPER: BACKGROUND & CUSTOM ANIMATIONS ---
def apply_custom_styling():
    # Option A: If you save a local ink wash image at "asset/ink_mountains.jpg", it will load it.
    # Option B: Otherwise, it falls back to a curated Unsplash ink-wash mountain landscape URL.
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

        /* Frosted Rice-Paper Container for Readability */
        .block-container {{
            background-color: rgba(255, 253, 249, 0.88);
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
        "desc": "Elegant, symmetrical, and deeply historic. Ideal for traditionalists who value structure and heritage.",
        "traits": "High Structure, High History",
        "image": "asset/font_0_seal.png",
    },
    1: {
        "name": "Clerical Script (隸書)",
        "desc": "Flat, wide, orderly, and official. Perfect for pragmatic, disciplined, and steady personalities.",
        "traits": "High Structure, Moderate Tempo",
        "image": "asset/font_1_clerical.png",
    },
    2: {
        "name": "Yan Style Standard Script (顏體楷書)",
        "desc": "Bold, dignified, and rigorous. Suits perfectionists who command respect and uphold strong principles.",
        "traits": "High Structure, Low Impulsivity",
        "image": "asset/font_2_yan.png",
    },
    3: {
        "name": "Liu Style Standard Script (柳體楷書)",
        "desc": "Clean, clear, and balanced. Great for steady, reliable, and modern professionals.",
        "traits": "Balanced Structure & Flow",
        "image": "asset/font_3_liu.png",
    },
    4: {
        "name": "Smooth Running Script (淡雅行書)",
        "desc": "Smooth, practical, and graceful. Fits adaptable diplomats who value balance and ease.",
        "traits": "Moderate Flow, High Adaptability",
        "image": "asset/font_4_smooth_running.png",
    },
    5: {
        "name": "Expressive Running Script (狂行書)",
        "desc": "Energetic, fast, and dynamic. Tailored for creative free spirits who love spontaneity.",
        "traits": "High Flow, High Expression",
        "image": "asset/font_5_expressive_running.png",
    },
    6: {
        "name": "Cursive Script (草書)",
        "desc": "Abstract, emotional, and sweeping. Perfect for passionate artists driven by raw emotion.",
        "traits": "High Energy, High Extroversion",
        "image": "asset/font_6_cursive.png",
    },
    7: {
        "name": "Slender Gold Calligraphy (瘦金體書法)",
        "desc": "Simplified, artistic, and minimalist. Matches modern minimalists who appreciate clean aesthetics.",
        "traits": "Low Complexity, High Flow",
        "image": "asset/font_7_slender.png",
    },
    8: {
        "name": "Heavy Splash Ink Style (重墨書法)",
        "desc": "Raw, heavy-set, and commanding. Designed for bold mavericks with an unyielding presence.",
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

# --- 3. MBTI LIKERT SCALE HELPER ---
LIKERT_OPTIONS = {
    "Strongly Disagree": 1.0,
    "Disagree": 2.0,
    "Neutral": 3.0,
    "Agree": 4.0,
    "Strongly Agree": 5.0,
}


def mbti_slider(label, key):
    choice = st.select_slider(
        label,
        options=list(LIKERT_OPTIONS.keys()),
        value="Neutral",
        key=key,
    )
    return LIKERT_OPTIONS[choice]


# --- 4. STREAMLIT UI ---
st.title("🖌️ Chinese Calligraphy Personality Matcher")
st.markdown(
    "Reflect on the **10 statements** below. Our K-Means clustering model will analyze your responses across 5 psychological dimensions to match you with **1 of 9 Chinese Calligraphy Styles**."
)
st.markdown("---")

with st.form("calligraphy_form"):
    st.subheader("Your Personality Assessment")

    # Dimension 1: Structure & Order (Q1 & Q2)
    q1 = mbti_slider(
        "1. I prefer clear rules, neat organization, and structured plans over creative chaos.",
        "q1",
    )
    q2 = mbti_slider(
        "2. I feel most comfortable when my daily schedule and workspace are well-ordered.",
        "q2",
    )

    # Dimension 2: Tempo & Energy (Q3 & Q4)
    q3 = mbti_slider(
        "3. I move, think, and make decisions quickly, often acting on instinct.",
        "q3",
    )
    q4 = mbti_slider(
        "4. I thrive in fast-paced environments rather than taking a slow, methodical approach.",
        "q4",
    )

    # Dimension 3: Expression & Emotion (Q5 & Q6)
    q5 = mbti_slider(
        "5. I express my emotions outwardly and passionately rather than keeping them bottled up.",
        "q5",
    )
    q6 = mbti_slider(
        "6. People around me can easily tell how I am feeling from my energy and tone.",
        "q6",
    )

    # Dimension 4: Complexity vs. Minimalism (Q7 & Q8)
    q7 = mbti_slider(
        "7. I am deeply fascinated by history, rich tradition, and intricate details.",
        "q7",
    )
    q8 = mbti_slider(
        "8. I appreciate layered, ornate craftsmanship more than sleek, bare minimalism.",
        "q8",
    )

    # Dimension 5: Adaptability & Flow (Q9 & Q10)
    q9 = mbti_slider(
        "9. I adapt smoothly to sudden changes rather than sticking rigidly to my original plan.",
        "q9",
    )
    q10 = mbti_slider(
        "10. I enjoy improvising in the moment when unexpected opportunities arise.",
        "q10",
    )

    submitted = st.form_submit_button("Find My Calligraphy Font!")

# --- 5. PREDICTION & RESULT DISPLAY ---
if submitted:
    # Average each pair of questions into the 5 core dimensions
    dim_structure = (q1 + q2) / 2.0
    dim_tempo = (q3 + q4) / 2.0
    dim_expression = (q5 + q6) / 2.0
    dim_complexity = (q7 + q8) / 2.0
    dim_adaptability = (q9 + q10) / 2.0

    user_vector = np.array(
        [[dim_structure, dim_tempo, dim_expression, dim_complexity, dim_adaptability]]
    )

    # Predict cluster
    cluster_id = model.predict(user_vector)[0]
    font_info = FONT_MAPPINGS[cluster_id]

    # Trigger falling hearts instead of balloons
    trigger_hearts()

    st.markdown("---")
    st.subheader("✨ Your Calligraphy Match Result")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Assigned Cluster ID", value=f"Cluster #{cluster_id}")
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
                "Cluster": list(FONT_MAPPINGS.keys()),
                "Font Style": [f["name"] for f in FONT_MAPPINGS.values()],
                "Distance Score (Lower is Closer)": distances,
            }
        ).sort_values("Distance Score (Lower is Closer)")
        st.dataframe(dist_df, hide_index=True)
