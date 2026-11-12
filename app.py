import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st

# --- 1. CONFIGURATION & FONT MAPPINGS ---
# Map the 9 cluster IDs (0 to 8) to your 9 Chinese Calligraphy styles
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
    # Create 9 distinct centers around which to cluster synthetic samples
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

    # Generate random points around these centers to train K-Means robustly
    data = []
    for center in centers:
        cluster_points = np.random.normal(loc=center, scale=0.4, size=(30, 5))
        data.append(cluster_points)

    X = np.vstack(data)
    # Clip values between 1 and 5 (Likert scale bounds)
    X = np.clip(X, 1.0, 5.0)

    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans


model = train_kmeans_model()

# --- 3. STREAMLIT UI ---
st.set_page_config(
    page_title="Personality Calligraphy Matcher", page_icon="🖌️", layout="centered"
)

st.title("🖌️ Chinese Calligraphy Personality Matcher")
st.markdown(
    "Answer the 5 psychological dimensions below. Our K-Means clustering model will map your unique personality profile to **1 of 9 Chinese Calligraphy Fonts**."
)
st.markdown("---")

with st.form("calligraphy_form"):
    st.subheader("Your Personality Assessment")

    # Dimension 1: Structure & Order
    q1 = st.slider(
        "1. Structure & Order: I prefer clear rules, neat organization, and structured plans over creative chaos.",
        1,
        5,
        3,
        help="1 = Total Chaos/Spontaneous, 5 = Highly Structured",
    )

    # Dimension 2: Tempo & Energy
    q2 = st.slider(
        "2. Tempo & Pacing: I move, think, and make decisions quickly, acting on impulse.",
        1,
        5,
        3,
        help="1 = Slow and deliberate, 2 = Fast and impulsive",
    )

    # Dimension 3: Expression & Emotion
    q3 = st.slider(
        "3. Emotional Expression: I express my emotions outwardly and passionately rather than keeping them bottled up.",
        1,
        5,
        3,
        help="1 = Private/Reserved, 5 = Outwardly Passionate",
    )

    # Dimension 4: Complexity vs. Minimalism
    q4 = st.slider(
        "4. Appreciation for Complexity: I am deeply fascinated by history, tradition, and intricate details.",
        1,
        5,
        3,
        help="1 = Modern Minimalist, 5 = Deeply Traditional/Complex",
    )

    # Dimension 5: Adaptability & Flow
    q5 = st.slider(
        "5. Adaptability & Flexibility: I adapt smoothly to changing circumstances rather than sticking rigidly to plans.",
        1,
        5,
        3,
        help="1 = Rigidly fixed, 5 = Highly fluid/adaptive",
    )

    submitted = st.form_submit_button("Find My Calligraphy Font!")

# --- 4. PREDICTION & RESULT DISPLAY ---
if submitted:
    # Construct user feature vector matching the model's expected 5 features
    user_vector = np.array([[q1, q2, q3, q4, q5]])

    # Predict cluster
    cluster_id = model.predict(user_vector)[0]
    font_info = FONT_MAPPINGS[cluster_id]

    st.balloons()
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
    # Display the physical preview image of the font
    st.image(
        font_info["image"],
        caption=f"Preview: {font_info['name']}",
        width="stretch",

    # Optional: Display distances to centroids for transparency
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
