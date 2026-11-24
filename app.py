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
        :root, .stApp {{
            color-scheme: light only !important;
        }}

        /* Global font: Times New Roman, dark text */
        html, body, [class*="css"], .stApp, .stMarkdown, .stText,
        p, h1, h2, h3, h4, h5, h6, label, .stSlider label,
        .stMetric, .stSuccess, .stInfo, .stWarning, .stError, .stExpander {{
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

        /* --- FIX FOR MOBILE DARK MODE ON RESULT & OUTPUT ALERT CARDS --- */
        div[data-testid="stAlert"],
        div[data-testid="stAlert"] > div,
        div[data-testid="stAlert"] [data-testid="stAlertContent"],
        div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {{
            background-color: #eef5ec !important;
            border-color: #bfd4ba !important;
            border-radius: 12px !important;
            color: #1a1a1a !important;
        }}
        div[data-testid="stAlert"] {{
            border: 1px solid #bfd4ba !important;
        }}
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span,
        div[data-testid="stAlert"] strong,
        div[data-testid="stAlert"] em,
        div[data-testid="stAlert"] div {{
            color: #1a1a1a !important;
        }}
        div[data-testid="stAlert"] svg {{
            color: #1a1a1a !important;
            fill: #1a1a1a !important;
        }}

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

        /* Light-colored submit & download buttons */
        div.stButton > button[kind="primary"],
        div.stButton > button,
        div.stDownloadButton > button,
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
        div.stDownloadButton > button:hover,
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
  "aaron": "亞倫",
  "abbey": "艾比",
  "abbie": "艾比",
  "abby": "艾比",
  "abel": "亞伯",
  "abigail": "艾比蓋兒",
  "abraham": "亞伯拉罕",
  "ada": "艾達",
  "adam": "亞當",
  "addison": "艾迪生",
  "adela": "愛黛拉",
  "adelaide": "愛黛萊德",
  "adele": "愛黛兒",
  "adeline": "愛德琳",
  "adrian": "亞德里恩",
  "adriana": "亞德里安娜",
  "adrienne": "亞德里安娜",
  "agatha": "阿嘉莎",
  "agnes": "艾格尼絲",
  "aida": "阿伊達",
  "aidan": "艾登",
  "aiden": "艾登",
  "aileen": "艾琳",
  "aimee": "艾美",
  "ainsley": "安斯利",
  "aisha": "愛莎",
  "alan": "艾倫",
  "alana": "艾拉娜",
  "albert": "亞伯特",
  "alberta": "亞伯塔",
  "alec": "亞歷克",
  "alejandra": "亞歷杭德拉",
  "alejandro": "亞歷杭德羅",
  "alex": "亞歷克斯",
  "alexa": "亞歷克莎",
  "alexander": "亞歷山大",
  "alexandra": "亞歷山德拉",
  "alexandria": "亞歷山德里亞",
  "alexis": "亞歷克西斯",
  "alfred": "阿爾弗雷德",
  "ali": "阿里",
  "alice": "愛麗絲",
  "alicia": "艾莉西亞",
  "alina": "艾琳娜",
  "alison": "艾莉森",
  "alissa": "艾莉莎",
  "allan": "艾倫",
  "allen": "艾倫",
  "allie": "艾莉",
  "allison": "艾莉森",
  "ally": "艾莉",
  "allyson": "艾莉森",
  "alma": "阿爾瑪",
  "alvin": "阿爾文",
  "alyssa": "艾莉莎",
  "amanda": "阿曼達",
  "amber": "安珀",
  "amelia": "艾蜜莉亞",
  "amy": "艾美",
  "ana": "安娜",
  "anastasia": "安娜塔西亞",
  "andre": "安德烈",
  "andrea": "安德莉亞",
  "andres": "安德烈斯",
  "andrew": "安德魯",
  "andy": "安迪",
  "angel": "安琪兒",
  "angela": "安琪拉",
  "angelica": "安潔莉卡",
  "angelina": "安潔莉娜",
  "angelo": "安傑洛",
  "angie": "安琪",
  "anita": "安妮塔",
  "ann": "安",
  "anna": "安娜",
  "annabelle": "安娜貝爾",
  "anne": "安妮",
  "annette": "安妮特",
  "annie": "安妮",
  "anthony": "安東尼",
  "antoinette": "安東妮特",
  "anton": "安東",
  "antonia": "安東妮亞",
  "antonio": "安東尼奧",
  "april": "艾普爾",
  "arabella": "阿拉貝拉",
  "archie": "阿奇",
  "ari": "阿里",
  "aria": "艾莉亞",
  "ariana": "亞莉安娜",
  "arianna": "亞莉安娜",
  "ariel": "艾莉兒",
  "arlene": "阿琳",
  "armando": "阿曼多",
  "arnold": "阿諾",
  "arthur": "亞瑟",
  "arya": "艾莉亞",
  "ash": "艾許",
  "ashlee": "艾希莉",
  "ashleigh": "艾希莉",
  "ashley": "艾希莉",
  "ashton": "艾希頓",
  "astrid": "阿斯特麗德",
  "athena": "雅典娜",
  "aubrey": "奧布里",
  "audrey": "奧黛麗",
  "august": "奧古斯特",
  "augustus": "奧古斯都",
  "aurelia": "奧蕾莉亞",
  "aurora": "奧蘿拉",
  "austin": "奧斯汀",
  "autumn": "奧特姆",
  "ava": "艾娃",
  "avery": "艾弗里",
  "axel": "阿克塞爾",
  "ayla": "艾拉",
  "bailey": "貝利",
  "barbara": "芭芭拉",
  "barry": "巴里",
  "bart": "巴特",
  "basil": "巴茲爾",
  "beatrice": "碧翠絲",
  "beau": "博",
  "becca": "貝卡",
  "becky": "貝基",
  "belinda": "貝琳達",
  "bella": "貝拉",
  "ben": "班",
  "benedict": "班奈狄克",
  "benjamin": "班傑明",
  "bennett": "班奈特",
  "benny": "班尼",
  "bernard": "伯納德",
  "bernice": "伯妮絲",
  "bert": "伯特",
  "bertha": "伯莎",
  "bessie": "貝西",
  "beth": "貝絲",
  "bethany": "貝瑟妮",
  "betsy": "貝琪",
  "betty": "貝蒂",
  "beverly": "貝弗莉",
  "bianca": "碧昂卡",
  "bill": "比爾",
  "billie": "比利",
  "billy": "比利",
  "birdy": "柏蒂",
  "blair": "布萊爾",
  "blaire": "布萊爾",
  "blake": "布萊克",
  "blanca": "布蘭卡",
  "blanche": "布蘭琪",
  "bob": "鮑勃",
  "bobbie": "鮑比",
  "bobby": "鮑比",
  "bonnie": "邦妮",
  "boris": "鮑里斯",
  "brad": "布萊德",
  "braden": "布雷登",
  "bradley": "布萊德利",
  "brady": "布雷迪",
  "brandon": "布蘭登",
  "brandy": "布蘭蒂",
  "brayden": "布雷登",
  "breanna": "布麗安娜",
  "brenda": "布蘭達",
  "brendan": "布倫丹",
  "brennan": "布倫南",
  "brent": "布倫特",
  "brett": "布雷特",
  "brian": "布萊恩",
  "briana": "布麗安娜",
  "brianna": "布麗安娜",
  "bridget": "布麗姬特",
  "brielle": "布麗埃爾",
  "britney": "布蘭妮",
  "brittany": "布列塔尼",
  "brock": "布羅克",
  "brody": "布羅迪",
  "brooke": "布魯克",
  "brooklyn": "布魯克林",
  "brooks": "布魯克斯",
  "bruce": "布魯斯",
  "bruno": "布魯諾",
  "bryan": "布萊恩",
  "bryant": "布萊恩特",
  "bryce": "布萊斯",
  "bryson": "布萊森",
  "byron": "拜倫",
  "cade": "凱德",
  "caden": "凱登",
  "caesar": "凱撒",
  "caitlin": "凱特琳",
  "caitlyn": "凱特琳",
  "caleb": "迦勒",
  "callie": "凱莉",
  "callum": "卡勒姆",
  "calvin": "卡爾文",
  "camden": "卡姆登",
  "cameron": "卡麥隆",
  "camila": "卡蜜拉",
  "camilla": "卡蜜拉",
  "camille": "卡蜜兒",
  "candace": "坎蒂絲",
  "candice": "坎蒂絲",
  "cara": "卡拉",
  "carl": "卡爾",
  "carla": "卡拉",
  "carlos": "卡洛斯",
  "carlton": "卡爾頓",
  "carly": "卡莉",
  "carmen": "卡門",
  "carol": "卡蘿",
  "carolina": "卡羅萊納",
  "caroline": "卡羅琳",
  "carolyn": "卡羅琳",
  "carrie": "凱莉",
  "carson": "卡森",
  "carter": "卡特",
  "casey": "凱西",
  "cassandra": "卡珊德拉",
  "cassidy": "卡西迪",
  "cassie": "凱西",
  "catalina": "卡塔琳娜",
  "catherine": "凱瑟琳",
  "cathy": "凱西",
  "cecelia": "西西莉亞",
  "cecil": "塞西爾",
  "cecilia": "西西莉亞",
  "cedric": "塞德里克",
  "celeste": "塞萊斯特",
  "celia": "西莉亞",
  "celine": "席琳",
  "chad": "查德",
  "chance": "錢斯",
  "chandler": "錢德勒",
  "chanel": "香奈兒",
  "chantal": "尚塔爾",
  "charity": "查莉蒂",
  "charlene": "夏琳",
  "charles": "查爾斯",
  "charlie": "查理",
  "charlotte": "夏洛特",
  "chase": "蔡斯",
  "chelsea": "雀兒喜",
  "cher": "雪兒",
  "cheri": "雪莉",
  "cheryl": "雪莉爾",
  "chester": "切斯特",
  "chloe": "克蘿伊",
  "chris": "克里斯",
  "christian": "克里斯蒂安",
  "christiana": "克里斯蒂安娜",
  "christina": "克里斯蒂娜",
  "christine": "克里斯汀",
  "christopher": "克里斯多福",
  "christy": "克里斯蒂",
  "chuck": "查克",
  "cindy": "辛蒂",
  "claire": "克萊兒",
  "clara": "克拉拉",
  "clare": "克萊兒",
  "clarence": "克拉倫斯",
  "clarissa": "克拉麗莎",
  "clark": "克拉克",
  "claude": "克勞德",
  "claudia": "克勞蒂亞",
  "clay": "克萊",
  "clayton": "克萊頓",
  "clement": "克萊門特",
  "clementine": "克萊門汀",
  "cliff": "克里夫",
  "clifford": "克里福德",
  "clint": "克林特",
  "clinton": "克林頓",
  "clyde": "克萊德",
  "cody": "科迪",
  "colby": "科爾比",
  "cole": "科爾",
  "coleman": "科爾曼",
  "colette": "科萊特",
  "colin": "科林",
  "colleen": "科琳",
  "collin": "科林",
  "colton": "科爾頓",
  "connie": "康妮",
  "connor": "康納",
  "conor": "康納",
  "conrad": "康拉德",
  "constance": "康斯坦絲",
  "cooper": "庫珀",
  "cora": "科拉",
  "corbin": "科賓",
  "corey": "科里",
  "corinne": "科琳",
  "cornelius": "科尼利厄斯",
  "cory": "科里",
  "courtney": "寇特妮",
  "craig": "克雷格",
  "cristian": "克里斯蒂安",
  "cristina": "克里斯蒂娜",
  "crystal": "克莉絲朵",
  "curtis": "柯蒂斯",
  "cynthia": "辛西亞",
  "cyrus": "賽勒斯",
  "jeffery": "傑佛瑞",
  "leo": "里奧",
  "nicholas": "尼古拉斯",
  "riva": "瑞娃",
  "scarlett": "史嘉蕾",
  "soso": "蘇蘇",
  "terry": "泰瑞",
  "toto": "托托",
  "daisy": "黛西",
  "dakota": "達科塔",
  "dale": "戴爾",
  "dallas": "達拉斯",
  "dalton": "道爾頓",
  "damian": "達米安",
  "damien": "達米恩",
  "damon": "戴蒙",
  "dan": "丹",
  "dana": "達娜",
  "dane": "戴恩",
  "daniel": "丹尼爾",
  "daniela": "丹妮拉",
  "daniella": "丹妮拉",
  "danielle": "丹妮爾",
  "danny": "丹尼",
  "dante": "但丁",
  "daphne": "達芙妮",
  "darcy": "達西",
  "daria": "達莉亞",
  "darian": "達里安",
  "darien": "達里恩",
  "dario": "達里奧",
  "darius": "達里厄斯",
  "darlene": "達琳",
  "darnell": "達內爾",
  "darrell": "達雷爾",
  "darren": "達倫",
  "darrin": "達林",
  "darryl": "達里爾",
  "darwin": "達爾文",
  "daryl": "達里爾",
  "dave": "戴夫",
  "david": "大衛",
  "davina": "達維娜",
  "davis": "戴維斯",
  "dawn": "唐恩",
  "dawson": "道森",
  "dax": "達克斯",
  "dayton": "代頓",
  "dean": "迪恩",
  "deanna": "迪安娜",
  "debbie": "黛比",
  "deborah": "黛博拉",
  "debra": "黛布拉",
  "declan": "德克蘭",
  "dee": "迪",
  "delaney": "德萊尼",
  "delia": "迪莉婭",
  "delilah": "黛莉拉",
  "della": "德拉",
  "delores": "德洛麗絲",
  "demetrius": "德米特里厄斯",
  "denise": "丹妮絲",
  "dennis": "丹尼斯",
  "denver": "丹佛",
  "derek": "德瑞克",
  "derrick": "德瑞克",
  "desiree": "德西蕾",
  "desmond": "戴斯蒙",
  "destiny": "黛絲蒂妮",
  "devin": "德文",
  "devon": "德文",
  "dewey": "杜威",
  "dexter": "德克斯特",
  "diamond": "黛蒙德",
  "diana": "黛安娜",
  "diane": "黛安",
  "dianna": "黛安娜",
  "dianne": "黛安",
  "dick": "迪克",
  "diego": "迪亞哥",
  "dillon": "狄倫",
  "dina": "迪娜",
  "dinah": "黛娜",
  "dion": "迪翁",
  "dirk": "德克",
  "dixie": "迪克西",
  "dolores": "多洛雷斯",
  "dominic": "多米尼克",
  "dominick": "多米尼克",
  "dominique": "多米妮克",
  "don": "唐",
  "donald": "唐納德",
  "donna": "唐娜",
  "donovan": "多諾萬",
  "dora": "朵拉",
  "doreen": "朵琳",
  "doris": "桃樂絲",
  "dorothy": "桃樂絲",
  "doug": "道格",
  "douglas": "道格拉斯",
  "drake": "德雷克",
  "drew": "德魯",
  "duane": "杜安",
  "dudley": "達德利",
  "duke": "杜克",
  "duncan": "鄧肯",
  "dustin": "達斯汀",
  "dwayne": "巨石",
  "dwight": "德懷特",
  "dylan": "狄倫",
  "earl": "厄爾",
  "earnest": "歐內斯特",
  "ebony": "艾博妮",
  "ed": "艾德",
  "eddie": "艾迪",
  "eddy": "艾迪",
  "eden": "伊甸",
  "edgar": "艾德加",
  "edith": "伊迪絲",
  "edmund": "艾德蒙",
  "edna": "艾德娜",
  "eduardo": "愛德華多",
  "edward": "愛德華",
  "edwin": "艾德溫",
  "edwina": "艾德溫娜",
  "effie": "埃菲",
  "eileen": "艾琳",
  "elaina": "伊萊娜",
  "elaine": "伊蓮",
  "eleanor": "埃莉諾",
  "elena": "艾蓮娜",
  "eli": "伊萊",
  "eliana": "艾莉安娜",
  "elias": "以利亞",
  "elijah": "以利亞",
  "elisa": "伊莉莎",
  "elisabeth": "伊莉莎白",
  "elise": "艾莉絲",
  "elisha": "以利沙",
  "eliza": "伊莉莎",
  "elizabeth": "伊莉莎白",
  "ella": "艾拉",
  "elle": "艾兒",
  "ellen": "艾倫",
  "ellie": "艾莉",
  "elliot": "艾略特",
  "elliott": "艾略特",
  "ellis": "艾利斯",
  "elmer": "埃爾默",
  "eloise": "艾洛伊斯",
  "elsa": "艾莎",
  "elsie": "艾爾西",
  "elton": "艾爾頓",
  "elvin": "埃爾文",
  "elvis": "貓王",
  "elyse": "艾莉絲",
  "emanuel": "伊曼紐爾",
  "ember": "安珀",
  "emerson": "愛默生",
  "emery": "艾默里",
  "emil": "埃米爾",
  "emilia": "艾蜜莉亞",
  "emiliano": "埃米利亞諾",
  "emilio": "埃米利奧",
  "emily": "艾蜜莉",
  "emma": "艾瑪",
  "emmanuel": "伊曼紐爾",
  "emmett": "艾米特",
  "enid": "伊妮德",
  "enrique": "恩里克",
  "eric": "艾瑞克",
  "erica": "艾瑞卡",
  "erick": "艾瑞克",
  "ericka": "艾瑞卡",
  "erik": "艾瑞克",
  "erika": "艾瑞卡",
  "erin": "艾琳",
  "ernest": "歐內斯特",
  "ernestine": "歐內斯廷",
  "ernesto": "埃內斯托",
  "ernie": "厄尼",
  "ervin": "歐文",
  "erwin": "歐文",
  "esmeralda": "埃斯梅拉達",
  "esperanza": "埃斯佩蘭薩",
  "esteban": "埃斯特萬",
  "estelle": "艾絲黛兒",
  "esther": "以斯帖",
  "ethan": "伊森",
  "ethel": "艾瑟爾",
  "eugene": "尤金",
  "eugenia": "尤金妮亞",
  "eunice": "尤妮絲",
  "eva": "伊娃",
  "evan": "伊凡",
  "evangeline": "伊凡潔琳",
  "eve": "夏娃",
  "evelyn": "伊芙琳",
  "everett": "艾弗雷特",
  "ezekiel": "以西結",
  "ezra": "以斯拉",
  "fabian": "法比安",
  "faith": "費絲",
  "fallon": "法倫",
  "fanny": "芬妮",
  "farah": "法拉",
  "farrah": "法拉",
  "fatima": "法蒂瑪",
  "faye": "費伊",
  "felicia": "費莉西亞",
  "felicity": "費莉西蒂",
  "felix": "菲利克斯",
  "fern": "芬恩",
  "fernando": "費爾南多",
  "fiona": "費歐娜",
  "fletcher": "弗萊徹",
  "flora": "芙蘿拉",
  "florence": "佛羅倫絲",
  "floyd": "弗洛伊德",
  "flynn": "弗林",
  "ford": "福特",
  "forest": "福雷斯特",
  "forrest": "福雷斯特",
  "frances": "法蘭西絲",
  "francesca": "法蘭西絲卡",
  "francine": "法蘭辛",
  "francis": "法蘭西斯",
  "francisco": "法蘭西斯科",
  "frank": "法蘭克",
  "frankie": "法蘭基",
  "franklin": "富蘭克林",
  "fred": "弗雷德",
  "freddie": "弗雷迪",
  "freddy": "弗雷迪",
  "frederick": "弗雷德里克",
  "freya": "芙蕾雅",
  "frida": "芙烈達",
  "frieda": "弗里達",
  "gabriel": "加百列",
  "gabriela": "加布里埃拉",
  "gabriella": "加布里埃拉",
  "gabrielle": "加布里埃爾",
  "gage": "蓋奇",
  "gail": "蓋兒",
  "garrett": "加勒特",
  "garrison": "加里森",
  "garry": "蓋瑞",
  "gary": "蓋瑞",
  "gavin": "蓋文",
  "gayle": "蓋兒",
  "gemma": "潔瑪",
  "gene": "吉恩",
  "genesis": "珍妮西絲",
  "geneva": "日內瓦",
  "genevieve": "珍妮薇芙",
  "geoffrey": "傑佛瑞",
  "george": "喬治",
  "georgia": "喬治亞",
  "georgina": "喬治娜",
  "gerald": "傑拉德",
  "geraldine": "傑拉爾丁",
  "gerard": "傑拉德",
  "gerardo": "赫拉爾多",
  "gerry": "格里",
  "gertrude": "格特魯德",
  "gia": "吉婭",
  "gianna": "吉安娜",
  "gibson": "吉布森",
  "gideon": "基甸",
  "gigi": "吉吉",
  "gilbert": "吉爾伯特",
  "gillian": "吉莉安",
  "gina": "吉娜",
  "ginger": "金潔",
  "ginny": "金妮",
  "giovanni": "喬瓦尼",
  "giselle": "吉賽兒",
  "gladys": "格拉迪絲",
  "glen": "葛倫",
  "glenda": "格倫達",
  "glenn": "葛倫",
  "gloria": "葛洛莉亞",
  "gordon": "戈登",
  "grace": "葛蕾絲",
  "gracie": "葛蕾西",
  "grady": "格雷迪",
  "graham": "葛拉漢",
  "grant": "格蘭特",
  "grayson": "格雷森",
  "greg": "格雷格",
  "gregg": "格雷格",
  "gregory": "格雷戈里",
  "greta": "葛莉塔",
  "gretchen": "格蕾琴",
  "griffin": "格里芬",
  "guadalupe": "瓜達盧佩",
  "guillermo": "吉列爾莫",
  "gunnar": "貢納爾",
  "gunner": "岡納",
  "gus": "格斯",
  "gustavo": "古斯塔沃",
  "guy": "蓋伊",
  "gwen": "關",
  "gwendolyn": "關朵琳",
  "hailey": "海莉",
  "hal": "哈爾",
  "haley": "海莉",
  "hallie": "哈莉",
  "hamilton": "漢密爾頓",
  "hank": "漢克",
  "hanna": "漢娜",
  "hannah": "漢娜",
  "hans": "漢斯",
  "harlan": "哈蘭",
  "harley": "哈莉",
  "harmony": "哈莫妮",
  "harold": "哈羅德",
  "harper": "哈珀",
  "harriet": "哈麗特",
  "harris": "哈里斯",
  "harrison": "哈里森",
  "harry": "哈利",
  "harvey": "哈維",
  "hattie": "海蒂",
  "haven": "海文",
  "hayden": "海登",
  "hayes": "海斯",
  "haylee": "海莉",
  "hayley": "海莉",
  "hazel": "海瑟",
  "heath": "希斯",
  "heather": "海瑟",
  "heaven": "海文",
  "hector": "赫克托",
  "heidi": "海蒂",
  "helen": "海倫",
  "helena": "海倫娜",
  "helene": "海倫",
  "henrietta": "亨利埃塔",
  "henry": "亨利",
  "herbert": "赫伯特",
  "herman": "赫爾曼",
  "hilda": "希爾達",
  "hilary": "希拉蕊",
  "hillary": "希拉蕊",
  "holden": "霍爾頓",
  "hollie": "荷莉",
  "holly": "荷莉",
  "homer": "荷馬",
  "hope": "霍普",
  "horace": "霍勒斯",
  "howard": "霍華德",
  "hudson": "哈德森",
  "hugh": "休",
  "hugo": "雨果",
  "hunter": "杭特",
  "ian": "伊恩",
  "ida": "艾達",
  "ignacio": "伊格納西奧",
  "ike": "艾克",
  "iliana": "伊利安娜",
  "imani": "伊瑪尼",
  "imogen": "伊莫金",
  "ina": "伊娜",
  "india": "印第安娜",
  "indiana": "印第安娜",
  "ines": "伊內絲",
  "ingrid": "英格麗",
  "ira": "艾拉",
  "irene": "艾琳",
  "iris": "艾莉絲",
  "irma": "厄瑪",
  "irvin": "歐文",
  "irving": "歐文",
  "isaac": "以撒",
  "isabel": "伊莎貝爾",
  "isabela": "伊莎貝拉",
  "isabella": "伊莎貝拉",
  "isabelle": "伊莎貝爾",
  "isaiah": "以賽亞",
  "isaias": "以賽亞",
  "ishmael": "以實瑪利",
  "isidora": "伊西多拉",
  "isla": "艾拉",
  "ismael": "伊斯梅爾",
  "israel": "以色列",
  "ivan": "伊凡",
  "ivana": "伊凡娜",
  "ivory": "艾芙莉",
  "ivy": "艾薇",
  "izabella": "伊莎貝拉"
  "jace": "傑斯",
  "jack": "傑克",
  "jackie": "賈姬",
  "jackson": "傑克森",
  "jacob": "雅各",
  "jacqueline": "賈桂琳",
  "jacquelyn": "賈桂琳",
  "jada": "潔達",
  "jade": "潔德",
  "jaden": "傑登",
  "jaelyn": "潔琳",
  "jaime": "海梅",
  "jake": "傑克",
  "jakob": "雅各",
  "jalen": "傑倫",
  "james": "詹姆斯",
  "jameson": "詹姆森",
  "jamie": "傑米",
  "jana": "珍娜",
  "jane": "珍",
  "janelle": "珍妮爾",
  "janet": "珍妮特",
  "janice": "珍妮絲",
  "janie": "珍妮",
  "janine": "珍妮",
  "jared": "傑瑞德",
  "jarrett": "賈勒特",
  "jarvis": "賈維斯",
  "jasmin": "茉莉",
  "jasmine": "茉莉",
  "jason": "傑森",
  "jasper": "賈斯珀",
  "javier": "哈維爾",
  "jax": "賈克斯",
  "jaxon": "傑克森",
  "jay": "傑",
  "jayce": "傑斯",
  "jayden": "傑登",
  "jayla": "潔拉",
  "jaylen": "傑倫",
  "jaylene": "潔琳",
  "jaylin": "傑林",
  "jayme": "傑米",
  "jayson": "傑森",
  "jean": "珍",
  "jeanette": "珍妮特",
  "jeanine": "珍妮",
  "jeanne": "珍",
  "jeannette": "珍妮特",
  "jed": "傑德",
  "jeff": "傑夫",
  "jeffrey": "傑佛瑞",
  "jenna": "珍娜",
  "jennie": "珍妮",
  "jennifer": "珍妮佛",
  "jenny": "珍妮",
  "jeremiah": "耶利米",
  "jeremy": "傑瑞米",
  "jermaine": "傑梅因",
  "jerome": "傑洛姆",
  "jerry": "傑瑞",
  "jesse": "傑西",
  "jessica": "潔西卡",
  "jessie": "潔西",
  "jesus": "赫蘇斯",
  "jill": "吉兒",
  "jillian": "吉莉安",
  "jim": "吉姆",
  "jimmie": "吉米",
  "jimmy": "吉米",
  "jo": "喬",
  "joan": "瓊",
  "joann": "喬安",
  "joanna": "喬安娜",
  "joanne": "喬安妮",
  "joaquin": "瓦金",
  "jocelyn": "喬瑟琳",
  "jodi": "茱蒂",
  "jodie": "茱蒂",
  "jody": "茱蒂",
  "joe": "喬",
  "joel": "喬爾",
  "joelle": "喬艾爾",
  "joey": "喬伊",
  "johan": "約翰",
  "johanna": "喬安娜",
  "john": "約翰",
  "johnathan": "強納森",
  "johnathon": "強納森",
  "johnnie": "強尼",
  "johnny": "強尼",
  "jolene": "喬琳",
  "jon": "喬恩",
  "jonah": "約拿",
  "jonas": "喬納斯",
  "jonathan": "強納森",
  "jonathon": "強納森",
  "joni": "喬妮",
  "jordan": "喬丹",
  "jordyn": "喬丹",
  "jorge": "豪爾赫",
  "jose": "荷西",
  "josef": "約瑟夫",
  "joseph": "約瑟夫",
  "josephine": "約瑟芬",
  "josh": "喬許",
  "joshua": "約書亞",
  "josiah": "約西亞",
  "josie": "喬西",
  "joss": "喬絲",
  "joy": "喬伊",
  "joyce": "喬伊斯",
  "juan": "胡安",
  "juana": "胡安娜",
  "juanita": "胡安妮塔",
  "judah": "猶大",
  "jude": "裘德",
  "judith": "茱蒂絲",
  "judy": "茱蒂",
  "jules": "朱爾斯",
  "julia": "茱莉亞",
  "julian": "朱利安",
  "juliana": "茱莉安娜",
  "julianna": "茱莉安娜",
  "julianne": "茱莉安",
  "julie": "茱莉",
  "julien": "朱利安",
  "juliet": "茱麗葉",
  "juliette": "茱麗葉",
  "julio": "胡利奧",
  "julius": "朱利葉斯",
  "june": "瓊",
  "junior": "小約翰",
  "juniper": "朱尼珀",
  "justin": "賈斯汀",
  "justina": "賈斯汀娜",
  "justine": "賈斯汀",
  "kaden": "凱登",
  "kaeden": "凱登",
  "kaelyn": "凱琳",
  "kai": "凱",
  "kaia": "凱亞",
  "kaila": "凱拉",
  "kailey": "凱莉",
  "kailyn": "凱琳",
  "kaitlin": "凱特琳",
  "kaitlyn": "凱特琳",
  "kaleb": "迦勒",
  "kali": "卡莉",
  "kaley": "凱莉",
  "kallie": "凱莉",
  "kamryn": "卡姆林",
  "kane": "肯恩",
  "kara": "卡拉",
  "karen": "凱倫",
  "kari": "卡莉",
  "karina": "卡琳娜",
  "karissa": "卡麗莎",
  "karl": "卡爾",
  "karla": "卡拉",
  "karlee": "卡莉",
  "karley": "卡莉",
  "karli": "卡莉",
  "karlie": "卡莉",
  "karly": "卡莉",
  "karma": "卡瑪",
  "karson": "卡森",
  "karter": "卡特",
  "kasey": "凱西",
  "kassandra": "卡珊德拉",
  "kassidy": "卡西迪",
  "kate": "凱特",
  "katelyn": "凱特琳",
  "katelynn": "凱特琳",
  "katherine": "凱瑟琳",
  "kathleen": "凱薩琳",
  "kathryn": "凱瑟琳",
  "kathy": "凱西",
  "katie": "凱蒂",
  "katrina": "卡崔娜",
  "katy": "凱蒂",
  "kay": "凱",
  "kaya": "凱亞",
  "kayden": "凱登",
  "kayla": "凱拉",
  "kaylee": "凱莉",
  "kayleigh": "凱莉",
  "kaylen": "凱琳",
  "kaylie": "凱莉",
  "kaylin": "凱琳",
  "keagan": "基根",
  "keanu": "基努",
  "keaton": "基頓",
  "keegan": "基根",
  "keelan": "基蘭",
  "keeley": "基利",
  "keely": "基利",
  "keenan": "基南",
  "keira": "綺拉",
  "keith": "基斯",
  "kellan": "凱蘭",
  "kellen": "凱倫",
  "kelley": "凱莉",
  "kelli": "凱莉",
  "kellie": "凱莉",
  "kelly": "凱莉",
  "kelsey": "凱爾西",
  "kelsie": "凱爾西",
  "kelvin": "凱爾文",
  "ken": "肯",
  "kendall": "肯達爾",
  "kendra": "肯德拉",
  "kendrick": "肯德里克",
  "kennedy": "甘迺迪",
  "kenneth": "肯尼斯",
  "kenny": "肯尼",
  "kent": "肯特",
  "kenton": "肯頓",
  "kenya": "肯亞",
  "kenzie": "肯齊",
  "keri": "凱莉",
  "kerri": "凱莉",
  "kerry": "凱莉",
  "kevin": "凱文",
  "khloe": "克蘿伊",
  "kia": "琪亞",
  "kian": "基安",
  "kiana": "琪安娜",
  "kiara": "琪亞拉",
  "kiera": "綺拉",
  "kieran": "基蘭",
  "kierra": "綺拉",
  "kiersten": "克絲汀",
  "kiley": "凱莉",
  "killian": "基利安",
  "kim": "金",
  "kimber": "金伯",
  "kimberly": "金柏莉",
  "king": "金",
  "kingston": "金斯頓",
  "kinley": "金莉",
  "kinsley": "金斯利",
  "kira": "綺拉",
  "kirby": "卡比",
  "kirk": "寇克",
  "kirsten": "克絲汀",
  "kirstin": "克絲汀",
  "kitty": "凱蒂",
  "knox": "諾克斯",
  "kobe": "柯比",
  "kody": "科迪",
  "kolby": "科爾比",
  "kole": "科爾",
  "kolton": "科爾頓",
  "konnor": "康納",
  "korbin": "科賓",
  "korey": "科里",
  "kory": "科里",
  "kourtney": "寇特妮",
  "krista": "克莉絲塔",
  "kristen": "克莉絲汀",
  "kristi": "克莉絲蒂",
  "kristian": "克里斯蒂安",
  "kristie": "克莉絲蒂",
  "kristin": "克莉絲汀",
  "kristina": "克莉絲蒂娜",
  "kristine": "克莉絲汀",
  "kristopher": "克里斯多福",
  "kristy": "克莉絲蒂",
  "krystal": "克莉絲朵",
  "kurt": "寇特",
  "kurtis": "柯蒂斯",
  "kyla": "凱拉",
  "kylan": "凱蘭",
  "kyle": "凱爾",
  "kylee": "凱莉",
  "kyleigh": "凱莉",
  "kyler": "凱勒",
  "kylie": "凱莉",
  "kyra": "綺拉",
  "lacey": "蕾西",
  "lachlan": "拉克蘭",
  "lacy": "蕾西",
  "ladonna": "拉唐娜",
  "lauren": "蘿倫",
  "laurence": "勞倫斯",
  "laurie": "蘿莉",
  "lawrence": "勞倫斯",
  "lawson": "勞森",
  "layla": "蕾拉",
  "layton": "萊頓",
  "lea": "莉亞",
  "leah": "莉亞",
  "leandro": "萊安德羅",
  "leann": "莉安",
  "leanna": "莉安娜",
  "leanne": "莉安",
  "lee": "李",
  "leia": "莉亞",
  "leigh": "李",
  "leila": "蕾拉",
  "leilani": "蕾拉妮",
  "leland": "利蘭",
  "lena": "莉娜",
  "lennon": "藍儂",
  "lennox": "萊諾克斯",
  "lenny": "萊尼",
  "leon": "里昂",
  "leona": "蕾歐娜",
  "leonard": "李奧納德",
  "leonardo": "李奧納多",
  "leonel": "萊昂內爾",
  "leroy": "勒羅伊",
  "leslie": "萊斯利",
  "lesly": "萊斯利",
  "lester": "萊斯特",
  "leticia": "萊蒂西亞",
  "levi": "李維",
  "lewis": "路易斯",
  "lexi": "萊克西",
  "lexie": "萊克西",
  "liam": "連恩",
  "liana": "莉安娜",
  "libby": "莉比",
  "liberty": "莉柏蒂",
  "lidia": "莉迪亞",
  "lila": "萊拉",
  "lilah": "萊拉",
  "lilian": "莉蓮",
  "liliana": "莉莉安娜",
  "lillian": "莉蓮",
  "lillie": "莉莉",
  "lilly": "莉莉",
  "lily": "莉莉",
  "lincoln": "林肯",
  "linda": "琳達",
  "lindsay": "琳賽",
  "lindsey": "琳賽",
  "linnea": "琳妮亞",
  "lionel": "萊昂內爾",
  "lisa": "麗莎",
  "lisette": "莉賽特",
  "liv": "麗芙",
  "livia": "莉薇婭",
  "liz": "麗茲",
  "liza": "麗莎",
  "lizbeth": "莉茲貝絲",
  "lizette": "莉賽特",
  "lizzie": "麗茲",
  "lloyd": "勞埃德",
  "logan": "羅根",
  "lois": "露易絲",
  "lola": "蘿拉",
  "london": "倫敦",
  "lonnie": "朗尼",
  "lora": "蘿拉",
  "lorelei": "蘿蕾萊",
  "loren": "羅倫",
  "lorena": "蘿蕾娜",
  "lorenzo": "羅倫佐",
  "loretta": "蘿レッ塔",
  "lori": "蘿莉",
  "lorraine": "蘿琳",
  "lottie": "洛蒂",
  "lou": "盧",
  "louie": "路易",
  "louis": "路易斯",
  "louisa": "路易莎",
  "louise": "路易絲",
  "lourdes": "盧爾德",
  "lowell": "洛威爾",
  "luca": "盧卡",
  "lucas": "盧卡斯",
  "lucia": "露西亞",
  "lucian": "盧西恩",
  "luciana": "露西安娜",
  "luciano": "盧西亞諾",
  "lucille": "露西爾",
  "lucinda": "露辛達",
  "lucky": "幸運",
  "lucy": "露西",
  "luis": "路易斯",
  "luisa": "路易莎",
  "luka": "盧卡",
  "lukas": "盧卡斯",
  "luke": "路克",
  "luna": "露娜",
  "luther": "路德",
  "luz": "露絲",
  "lydia": "莉迪亞",
  "lyla": "萊拉",
  "lyle": "萊爾",
  "lynda": "琳達",
  "lynette": "琳內特",
  "lynn": "琳恩",
  "lynne": "琳恩",
  "lyric": "莉瑞克"
  "mabel": "梅寶",
  "mac": "麥克",
  "macey": "梅西",
  "maci": "梅西",
  "macie": "梅西",
  "mack": "麥克",
  "mackenzie": "麥肯齊",
  "macy": "梅西",
  "maddox": "馬多克斯",
  "madeleine": "瑪德琳",
  "madeline": "瑪德琳",
  "madelyn": "瑪德琳",
  "madelynn": "瑪德琳",
  "madison": "麥迪遜",
  "madisyn": "麥迪遜",
  "mae": "梅",
  "maeve": "梅芙",
  "magdalena": "抹大拉",
  "maggie": "瑪姬",
  "magnolia": "木蘭",
  "maia": "瑪雅",
  "maison": "梅森",
  "major": "梅傑",
  "makayla": "麥凱拉",
  "makenna": "麥肯娜",
  "makenzie": "麥肯齊",
  "malachi": "瑪拉基",
  "malcolm": "麥爾坎",
  "malia": "瑪莉亞",
  "malik": "馬利克",
  "mallory": "瑪洛莉",
  "mamie": "梅米",
  "mandy": "曼蒂",
  "manuel": "曼努埃爾",
  "manuela": "曼努埃拉",
  "mara": "瑪拉",
  "marc": "馬克",
  "marcel": "馬塞爾",
  "marcella": "馬塞拉",
  "marcelo": "馬塞洛",
  "marcia": "瑪西亞",
  "marco": "馬可",
  "marcos": "馬科斯",
  "marcus": "馬庫斯",
  "marcy": "瑪西",
  "margaret": "瑪格麗特",
  "margarita": "瑪格麗塔",
  "margie": "瑪吉",
  "margo": "瑪戈",
  "margot": "瑪格",
  "marguerite": "瑪格麗特",
  "maria": "瑪麗亞",
  "mariah": "瑪麗亞",
  "mariam": "瑪麗亞姆",
  "mariana": "瑪麗安娜",
  "marianna": "瑪麗安娜",
  "marianne": "瑪麗安",
  "maribel": "瑪麗貝爾",
  "marie": "瑪麗",
  "mariela": "瑪麗埃拉",
  "marilyn": "瑪麗蓮",
  "marina": "瑪麗娜",
  "mario": "馬里奧",
  "marion": "馬里昂",
  "marisa": "瑪麗莎",
  "marisol": "瑪麗索爾",
  "marissa": "瑪麗莎",
  "maritza": "瑪麗莎",
  "marjorie": "瑪喬麗",
  "mark": "馬克",
  "marla": "瑪拉",
  "marlene": "瑪琳",
  "marley": "瑪莉",
  "marlon": "馬龍",
  "marquis": "馬奎斯",
  "marsha": "瑪莎",
  "marshall": "馬歇爾",
  "marta": "瑪塔",
  "martha": "瑪莎",
  "martin": "馬丁",
  "martina": "瑪蒂娜",
  "marty": "馬蒂",
  "marvin": "馬文",
  "mary": "瑪麗",
  "maryam": "瑪麗亞姆",
  "maryann": "瑪麗安",
  "mason": "梅森",
  "mateo": "馬特奧",
  "mathew": "馬修",
  "matilda": "瑪蒂達",
  "matteo": "馬特奧",
  "matthew": "馬修",
  "mattie": "馬蒂",
  "maureen": "莫琳",
  "maurice": "莫里斯",
  "mauricio": "毛里西奧",
  "maverick": "馬弗里克",
  "max": "麥克斯",
  "maxim": "馬克西姆",
  "maximilian": "馬克西米利安",
  "maximus": "馬克西姆斯",
  "maxine": "瑪克辛",
  "maxwell": "麥克斯韋",
  "maya": "瑪雅",
  "mayra": "邁拉",
  "mckenna": "麥肯娜",
  "mckenzie": "麥肯齊",
  "meadow": "梅多",
  "meagan": "梅根",
  "megan": "梅根",
  "meghan": "梅根",
  "melanie": "梅蘭妮",
  "melany": "梅蘭妮",
  "melinda": "梅琳達",
  "melissa": "梅麗莎",
  "melody": "美樂蒂",
  "melvin": "梅爾文",
  "mercedes": "梅賽德斯",
  "meredith": "梅雷迪思",
  "merle": "梅爾",
  "mia": "米婭",
  "micah": "彌迦",
  "michael": "麥可",
  "michaela": "米凱拉",
  "micheal": "麥可",
  "michel": "米歇爾",
  "michele": "蜜雪兒",
  "michelle": "蜜雪兒",
  "mick": "米克",
  "mickey": "米奇",
  "miguel": "米格爾",
  "mikaela": "米凱拉",
  "mikayla": "米凱拉",
  "mike": "麥克",
  "mikel": "米克爾",
  "mila": "米拉",
  "milan": "米蘭",
  "mildred": "米爾德麗德",
  "miles": "邁爾斯",
  "miley": "麥莉",
  "millie": "米莉",
  "milo": "米洛",
  "milton": "米爾頓",
  "mina": "米娜",
  "mindy": "明迪",
  "minerva": "米涅瓦",
  "minnie": "米妮",
  "mira": "米拉",
  "miracle": "米拉克爾",
  "miranda": "米蘭達",
  "miriam": "米里亞姆",
  "misty": "米斯蒂",
  "mitch": "米奇",
  "mitchell": "米切爾",
  "mohamed": "穆罕默德",
  "mohammad": "穆罕默德",
  "mohammed": "穆罕默德",
  "moira": "莫伊拉",
  "moises": "摩西",
  "mollie": "莫莉",
  "molly": "莫莉",
  "mona": "莫娜",
  "monica": "莫妮卡",
  "monique": "莫妮克",
  "monroe": "門羅",
  "montana": "蒙大拿",
  "monte": "蒙特",
  "montgomery": "蒙哥馬利",
  "morgan": "摩根",
  "morris": "莫里斯",
  "moses": "摩西",
  "muhammad": "穆罕默德",
  "muriel": "穆里爾",
  "murphy": "墨菲",
  "murray": "默里",
  "mya": "邁亞",
  "myers": "邁爾斯",
  "myles": "邁爾斯",
  "myra": "邁拉",
  "myrna": "默娜",
  "myrtle": "默特爾",
  "nadia": "納迪亞",
  "nadine": "納丁",
  "nancy": "南希",
  "naomi": "娜歐蜜",
  "nash": "納什",
  "natalia": "娜塔莉亞",
  "natalie": "娜塔莉",
  "nataly": "娜塔莉",
  "natasha": "娜塔莎",
  "nate": "內特",
  "nathan": "內森",
  "nathanael": "拿但業",
  "nathaniel": "納撒尼爾",
  "neal": "尼爾",
  "nehemiah": "尼希米",
  "neil": "尼爾",
  "nellie": "內莉",
  "nelson": "尼爾森",
  "nevaeh": "內瓦埃",
  "nia": "妮婭",
  "nichole": "妮可",
  "nick": "尼克",
  "nickolas": "尼古拉斯",
  "nicolas": "尼古拉斯",
  "nicole": "妮可",
  "nicolette": "妮可萊特",
  "nigel": "奈傑爾",
  "nikita": "尼基塔",
  "nikki": "妮琪",
  "nikolas": "尼古拉斯",
  "nina": "妮娜",
  "noah": "諾亞",
  "noel": "諾埃爾",
  "noelle": "諾埃爾",
  "noemi": "諾埃米",
  "nolan": "諾蘭",
  "nora": "諾拉",
  "norah": "諾拉",
  "norma": "諾瑪",
  "norman": "諾曼",
  "nova": "諾娃",
  "nyla": "奈拉",
  "oakley": "奧克利",
  "octavia": "奧克塔維亞",
  "octavio": "奧克塔維奧",
  "odin": "奧丁",
  "olga": "奧爾加",
  "olive": "奧莉薇",
  "oliver": "奧利佛",
  "olivia": "奧莉維亞",
  "omar": "奧馬爾",
  "opal": "歐泊",
  "ophelia": "奧菲莉亞",
  "orion": "獵戶座",
  "orlando": "奧蘭多",
  "oscar": "奧斯卡",
  "otis": "奧蒂斯",
  "otto": "奧托",
  "owen": "歐文",
  "pablo": "巴勃羅",
  "paige": "佩姬",
  "paisley": "佩斯利",
  "palmer": "帕爾默",
  "paloma": "帕洛瑪",
  "pam": "潘",
  "pamela": "潘蜜拉",
  "paola": "保拉",
  "paris": "帕麗斯",
  "parker": "帕克",
  "pat": "帕特",
  "patricia": "派翠西亞",
  "patrick": "派翠克",
  "patsy": "帕齊",
  "patti": "帕蒂",
  "patty": "帕蒂",
  "paul": "保羅",
  "paula": "保拉",
  "paulette": "保萊特",
  "paulina": "保琳娜",
  "pauline": "保琳",
  "paxton": "帕克斯頓",
  "payton": "佩頓",
  "pearl": "珀爾",
  "pedro": "佩德羅",
  "peggy": "佩吉",
  "penelope": "潘妮洛普",
  "penny": "潘妮",
  "percy": "珀西",
  "perla": "珀拉",
  "perry": "佩里",
  "pete": "皮特",
  "peter": "彼得",
  "petra": "佩特拉",
  "peyton": "佩頓",
  "phil": "菲爾",
  "philip": "菲利普",
  "phillip": "菲利普",
  "phoebe": "菲比",
  "phoenix": "菲尼克斯",
  "phyllis": "菲利斯",
  "pierce": "皮爾斯",
  "pierre": "皮埃爾",
  "piper": "派珀",
  "polly": "波莉",
  "poppy": "波比",
  "porter": "波特",
  "portia": "波西亞",
  "preston": "普雷斯頓",
  "prince": "普林斯",
  "princess": "普林塞斯",
  "priscilla": "普莉希拉",
  "quentin": "昆汀",
  "quincy": "昆西",
  "quinn": "昆恩",
  "quinton": "昆頓",
  "rachael": "瑞秋",
  "rachel": "瑞秋",
  "rachelle": "瑞秋",
  "raegan": "雷根",
  "rafael": "拉斐爾",
  "raiden": "雷登",
  "ralph": "拉爾夫",
  "ramiro": "拉米羅",
  "ramon": "拉蒙",
  "ramona": "拉蒙娜",
  "randal": "蘭德爾",
  "randall": "蘭德爾",
  "randi": "蘭迪",
  "randy": "蘭迪",
  "raphael": "拉斐爾",
  "raquel": "拉克爾",
  "raul": "勞爾",
  "raven": "瑞文",
  "ray": "雷",
  "raymond": "雷蒙德",
  "rayna": "雷娜",
  "reagan": "雷根",
  "rebecca": "麗貝卡",
  "rebekah": "麗貝卡",
  "reece": "里斯",
  "reed": "里德",
  "reese": "里斯",
  "reggie": "雷吉",
  "regina": "雷吉娜",
  "reginald": "雷金納德",
  "reid": "里德",
  "reina": "蕾娜",
  "remington": "雷明頓",
  "remy": "雷米",
  "renata": "雷納塔",
  "rene": "勒內",
  "renee": "蕾妮",
  "reuben": "魯本",
  "rex": "雷克斯",
  "rey": "雷",
  "reyna": "蕾娜",
  "rhett": "瑞特",
  "rhonda": "朗達",
  "ricardo": "里卡多",
  "richard": "理查",
  "richie": "里奇",
  "rick": "瑞克",
  "rickey": "瑞奇",
  "ricky": "瑞奇",
  "rider": "萊德",
  "riley": "萊利",
  "rita": "麗塔",
  "river": "里弗",
  "robbie": "羅比",
  "robert": "羅伯特",
  "roberta": "羅伯塔",
  "roberto": "羅伯托",
  "robin": "羅賓",
  "robyn": "羅賓",
  "rocco": "羅科",
  "rochelle": "羅謝爾",
  "rocky": "洛基",
  "rod": "羅德",
  "roderick": "羅德里克",
  "rodney": "羅德尼",
  "rodolfo": "羅道夫",
  "rodrigo": "羅德里戈",
  "rogelio": "羅赫略",
  "roger": "羅傑",
  "rohan": "羅漢",
  "roland": "羅蘭",
  "rolando": "羅蘭多",
  "roman": "羅曼",
  "romeo": "羅密歐",
  "ron": "羅恩",
  "ronald": "羅納德",
  "ronnie": "羅尼",
  "rory": "羅里",
  "rosa": "羅莎",
  "rosalie": "羅莎莉",
  "rosalind": "羅莎琳德",
  "rosalinda": "羅莎琳達",
  "rosalyn": "羅莎琳",
  "rose": "蘿絲",
  "rosemarie": "蘿絲瑪麗",
  "rosemary": "蘿絲瑪麗",
  "rosie": "蘿西",
  "ross": "羅斯",
  "rowan": "羅文",
  "rowena": "羅威娜",
  "roxanne": "羅克珊",
  "roy": "羅伊",
  "royal": "羅亞爾",
  "royce": "羅伊斯",
  "ruben": "魯本",
  "ruby": "露比",
  "rudolph": "魯道夫",
  "rudy": "魯迪",
  "rufus": "魯弗斯",
  "russell": "羅素",
  "ruth": "露絲",
  "ryan": "萊恩",
  "ryder": "萊德",
  "ryker": "萊克",
  "rylan": "萊蘭",
  "rylee": "萊莉",
  "ryleigh": "萊莉"
  "sabrina": "莎賓娜",
  "sadie": "莎蒂",
  "sage": "賽奇",
  "saige": "賽奇",
  "sal": "薩爾",
  "salma": "莎瑪",
  "salvador": "薩爾瓦多",
  "salvatore": "薩爾瓦多",
  "sam": "山姆",
  "samantha": "莎曼珊",
  "samara": "薩馬拉",
  "samir": "薩米爾",
  "samira": "薩米拉",
  "sammie": "薩米",
  "sammy": "薩米",
  "samuel": "山繆",
  "sandra": "珊卓",
  "sandy": "珊蒂",
  "santiago": "聖地亞哥",
  "santos": "桑托斯",
  "sara": "莎拉",
  "sarah": "莎拉",
  "sarai": "撒萊",
  "sariah": "薩賴亞",
  "sasha": "莎夏",
  "saul": "索爾",
  "savanna": "薩凡納",
  "savannah": "薩凡納",
  "sawyer": "索耶",
  "scott": "史考特",
  "scottie": "史考蒂",
  "scotty": "史考蒂",
  "sean": "尚恩",
  "sebastian": "賽巴斯蒂安",
  "selena": "席琳娜",
  "selina": "席琳娜",
  "serena": "瑟琳娜",
  "serenity": "瑟蕾妮蒂",
  "sergio": "塞爾吉奧",
  "seth": "賽斯",
  "shane": "尚恩",
  "shania": "仙妮亞",
  "shannon": "雪儂",
  "shari": "雪莉",
  "sharon": "雪倫",
  "shaun": "尚恩",
  "shauna": "尚娜",
  "shawn": "尚恩",
  "shawna": "尚娜",
  "shay": "謝伊",
  "shayla": "謝拉",
  "shea": "謝伊",
  "sheena": "希娜",
  "sheila": "希拉",
  "shelby": "雪比",
  "sheldon": "謝爾頓",
  "shelley": "雪莉",
  "shelly": "雪莉",
  "sherri": "雪莉",
  "sherrie": "雪莉",
  "sherry": "雪莉",
  "sheryl": "雪莉爾",
  "shiloh": "夏洛",
  "shirley": "雪莉",
  "sidney": "雪梨",
  "sienna": "席安娜",
  "sierra": "席拉",
  "silas": "塞拉斯",
  "silvia": "希爾維亞",
  "simon": "賽門",
  "simone": "西蒙妮",
  "skye": "絲凱",
  "skylar": "史凱勒",
  "skyler": "史凱勒",
  "sloane": "斯隆",
  "sofia": "蘇菲亞",
  "solomon": "所羅門",
  "sonia": "索尼婭",
  "sonny": "桑尼",
  "sonya": "索尼婭",
  "sophia": "蘇菲亞",
  "sophie": "蘇菲",
  "spencer": "史賓塞",
  "stacey": "史黛西",
  "staci": "史黛西",
  "stacie": "史黛西",
  "stacy": "史黛西",
  "stan": "史丹",
  "stanley": "史丹利",
  "star": "史塔",
  "stella": "史黛拉",
  "stephan": "史蒂芬",
  "stephanie": "史蒂芬妮",
  "stephen": "史蒂芬",
  "sterling": "史特林",
  "steve": "史蒂夫",
  "steven": "史蒂文",
  "stevie": "史蒂薇",
  "stewart": "史都華",
  "stone": "史東",
  "storm": "史東",
  "stuart": "史都華",
  "sue": "蘇",
  "sullivan": "蘇利文",
  "summer": "夏沫",
  "sunny": "桑妮",
  "susan": "蘇珊",
  "susana": "蘇珊娜",
  "susanna": "蘇珊娜",
  "susanne": "蘇珊",
  "susie": "蘇西",
  "suzanne": "蘇珊",
  "suzette": "蘇澤特",
  "suzy": "蘇西",
  "sybil": "西比爾",
  "sydney": "雪梨",
  "sylvester": "席維斯",
  "sylvia": "希爾維亞",
  "tabitha": "塔比莎",
  "talia": "塔莉亞",
  "tamara": "塔瑪拉",
  "tameka": "塔梅卡",
  "tami": "塔米",
  "tammy": "塔米",
  "tania": "坦妮婭",
  "tanner": "坦納",
  "tanya": "坦妮婭",
  "tara": "泰拉",
  "tasha": "塔莎",
  "tate": "泰特",
  "tatiana": "塔蒂亞娜",
  "tatum": "泰頓",
  "taylor": "泰勒",
  "ted": "泰德",
  "teddy": "泰迪",
  "tegan": "蒂根",
  "teresa": "泰瑞莎",
  "teri": "泰瑞",
  "terra": "泰拉",
  "terrance": "泰倫斯",
  "terrell": "泰瑞爾",
  "terrence": "泰倫斯",
  "terri": "泰瑞",
  "tessa": "泰莎",
  "thaddeus": "賽迪斯",
  "thea": "西婭",
  "thelma": "塞爾瑪",
  "theo": "西奧",
  "theodore": "西奧多",
  "theresa": "泰瑞莎",
  "therese": "泰瑞絲",
  "thomas": "湯瑪斯",
  "tia": "蒂亞",
  "tiana": "蒂安娜",
  "tiara": "蒂阿拉",
  "tiffany": "蒂芬妮",
  "tim": "提姆",
  "timmy": "提米",
  "timothy": "提摩西",
  "tina": "蒂娜",
  "titus": "提圖斯",
  "tobias": "托拜厄斯",
  "toby": "托比",
  "todd": "陶德",
  "tom": "湯姆",
  "tomas": "托馬斯",
  "tommy": "湯米",
  "toni": "托妮",
  "tonia": "托妮亞",
  "tony": "東尼",
  "tonya": "托妮亞",
  "tori": "托莉",
  "tracey": "崔西",
  "traci": "崔西",
  "tracie": "崔西",
  "tracy": "崔西",
  "travis": "崔維斯",
  "trent": "特倫特",
  "trenton": "特倫頓",
  "trevor": "崔佛",
  "trey": "特雷",
  "tricia": "翠西亞",
  "trina": "翠娜",
  "trinity": "崔妮蒂",
  "trisha": "翠莎",
  "tristan": "崔斯坦",
  "tristen": "崔斯坦",
  "tristin": "崔斯坦",
  "troy": "特洛伊",
  "trudy": "特魯迪",
  "tucker": "塔克",
  "ty": "泰",
  "tyler": "泰勒",
  "tyrone": "泰隆",
  "tyson": "泰森",
  "ulises": "尤利西斯",
  "ulysses": "尤利西斯",
  "uma": "烏瑪",
  "uri": "烏里",
  "uriah": "烏利亞",
  "uriel": "烏列爾",
  "ursula": "烏蘇拉",
  "valentina": "瓦倫蒂娜",
  "valentine": "瓦倫丁",
  "valeria": "瓦萊麗亞",
  "valerie": "瓦萊麗",
  "van": "范",
  "vance": "萬斯",
  "vanessa": "凡妮莎",
  "vaughn": "沃恩",
  "vera": "薇拉",
  "verna": "維爾納",
  "vernon": "弗農",
  "veronica": "薇若妮卡",
  "vic": "維克",
  "vicki": "維琪",
  "vickie": "維琪",
  "vicky": "維琪",
  "victor": "維克多",
  "victoria": "維多利亞",
  "vienna": "維也納",
  "vincent": "文森",
  "vinnie": "文尼",
  "viola": "維奧拉",
  "violet": "紫羅蘭",
  "virginia": "維吉尼亞",
  "vivian": "薇薇安",
  "viviana": "薇薇安娜",
  "vivienne": "薇薇安",
  "wade": "韋德",
  "walker": "沃克",
  "wallace": "華萊士",
  "wally": "沃利",
  "walter": "華特",
  "wanda": "旺達",
  "warren": "華倫",
  "waylon": "韋倫",
  "wayne": "韋恩",
  "wendell": "溫德爾",
  "wendy": "溫蒂",
  "wes": "韋斯",
  "wesley": "衛斯理",
  "weston": "韋斯頓",
  "whitney": "惠特妮",
  "wilbur": "威爾伯",
  "wiley": "威利",
  "wilfred": "威爾弗雷德",
  "will": "威爾",
  "willa": "薇拉",
  "willard": "威拉德",
  "william": "威廉",
  "willie": "威利",
  "willis": "威利斯",
  "willow": "薇洛",
  "willy": "威利",
  "wilma": "威爾瑪",
  "wilson": "威爾遜",
  "winnie": "溫妮",
  "winston": "溫斯頓",
  "winter": "溫特",
  "wyatt": "懷亞特",
  "xander": "贊德",
  "xavier": "賽維爾",
  "ximena": "希梅娜",
  "yadira": "亞迪拉",
  "yahir": "亞希爾",
  "yasmin": "亞斯敏",
  "yasmine": "亞斯敏",
  "yesenia": "葉塞尼婭",
  "yolanda": "尤蘭達",
  "yusuf": "優素福",
  "yvette": "伊薇特",
  "yvonne": "伊馮娜",
  "zach": "札克",
  "zachariah": "撒迦利亞",
  "zachary": "札克瑞",
  "zachery": "札克瑞",
  "zack": "札克",
  "zackary": "札克瑞",
  "zaiden": "贊登",
  "zain": "贊恩",
  "zander": "贊德",
  "zane": "贊恩",
  "zara": "札拉",
  "zayden": "贊登",
  "zeke": "齊克",
  "zelda": "薩爾達",
  "zion": "錫安",
  "zoe": "柔伊",
  "zoey": "柔伊",
  "zola": "佐拉",
  "zoie": "柔伊"
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
