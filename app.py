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
  "aaron": "艾倫",
"abel": "亞伯",
"abraham": "亞伯拉罕",
"adam": "亞當",
"adrian": "亞德里恩",
"alan": "艾倫",
"albert": "亞伯特",
"alexander": "亞歷山大",
"alfred": "亞佛烈德",
"allen": "艾倫",
"alvin": "阿爾文",
"andrew": "安德魯",
"andy": "安迪",
"angus": "安格斯",
"anthony": "安東尼",
"archie": "阿奇",
"arnold": "阿諾德",
"arthur": "亞瑟",
"austin": "奧斯汀",
"baldwin": "鮑德溫",
"barry": "巴里",
"benjamin": "班傑明",
"bernard": "伯納德",
"bill": "比爾",
"billy": "比利",
"bob": "鮑伯",
"bradley": "布萊德利",
"brandon": "布蘭登",
"brian": "布萊恩",
"bruce": "布魯斯",
"bryan": "布萊恩",
"byron": "拜倫",
"calvin": "喀爾文",
"carl": "卡爾",
"cecil": "塞西爾",
"charles": "查爾斯",
"chris": "克里斯",
"christian": "克里斯汀",
"christopher": "克里斯多福",
"clarence": "克拉倫斯",
"clark": "克拉克",
"claude": "克勞德",
"clifford": "柯利弗德",
"clyde": "克萊德",
"colin": "科林",
"conrad": "康拉德",
"craig": "克雷格",
"curtis": "柯蒂斯",
"cyril": "西瑞爾",
"daniel": "丹尼爾",
"darren": "達倫",
"david": "大衛",
"dean": "迪恩",
"dennis": "丹尼斯",
"derek": "戴瑞克",
"dick": "狄克",
"dominic": "多明尼克",
"don": "唐",
"donald": "唐納德",
"douglas": "道格拉斯",
"duncan": "鄧肯",
"dylan": "狄倫",
"earl": "厄爾",
"edgar": "愛德格",
"edmund": "艾德蒙",
"edward": "愛德華",
"edwin": "愛德溫",
"eli": "伊萊",
"elliot": "艾略特",
"ellis": "艾利斯",
"elvis": "艾維斯",
"eric": "艾瑞克",
"ernest": "恩斯特",
"eugene": "尤金",
"evan": "伊凡",
"felix": "菲力克斯",
"francis": "法蘭西斯",
"frank": "法蘭克",
"franklin": "富蘭克林",
"fred": "弗雷德",
"gabriel": "加百列",
"gary": "蓋瑞",
"gavin": "蓋文",
"geoffrey": "傑佛瑞",
"george": "喬治",
"gerald": "傑拉德",
"gilbert": "吉伯特",
"glen": "葛倫",
"gordon": "高登",
"graham": "格雷姆",
"gregory": "葛列格里",
"guy": "蓋伊",
"harold": "哈羅德",
"harrison": "哈里遜",
"harry": "哈利",
"harvey": "哈維",
"hector": "赫克特",
"henry": "亨利",
"herbert": "赫伯特",
"herman": "赫爾曼",
"howard": "霍華德",
"hugh": "休",
"ian": "伊恩",
"isaac": "艾薩克",
"ivan": "伊凡",
"jack": "傑克",
"jackson": "傑克森",
"jacob": "雅各",
"james": "詹姆斯",
"jason": "傑森",
"jeff": "傑夫",
"jeffrey": "傑佛瑞",
"jeremy": "傑瑞米",
"jerome": "傑羅姆",
"jerry": "傑瑞",
"jim": "吉姆",
"jimmy": "吉米",
"joe": "喬",
"john": "約翰",
"johnny": "強尼",
"jonathan": "強納森",
"joseph": "約瑟夫",
"joshua": "約書亞",
"justin": "賈斯汀",
"keith": "基斯",
"ken": "肯",
"kenneth": "肯尼斯",
"kevin": "凱文",
"larry": "賴瑞",
"leo": "里奧",
"leonard": "李奧納德",
"lewis": "路易斯",
"louis": "路易斯",
"luke": "路克",
"malcolm": "麥爾坎",
"marcus": "馬可斯",
"mark": "馬克",
"martin": "馬丁",
"matthew": "馬修",
"maurice": "莫里斯",
"michael": "麥可",
"miles": "邁爾斯",
"milo": "麥洛",
"nathan": "南森",
"neil": "尼爾",
"nicholas": "尼可拉斯",
"nick": "尼克",
"nigel": "奈吉爾",
"norman": "諾曼",
"oliver": "奧利佛",
"oscar": "奧斯卡",
"patrick": "派翠克",
"paul": "保羅",
"peter": "彼得",
"philip": "菲利普",
"ralph": "雷夫",
"randy": "藍迪",
"raymond": "雷蒙",
"richard": "理查",
"robert": "羅伯特",
"robin": "羅賓",
"roger": "羅傑",
"ronald": "羅納德",
"roy": "羅伊",
"ryan": "萊恩",
"sam": "山姆",
"samuel": "撒母耳",
"scott": "史考特",
"sean": "西恩",
"simon": "賽門",
"spencer": "史賓塞",
"stanley": "史丹利",
"stephen": "史蒂芬",
"steven": "史蒂文",
"stuart": "史都華",
"ted": "泰德",
"thomas": "湯瑪斯",
"tim": "提姆",
"timothy": "提摩太",
"tom": "湯姆",
"tony": "東尼",
"travis": "崔維斯",
"tyler": "泰勒",
"victor": "維克多",
"vincent": "文森",
"walter": "華特",
"warren": "華倫",
"wayne": "韋恩",
"william": "威廉",
"willie": "威利",
"zachary": "柴克瑞",
"abner": "阿布納",
"adolph": "阿道夫",
"aiden": "艾登",
"alastair": "阿拉斯泰爾",
"alfie": "阿飛",
"amos": "阿摩斯",
"andre": "安德烈",
"angelo": "安傑洛",
"ansel": "安塞爾",
"anton": "安東",
"ariel": "艾瑞爾",
"asher": "亞舍",
"atticus": "阿提克斯",
"aubrey": "奧布里",
"august": "奧古斯特",
"axel": "阿克塞爾",
"barnaby": "巴納比",
"bartholomew": "巴多羅買",
"basil": "巴澤爾",
"benedict": "本尼迪克",
"bertram": "伯特倫",
"blake": "布雷克",
"boris": "鮑里斯",
"boyd": "波伊德",
"brent": "布倫特",
"brett": "布雷特",
"bruno": "布魯諾",
"bryce": "布萊斯",
"caleb": "迦勒",
"callum": "卡倫",
"cameron": "卡麥隆",
"carey": "凱里",
"carlton": "卡爾頓",
"casey": "凱西",
"chad": "查德",
"chandler": "錢德勒",
"charlie": "查理",
"chester": "切斯特",
"clayton": "克萊頓",
"clint": "克林特",
"cody": "科迪",
"cole": "科爾",
"corey": "柯瑞",
"cornelius": "科尼利厄斯",
"curt": "柯特",
"dale": "戴爾",
"dallas": "達拉斯",
"damon": "戴蒙",
"dane": "戴恩",
"dante": "但丁",
"darius": "達瑞斯",
"darryl": "達里爾",
"dave": "戴夫",
"dawson": "道森",
"declan": "戴克蘭",
"dexter": "德克斯特",
"dominick": "多明尼克",
"donnie": "唐尼",
"dorian": "多里安",
"dustin": "達斯汀",
"chloe": "克蘿伊",
"riva": "莉娃",
"birdy": "柏蒂",
"mary": "瑪麗",
"emily": "艾蜜莉",
"sophia": "蘇菲亞",
"olivia": "奧莉薇亞",
"grace": "葛蕾絲",
"anna": "安娜",
"abigail": "艾比蓋爾",
"ada": "艾達",
"adela": "愛黛拉",
"adeline": "愛黛琳",
"agnes": "愛格妮絲",
"alice": "愛麗絲",
"alicia": "艾莉西亞",
"alison": "艾莉森",
"amanda": "阿曼達",
"amber": "安珀",
"amelia": "艾蜜莉亞",
"amy": "艾美",
"andrea": "安德莉亞",
"angela": "安琪拉",
"angelina": "安潔莉娜",
"anita": "安妮塔",
"ann": "安",
"anne": "安妮",
"annie": "安妮",
"april": "艾波",
"ashley": "艾希莉",
"audrey": "奧黛莉",
"aurora": "奧蘿拉",
"ava": "艾娃",
"barbara": "芭芭拉",
"beatrice": "碧翠絲",
"bella": "貝拉",
"bernice": "柏妮絲",
"bertha": "柏莎",
"beth": "貝絲",
"betty": "貝蒂",
"beverly": "貝芙莉",
"bonnie": "邦妮",
"brenda": "布蘭達",
"brooke": "布魯克",
"camila": "卡蜜拉",
"caroline": "卡洛琳",
"catherine": "凱瑟琳",
"cathy": "凱西",
"cecilia": "塞西莉亞",
"charlotte": "夏綠蒂",
"cheryl": "雪柔",
"christina": "克莉絲汀娜",
"christine": "克莉絲汀",
"cindy": "辛蒂",
"claire": "克萊兒",
"clara": "克拉拉",
"claudia": "克勞蒂亞",
"constance": "康斯坦絲",
"cora": "柯拉",
"crystal": "克莉絲朵",
"cynthia": "辛西亞",
"daisy": "黛西",
"dana": "黛娜",
"danielle": "丹妮爾",
"daphne": "黛芙妮",
"dawn": "唐恩",
"deborah": "黛博拉",
"diana": "黛安娜",
"donna": "唐娜",
"doris": "桃樂絲",
"dorothy": "桃樂蒂",
"edith": "伊迪絲",
"elaine": "伊蓮",
"eleanor": "艾蓮娜",
"elizabeth": "伊莉莎白",
"ella": "艾拉",
"ellen": "艾倫",
"eloise": "愛洛伊絲",
"emma": "艾瑪",
"erica": "艾瑞卡",
"esther": "以斯帖",
"ethel": "艾瑟兒",
"eva": "伊娃",
"evelyn": "伊芙琳",
"faith": "費絲",
"fiona": "費歐娜",
"florence": "佛羅倫絲",
"frances": "法蘭西絲",
"gabrielle": "嘉布莉兒",
"gail": "蓋兒",
"gemma": "潔瑪",
"georgia": "喬治亞",
"gloria": "葛洛莉亞",
"hannah": "漢娜",
"harriet": "哈莉特",
"hazel": "海柔爾",
"heather": "赫瑟爾",
"helen": "海倫",
"holly": "荷莉",
"irene": "艾琳",
"iris": "艾瑞絲",
"isabel": "伊莎貝爾",
"isabella": "伊莎貝拉",
"ivy": "艾薇",
"jacqueline": "賈桂琳",
"jane": "珍",
"janet": "珍妮特",
"jasmine": "茉莉",
"jean": "琴",
"jennifer": "珍妮佛",
"jessica": "潔西卡",
"jill": "吉兒",
"joan": "瓊",
"joanna": "喬安娜",
"josephine": "約瑟芬",
"joy": "喬伊",
"joyce": "喬伊斯",
"judith": "茱蒂絲",
"judy": "茱蒂",
"julia": "茱莉亞",
"julie": "茱莉",
"karen": "凱倫",
"katherine": "凱瑟琳",
"kathleen": "凱瑟琳",
"kathryn": "凱瑟琳",
"kathy": "凱西",
"katie": "凱蒂",
"kayla": "凱拉",
"kelly": "凱莉",
"kimberly": "金柏莉",
"kirsten": "克絲汀",
"laura": "蘿拉",
"lauren": "蘿倫",
"leah": "莉亞",
"lena": "莉娜",
"leslie": "萊斯莉",
"lily": "莉莉",
"linda": "琳達",
"lisa": "麗莎",
"lillian": "莉莉安",
"lois": "露易絲",
"lorraine": "洛琳",
"louise": "露易絲",
"lucy": "露西",
"lydia": "莉蒂亞",
"lynn": "琳恩",
"mabel": "梅寶",
"madeline": "瑪德琳",
"madison": "麥迪遜",
"maggie": "瑪姬",
"mandy": "曼蒂",
"margaret": "瑪格麗特",
"maria": "瑪麗亞",
"marian": "瑪麗安",
"marie": "瑪麗",
"marilyn": "瑪麗蓮",
"martha": "瑪莎",
"maureen": "莫琳",
"megan": "梅根",
"melanie": "梅蘭妮",
"melissa": "梅麗莎",
"mia": "米婭",
"michelle": "蜜雪兒",
"mildred": "米爾德麗",
"molly": "莫莉",
"monica": "莫妮卡",
"nancy": "南西",
"naomi": "娜歐蜜",
"natalie": "娜塔莉",
"nicole": "妮可",
"nina": "妮娜",
"nora": "諾拉",
"pamela": "潘蜜拉",
"patricia": "派翠西亞",
"paula": "寶拉",
"pearl": "珮兒",
"peggy": "佩姬",
"penelope": "潘妮洛普",
"phoebe": "菲比",
"phyllis": "菲莉絲",
"rachel": "瑞秋",
"rebecca": "蕾貝卡",
"regina": "蕾吉娜",
"rita": "麗塔",
"roberta": "羅貝塔",
"rose": "蘿絲",
"rosemary": "蘿絲瑪麗",
"ruby": "露比",
"ruth": "露絲",
"sabrina": "莎柏琳娜",
"sally": "莎莉",
"samantha": "莎曼珊",
"sandra": "珊卓拉",
"sara": "莎拉",
"sarah": "莎拉",
"sharon": "雪倫",
"sheila": "希拉",
"shirley": "雪莉",
"sophie": "蘇菲",
"stella": "史黛拉",
"stephanie": "史蒂芬妮",
"susan": "蘇珊",
"suzanne": "蘇珊娜",
"sylvia": "希薇亞",
"tammy": "泰咪",
"tanya": "坦雅",
"teresa": "泰瑞莎",
"tiffany": "蒂芬妮",
"tina": "蒂娜",
"tracy": "翠西",
"valerie": "瓦萊麗",
"vanessa": "凡妮莎",
"veronica": "薇若妮卡",
"victoria": "維多利亞",
"violet": "薇奧麗",
"virginia": "維吉尼亞",
"vivian": "薇薇安",
"wendy": "溫蒂",
"yvonne": "伊芳",
"zoe": "柔伊",
"adriana": "艾德里安娜",
"aileen": "艾琳",
"aimee": "艾梅",
"alana": "阿拉娜",
"alexa": "艾莉克莎",
"alexandra": "亞莉珊卓",
"alexis": "艾莉克絲",
"alma": "艾爾瑪",
"amara": "阿瑪拉",
"anastasia": "安娜塔西亞",
"angel": "安琪",
"annette": "安妮特",
"antonia": "安東妮亞",
"ariana": "亞莉安娜",
"arlene": "阿琳",
"astrid": "阿斯特麗",
"avery": "艾佛莉",
"bianca": "碧昂卡",
"bridget": "布莉姬",
"caitlin": "凱特琳",
"camille": "卡蜜兒",
"candace": "坎蒂絲",
"cara": "卡拉",
"carmen": "卡門",
"carol": "卡蘿",
"cassandra": "卡珊卓",
"cecelia": "塞西莉亞",
"celeste": "瑟蕾絲特",
"charmaine": "夏曼",
"chelsea": "雀兒喜",
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
