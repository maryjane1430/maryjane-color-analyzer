import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import webcolors
import pandas as pd

# ========== Styling Part ===========
st.set_page_config(
    page_title="Maryjane Color Analyzer from Image",
    layout="wide",
    initial_sidebar_state="collapsed"
)

FONT_URL = "https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Prompt:wght@500;700&display=swap"
ACCENT_COLOR = "#28a7a0"
CARD_BG = "#ecf4ff"
TITLE = "Maryjane Color Analyzer from Image"

st.markdown(
    f"""
    <style>
    @import url('{FONT_URL}');
    html, body, [data-testid="stApp"]  {{
        background: linear-gradient(135deg, {CARD_BG} 20%, #c7fcff 100%);
        color: #303030;
        font-family: 'Prompt', 'Kanit', 'Orbitron', 'Arial', sans-serif;
        letter-spacing: 0.01em;
    }}
    h1, .main-title {{
        font-family: 'Orbitron', 'Prompt', sans-serif;
        font-weight: 700 !important;
        color: {ACCENT_COLOR};
        letter-spacing: .15em;
        font-size:2.7em;
        text-align:center;
    }}
    .subtitle-mj {{
        display:block;
        background: linear-gradient(107deg, #ffecd2 0%, #fcb69f 100%);
        color:#222022;font-size:1.23em;border-radius:1em;
        padding:.9em 1.6em .7em 1.6em; margin-bottom:1.6em; margin-top:0.4em;
        box-shadow: 0 6px 18px rgba(23,45,56,0.09);
    }}
    .circle-color {{
        width:34px; height:34px; border-radius:50%; display:inline-block;
        margin:0 0.2em 0 0; border:2px solid #fff;
        box-shadow: 1.5px 2.5px 4px rgba(23,56,135,0.05);
        vertical-align:middle;
    }}
    .bigcircle-color {{
        width:56px;height:56px;border-radius:50%;display:inline-block;
        border:3px solid #222;margin:.2em 2em .2em .2em;
        box-shadow:0 3px 14px #8882;
    }}
    .mj-card  {{
        background:#fff;
        border-radius: 13px;
        box-shadow:0 6px 22px #839bab18, 0 1px 0 #eee;
        margin: 0 0 1.2em 0;
        padding:1.6em 2.2em 1.4em 2.2em;
        font-size:1.07em;
    }}
    hr {{
        margin:2.6em 0 .9em 0;
        border: none;
        border-bottom: 2px solid {ACCENT_COLOR};
        opacity:.18;
    }}
    label, .stSlider, .stNumberInput>div>label,
    .stDataFrame, .stMarkdown>p {{
        font-family:'Prompt',sans-serif;
        font-weight:500;
        font-size:1.09em !important;
        color:#232321 !important;
    }}
    td, th {{
        padding:.6em 1.1em !important;
        font-family:'Prompt',sans-serif;
        font-size:1.08em;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(f"<h1 class='main-title'>{TITLE}</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='subtitle-mj'>
    คนดีของพี่ แยกสีให้สนุก💋 อัปโหลดภาพแล้วระบบจะวิเคราะห์ทุกสีหลักในภาพ
    พร้อมชื่อสี รหัส RGB/HEX และตัวอย่าง
    </div>
    """,
    unsafe_allow_html=True
)

# ============== Function Section ==============
def rgb_to_name(rgb_triplet):
    try:
        return webcolors.rgb_to_name(rgb_triplet)
    except ValueError:
        min_colors = {}
        for name in webcolors.CSS3_NAMES_TO_HEX:
            hex_code = webcolors.name_to_hex(name)
            r, g, b = webcolors.hex_to_rgb(hex_code)
            min_colors[(r-rgb_triplet[0])**2 +
                       (g-rgb_triplet[1])**2 +
                       (b-rgb_triplet[2])**2] = name
        return min_colors[min(min_colors.keys())]

def rgb_to_hex(rgb_triplet):
    return '#{:02x}{:02x}{:02x}'.format(*rgb_triplet)

# =============== Web Main ===============
uploaded_file = st.file_uploader(
    "อัปโหลดไฟล์ภาพ (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="ตัวอย่างภาพที่อัปโหลด", use_container_width=True)

    img_np = np.array(image)
    img_flat = img_np.reshape((-1, 3))

    st.markdown("<hr>", unsafe_allow_html=True)
    n_colors = st.slider("How many dominant colors to analyze?", 2, 40, 8)

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=8)
    labels = kmeans.fit_predict(img_flat)
    counts = Counter(labels)

    total = sum(counts.values())
    color_out = []

    for idx, count in counts.items():
        perc = count * 100 / total
        rgb = tuple(np.round(kmeans.cluster_centers_[idx]).astype(int))
        color_out.append((rgb_to_name(rgb), rgb, rgb_to_hex(rgb), perc))

    color_out.sort(key=lambda x: x[3], reverse=True)

    st.markdown("### เฉดสีหลักในภาพ")
    df = pd.DataFrame([
        {
            "Color": f"<span class='circle-color' style='background:{hex_col}'></span>",
            "Name": name.title(),
            "RGB": str(rgb),
            "HEX": hex_col.upper(),
            "Proportion (%)": f"{perc:.2f}"
        }
        for name, rgb, hex_col, perc in color_out
    ])

    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        [p for _, _, _, p in color_out],
        colors=[h for _, _, h, _ in color_out],
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Pixel Color Picker")

    h, w, _ = img_np.shape
    x = st.number_input("X", 0, w-1, w//2)
    y = st.number_input("Y", 0, h-1, h//2)

    rgb_at = tuple(img_np[y, x])
    st.markdown(
        f"""
        <div class='mj-card' style='display:flex;align-items:center;'>
            <span class='bigcircle-color'
                style='background:{rgb_to_hex(rgb_at)}'></span>
            <div>
                <b>RGB:</b> {rgb_at}<br>
                <b>HEX:</b> {rgb_to_hex(rgb_at).upper()}<br>
                <b>Name:</b> {rgb_to_name(rgb_at).title()}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("โปรดอัปโหลดไฟล์ภาพเพื่อเริ่มต้นใช้งาน")