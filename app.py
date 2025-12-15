import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import webcolors
import pandas as pd

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Maryjane Color Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= THEME =================
FONT_URL = "https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Prompt:wght@300;500;700&display=swap"

BLACK = "#0b0b0c"
DARK = "#121214"
RED = "#8b0000"
GOLD = "#d4af37"
TEXT = "#e6e6e6"

TITLE = "MARYJANE COLOR ANALYZER"

st.markdown(
    f"""
    <style>
    @import url('{FONT_URL}');

    html, body, [data-testid="stApp"] {{
        background: radial-gradient(circle at top, #1a1a1d 0%, {BLACK} 65%);
        color: {TEXT};
        font-family: 'Prompt', sans-serif;
    }}

    h1 {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 800;
        letter-spacing: .28em;
        text-align: center;
        color: {GOLD};
        margin-bottom: .4em;
    }}

    .subtitle-mj {{
        text-align: center;
        font-size: 1.05em;
        color: #f1dca7;
        background: linear-gradient(135deg, #3a0000, {RED});
        padding: 1em 2em;
        border-radius: 14px;
        box-shadow: 0 8px 28px rgba(0,0,0,.45);
        margin-bottom: 2em;
    }}

    .mj-card {{
        background: linear-gradient(145deg, #161618, #0e0e10);
        border-radius: 16px;
        padding: 1.6em 2em;
        border: 1px solid rgba(212,175,55,.25);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,.02),
                    0 10px 30px rgba(0,0,0,.55);
        margin-bottom: 1.4em;
    }}

    hr {{
        border: none;
        border-top: 1px solid rgba(212,175,55,.35);
        margin: 2.6em 0;
    }}

    label {{
        color: {GOLD} !important;
        font-weight: 500;
    }}

    .circle-color {{
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: inline-block;
        border: 2px solid {GOLD};
        box-shadow: 0 0 10px rgba(0,0,0,.6);
    }}

    .bigcircle-color {{
        width: 64px;
        height: 64px;
        border-radius: 50%;
        border: 3px solid {GOLD};
        box-shadow: 0 0 18px rgba(212,175,55,.45);
        margin-right: 1.5em;
    }}

    table {{
        color: {TEXT};
        border-collapse: collapse;
        width: 100%;
    }}

    th {{
        background: #1c1c1f;
        color: {GOLD};
        letter-spacing: .08em;
        border-bottom: 1px solid rgba(212,175,55,.4);
    }}

    td {{
        background: #121214;
        border-bottom: 1px solid rgba(255,255,255,.05);
    }}

    code {{
        color: {GOLD};
        background: #1a1a1d;
        padding: .2em .5em;
        border-radius: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ================= HEADER =================
st.markdown(f"<h1>{TITLE}</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='subtitle-mj'>
    Luxury Color Intelligence ✦ วิเคราะห์เฉดสีจากภาพ พร้อมชื่อสี RGB / HEX ระดับพรีเมียม
    </div>
    """,
    unsafe_allow_html=True
)

# ================= FUNCTIONS =================
def rgb_to_name(rgb_triplet):
    try:
        return webcolors.rgb_to_name(rgb_triplet)
    except ValueError:
        min_colors = {}
        for name in webcolors.CSS3_NAMES_TO_HEX:
            hex_code = webcolors.name_to_hex(name)
            r, g, b = webcolors.hex_to_rgb(hex_code)
            min_colors[(r-rgb_triplet[0])**2 + (g-rgb_triplet[1])**2 + (b-rgb_triplet[2])**2] = name
        return min_colors[min(min_colors.keys())]

def rgb_to_hex(rgb_triplet):
    return '#{:02x}{:02x}{:02x}'.format(*rgb_triplet)

# ================= MAIN =================
uploaded_file = st.file_uploader("Upload image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_flat = img_np.reshape((-1, 3))

    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 1. Select number of dominant colors")

    n_colors = st.slider("Number of colors", 2, 40, 8)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=8)
    labels = kmeans.fit_predict(img_flat)
    counts = Counter(labels)

    total = sum(counts.values())
    color_out = []

    for idx, count in counts.items():
        perc = count * 100 / total
        rgb = tuple(np.round(kmeans.cluster_centers_[idx]).astype(int))
        color_out.append((
            rgb_to_name(rgb),
            rgb,
            rgb_to_hex(rgb),
            perc
        ))

    color_out.sort(key=lambda x: x[3], reverse=True)

    st.markdown("### 2. Dominant Color Palette")
    df = pd.DataFrame([
        {
            "Color": f"<span class='circle-color' style='background:{hex_col}'></span>",
            "Name": name.title(),
            "RGB": str(rgb),
            "HEX": f"<code>{hex_col.upper()}</code>",
            "Proportion (%)": f"{perc:.2f}"
        }
        for name, rgb, hex_col, perc in color_out
    ])

    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        [p for _, _, _, p in color_out],
        colors=[h for _, _, h, _ in color_out],
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 3. Pixel Color Picker")

    h, w, _ = img_np.shape
    col1, col2 = st.columns(2)

    with col1:
        x = st.number_input("X", 0, w-1, w//2)
    with col2:
        y = st.number_input("Y", 0, h-1, h//2)

    rgb_at = tuple(img_np[y, x])
    hex_at = rgb_to_hex(rgb_at)
    name_at = rgb_to_name(rgb_at)

    st.markdown(
        f"""
        <div class="mj-card" style="display:flex;align-items:center;">
            <span class="bigcircle-color" style="background:{hex_at}"></span>
            <div>
                <b>Pixel ({x}, {y})</b><br>
                Name: {name_at.title()}<br>
                RGB: {rgb_at}<br>
                HEX: <code>{hex_at.upper()}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<hr><p style='text-align:center;color:#777;letter-spacing:.2em;'>MARYJANE AI COLOR PLATFORM © 2025</p>",
        unsafe_allow_html=True
    )

else:
    st.markdown(
        "<p style='text-align:center;color:#aaa;margin-top:2em;'>Upload an image to begin analysis</p>",
        unsafe_allow_html=True
    )