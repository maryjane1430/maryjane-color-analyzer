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
        width:56px;height:56px;border-radius:50%;display:inline-block;border:3px solid #222;margin:.2em 2em .2em .2em;box-shadow:0 3px 14px #8882;
    }}
    .mj-card  {{
        background:#fff;
        border-radius: 13px; box-shadow:0 6px 22px #839bab18, 0 1px 0 #eee;
        margin: 0 0 1.2em 0; padding:1.6em 2.2em 1.4em 2.2em;
        font-size:1.07em;
    }}
    hr {{
        margin:2.6em 0 .9em 0;
        border: none;
        border-bottom: 2px solid {ACCENT_COLOR};
        opacity:.18;
    }}
    label, .stSlider, .stNumberInput>div>label, .stDataFrame, .stMarkdown>p  {{
        font-family:'Prompt',sans-serif;font-weight:500;font-size:1.09em !important;
        color:#232321 !important;
    }}
    td, th {{
        padding:.6em 1.1em !important;
        font-family:'Prompt',sans-serif;font-size:1.08em;
    }}
    </style>
    """, unsafe_allow_html=True
)

st.markdown(f"<h1 class='main-title'>{TITLE}</h1>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class='subtitle-mj'>
    คนดีของพี่ แยกสีให้สนุก💋 อัปโหลดภาพแล้วระบบจะวิเคราะห์ทุกสีหลักในภาพ พร้อมชื่อสี รหัส RGB/HEX และตัวอย่าง
    </div>
    """, unsafe_allow_html=True
)

# ============== Function Section ==============
def rgb_to_name(rgb_triplet):
    try:
        color_name = webcolors.rgb_to_name(rgb_triplet)
    except ValueError:
        min_colors = {}
        if hasattr(webcolors, 'CSS3_NAMES_TO_HEX'):
            all_names = webcolors.CSS3_NAMES_TO_HEX.keys()
        elif hasattr(webcolors, 'HTML4_NAMES_TO_HEX'):
            all_names = webcolors.HTML4_NAMES_TO_HEX.keys()
        else:
            all_names = ['white', 'black', 'gray', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        for key in all_names:
            hex_code = webcolors.name_to_hex(key)
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
            rd = (r_c - rgb_triplet[0])**2
            gd = (g_c - rgb_triplet[1])**2
            bd = (b_c - rgb_triplet[2])**2
            min_colors[(rd+gd+bd)] = key
        color_name = min_colors[min(min_colors.keys())]
    return color_name

def rgb_to_hex(rgb_triplet):
    return '#{:02x}{:02x}{:02x}'.format(*rgb_triplet)

# =============== Web Main ===============
uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.container():
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="ตัวอย่างภาพที่อัปโหลด", use_container_width=True)
        img_np = np.array(image)
        img_flat = img_np.reshape((-1, 3))

    with st.container():
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### <span style='color:#045;letter-spacing:0.08em;'>1. เลือกจำนวนสีหลักที่ต้องการแยกดู</span>", unsafe_allow_html=True)
        n_colors = st.slider('How many dominant colors to analyze?', 2, 40, 8,
            help='(สามารถเลือกได้ตั้งแต่ 2 ถึง 40 สี)')
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=8)
        labels = kmeans.fit_predict(img_flat)
        counts = Counter(labels)
        total = sum(counts.values())
        color_out = []
        for idx, count in counts.items():
            perc = count * 100 / total
            rgb = tuple(np.round(kmeans.cluster_centers_[idx]).astype(int))
            name = rgb_to_name(rgb)
            hex_col = rgb_to_hex(rgb)
            color_out.append((name, rgb, hex_col, perc))
        color_out.sort(key=lambda x: x[3], reverse=True)

    with st.container():
        st.markdown("#### <span style='color:#045;letter-spacing:0.08em;'>2. เฉดสีหลักในภาพ (เรียงแถวตามสัดส่วนมากไปน้อย)</span>", unsafe_allow_html=True)
        color_table = pd.DataFrame([
            {
                "Color": f"<span class='circle-color' style='background-color:{hex_col};'></span>",
                "Name": name.title(),
                "RGB": str(rgb),
                "HEX": f"<span style='font-family:Mono;font-size:1.04em'>{hex_col.upper()}</span>",
                "Proportion (%)": f"{perc:.2f}"
            }
            for name, rgb, hex_col, perc in color_out
        ])
        st.write(
            color_table.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )

    with st.container():
        fig, ax = plt.subplots(figsize=(7, 7))
        pie_colors = [hex_col for _, _, hex_col, _ in color_out]
        labels_pie = [f"{name.title()} ({perc:.1f}%)" for name, _, _, perc in color_out]
        wedges, texts = ax.pie([perc for _, _, _, perc in color_out], colors=pie_colors, startangle=90)
        ax.axis('equal')
        plt.legend(wedges, labels_pie, title='Color Name', bbox_to_anchor=(1.1, 0.9), loc="upper left", fontsize=11)
        st.pyplot(fig, use_container_width=True)

    with st.container():
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom:.5em;'>3. ตรวจสอบ/เจาะจุดสีเฉพาะที่ต้องการ (Pixel Color Picker)</h4>", unsafe_allow_html=True)
        h, w, _ = img_np.shape
        st.markdown(
            f"<span style='color:#576;font-size:1.03em;'>ภาพนี้กว้าง <b>{w}</b> พิกเซล × สูง <b>{h}</b> พิกเซล</span>",
            unsafe_allow_html=True)

        col_x, col_y = st.columns(2)
        with col_x:
            x = st.number_input("ตำแหน่ง X (Horizontal)", min_value=0, max_value=w-1, value=w//2)
        with col_y:
            y = st.number_input("ตำแหน่ง Y (Vertical)", min_value=0, max_value=h-1, value=h//2)

        try:
            rgb_at = tuple(img_np[y, x])
            name_at = rgb_to_name(rgb_at)
            hex_at = rgb_to_hex(rgb_at)
            st.markdown(f"""
            <div class='mj-card' style='display:flex;align-items:center;'>
            <span class='bigcircle-color' style='background-color:{hex_at};'></span>
            <div>
            <b style='font-size:1.1em'>Pixel ({x}, {y})</b><br>
            <b>Name:</b> {name_at.title()} <br>
            <b>RGB:</b> {rgb_at} <br>
            <b>HEX:</b> <code>{hex_at.upper()}</code>
            </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.info("กรุณาเลือกค่าตำแหน่ง X/Y ที่อยู่ในขอบเขตของภาพ")

    st.markdown(
        "<hr><p style='color:#828; text-align:center; margin-top:2.5em; letter-spacing:.12em;font-family:Orbitron,Prompt,sans;'>Maryjane AI Color Platform &copy; 2025</p>",
        unsafe_allow_html=True
    )

else:
    st.markdown(
        "<div style='color:#787;font-size:1.14em;text-align:center;margin-top:2em;'>"
        "โปรดอัปโหลดไฟล์ JPG, JPEG หรือ PNG เพื่อเริ่มสนุกกับ Maryjane Color Analyzer<br>เฮอะเฮเล๊!"
        "</div>", unsafe_allow_html=True
    )