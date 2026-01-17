
# # ======================================================
# # 🌾 Precision Agriculture Analytics (Pixel-Level System)
# # ======================================================

# import logging
# import numpy as np
# import cv2
# from PIL import Image
# import streamlit as st

# # ------------------------------------------------------
# # Config
# # ------------------------------------------------------
# logging.getLogger("streamlit").setLevel(logging.ERROR)
# np.random.seed(42)

# st.set_page_config(
#     page_title="Precision Agriculture Analytics",
#     layout="wide"
# )

# # ------------------------------------------------------
# # Global Styling (Agriculture Theme)
# # ------------------------------------------------------
# st.markdown("""
# <style>
# body {
#     background-color: #f4f8f2;
# }
# .block-container {
#     padding-top: 1.5rem;
# }
# .header-box {
#     background: linear-gradient(90deg, #2e7d32, #66bb6a);
#     padding: 25px;
#     border-radius: 14px;
#     color: white;
#     text-align: center;
# }
# .card {
#     background: white;
#     padding: 18px;
#     border-radius: 14px;
#     box-shadow: 0 4px 12px rgba(0,0,0,0.08);
# }
# .metric {
#     text-align: center;
#     padding: 20px;
#     border-radius: 14px;
#     background: #eef5ea;
# }
# .metric h2 {
#     margin: 0;
#     color: #2e7d32;
# }
# .metric p {
#     color: #4e5d52;
#     font-size: 14px;
# }
# </style>
# """, unsafe_allow_html=True)

# # ------------------------------------------------------
# # Header
# # ------------------------------------------------------
# st.markdown("""
# <div class="header-box">
#     <h1>🌾 Precision Agriculture Analytics</h1>
#     <p>
#         Pixel-level analysis of drone imagery for pest risk,
#         nutrient stress & yield estimation
#     </p>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)

# # ======================================================
# # IMAGE PROCESSING (PIXEL LEVEL)
# # ======================================================

# def preprocess_image(image):
#     img = np.array(image)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     img = cv2.resize(img, (900, 600))
#     img = cv2.GaussianBlur(img, (5, 5), 0)
#     return img


# def vegetation_index_exg(img):
#     img = img.astype(np.float32)
#     B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]
#     exg = 2 * G - R - B
#     return cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# def vegetation_mask(exg):
#     _, veg = cv2.threshold(exg, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))
#     veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, np.ones((15,15),np.uint8))
#     return veg


# def stress_mask(exg, veg):
#     if np.sum(veg) == 0:
#         return np.zeros_like(veg)

#     stress_score = cv2.bitwise_not(exg) * veg
#     threshold = np.percentile(stress_score[veg==1], 85)
#     stress = (stress_score >= threshold).astype(np.uint8)

#     return cv2.morphologyEx(stress, cv2.MORPH_CLOSE, np.ones((25,25),np.uint8))


# # ======================================================
# # 🐛 PIXEL-LEVEL PEST RISK (TEXTURE-BASED)
# # ======================================================

# def pest_nutrient_segmentation(stress, image):

#     if np.sum(stress) == 0:
#         return np.zeros_like(stress), np.zeros_like(stress)

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
#     lap = cv2.normalize(lap, None, 0, 1, cv2.NORM_MINMAX)

#     edges = cv2.Canny(gray, 80, 160) / 255.0

#     blur = cv2.GaussianBlur(gray, (15, 15), 0)
#     contrast = cv2.normalize(np.abs(gray - blur), None, 0, 1, cv2.NORM_MINMAX)

#     pest_score = (0.4 * lap + 0.4 * edges + 0.2 * contrast) * stress

#     if np.sum(pest_score) == 0:
#         return np.zeros_like(stress), stress

#     thresh = np.percentile(pest_score[stress==1], 80)

#     pest = (pest_score >= thresh).astype(np.uint8)
#     nutrient = stress.copy()
#     nutrient[pest == 1] = 0

#     return pest, nutrient


# def overlay(image, pest, nutrient):
#     overlay = image.copy()
#     overlay[pest==1] = (0, 0, 255)       # Pest Risk
#     overlay[nutrient==1] = (0, 255, 255) # Nutrient Stress
#     return cv2.addWeighted(overlay, 0.35, image, 0.65, 0)


# # ======================================================
# # ANALYTICS (PIXEL AGGREGATION)
# # ======================================================

# def area_percentage(mask):
#     return round((np.sum(mask) / mask.size) * 100, 2)


# def predict_yield(pest_area, nutrient_area):
#     base = 2500
#     loss = min(pest_area * 2.2 + nutrient_area, 50)
#     return round(base * (1 - loss / 100), 1), round(loss, 1)


# def confidence_score(veg, stress):
#     if np.sum(veg) == 0:
#         return 0
#     ratio = np.sum(stress) / np.sum(veg)
#     return round((1 - ratio) * 100, 1)


# # ======================================================
# # SIDEBAR (START POINT)
# # ======================================================
# st.sidebar.header("🚜 Drone Image Input")
# uploaded = st.sidebar.file_uploader(
#     "Upload Crop Field Image (Drone)",
#     type=["jpg", "jpeg", "png"]
# )
# run = st.sidebar.button("🌱 Analyze Field")

# # ======================================================
# # MAIN PIPELINE
# # ======================================================
# if uploaded and run:

#     image = Image.open(uploaded).convert("RGB")
#     processed = preprocess_image(image)

#     exg = vegetation_index_exg(processed)
#     veg = vegetation_mask(exg)
#     stress = stress_mask(exg, veg)
#     pest, nutrient = pest_nutrient_segmentation(stress, processed)
#     output = overlay(processed, pest, nutrient)

#     pest_area = area_percentage(pest)
#     nutrient_area = area_percentage(nutrient)
#     yield_val, loss = predict_yield(pest_area, nutrient_area)
#     confidence = confidence_score(veg, stress)

#     # ---------------- IMAGES ----------------
#     st.subheader("🛰️ Drone Imagery Analysis")
#     img_col1, img_col2 = st.columns(2)

#     with img_col1:
#         st.image(image, caption="Original Drone Image", width="stretch")

#     with img_col2:
#         st.image(
#             cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
#             caption="Pixel-Level Stress Map (Red = Pest Risk | Yellow = Nutrient Stress)",
#             width="stretch"
#         )

#     # ---------------- RESULTS ----------------
#     st.markdown("<br>", unsafe_allow_html=True)
#     st.subheader("📊 Field Analysis Results")

#     r1, r2, r3, r4 = st.columns(4)

#     for col, title, val in zip(
#         [r1, r2, r3, r4],
#         ["Pest Risk Area (%)", "Nutrient Stress Area (%)", "Total Affected (%)", "Estimated Yield (kg/acre)"],
#         [pest_area, nutrient_area, round(pest_area+nutrient_area,2), yield_val]
#     ):
#         col.markdown(f"""
#         <div class="metric">
#             <h2>{val}</h2>
#             <p>{title}</p>
#         </div>
#         """, unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)

#     st.progress(int(confidence))
#     st.caption(f"📈 Detection Confidence: {confidence}% (pixel consistency based)")

#     st.info(
#         "This system performs **pixel-level segmentation** to identify pest-risk and nutrient-stress zones. "
#         "Results are aggregated to provide actionable field-level insights without false positives."
#     )

# else:
#     st.info("⬅ Upload a drone image to begin pixel-level field analysis.")

# ======================================================
# 🌾 Precision Agriculture Analytics – Advanced Pixel-Level System
# ======================================================

import logging
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# ------------------------------------------------------
# Config
# ------------------------------------------------------
logging.getLogger("streamlit").setLevel(logging.ERROR)
np.random.seed(42)

st.set_page_config(
    page_title="Precision Agriculture Analytics",
    layout="wide"
)

# ------------------------------------------------------
# Styling
# ------------------------------------------------------
st.markdown("""
<style>
.header-box {
    background: linear-gradient(90deg, #1b5e20, #66bb6a);
    padding: 24px;
    border-radius: 14px;
    color: white;
    text-align: center;
}
.metric {
    text-align: center;
    padding: 18px;
    border-radius: 14px;
    background: #eef5ea;
}
.metric h2 { color:#1b5e20; margin:0; }
.metric p { font-size:13px; color:#4e5d52; }
.badge {
    display:inline-block;
    padding:6px 14px;
    border-radius:20px;
    font-size:13px;
    color:white;
}
.high { background:#c62828; }
.medium { background:#f9a825; }
.low { background:#2e7d32; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
<h1>🌾 Precision Agriculture Analytics</h1>
<p>Advanced pixel-level pest risk & nutrient stress analysis using drone imagery</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ======================================================
# IMAGE PROCESSING
# ======================================================

def preprocess_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (900, 600))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def vegetation_index_exg(img):
    img = img.astype(np.float32)
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]
    exg = 2 * G - R - B
    return cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def vegetation_mask(exg):
    _, veg = cv2.threshold(exg, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, np.ones((15,15),np.uint8))
    return veg


def stress_mask(exg, veg):
    if np.sum(veg) == 0:
        return np.zeros_like(veg)

    stress_score = cv2.bitwise_not(exg) * veg
    thresh = np.percentile(stress_score[veg == 1], 90)
    stress = (stress_score >= thresh).astype(np.uint8)

    stress = cv2.morphologyEx(stress, cv2.MORPH_CLOSE, np.ones((25,25),np.uint8))
    return stress


# ======================================================
# ADVANCED PEST & NUTRIENT SEGMENTATION
# ======================================================

def pest_nutrient_segmentation(stress, image):

    if np.sum(stress) == 0:
        return np.zeros_like(stress), np.zeros_like(stress), 0, 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lap = cv2.normalize(np.abs(cv2.Laplacian(gray, cv2.CV_64F)), None, 0, 1, cv2.NORM_MINMAX)
    edges = cv2.Canny(gray, 90, 180) / 255.0
    contrast = cv2.normalize(np.abs(gray - cv2.GaussianBlur(gray,(21,21),0)), None, 0, 1, cv2.NORM_MINMAX)

    pest_score = (0.5 * lap + 0.3 * edges + 0.2 * contrast) * stress

    if np.sum(pest_score) == 0:
        return np.zeros_like(stress), stress, 0, area_percentage(stress)

    pest_thresh = np.percentile(pest_score[stress == 1], 85)
    pest = (pest_score >= pest_thresh).astype(np.uint8)

    # Remove tiny false positives
    pest = remove_small_regions(pest, min_pixels=300)

    nutrient = stress.copy()
    nutrient[pest == 1] = 0

    return pest, nutrient, area_percentage(pest), area_percentage(nutrient)


def remove_small_regions(mask, min_pixels=200):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            clean[labels == i] = 1
    return clean


def overlay(image, pest, nutrient):
    out = image.copy()
    out[pest == 1] = (0, 0, 255)
    out[nutrient == 1] = (0, 255, 255)
    return cv2.addWeighted(out, 0.35, image, 0.65, 0)


# ======================================================
# ANALYTICS
# ======================================================

def area_percentage(mask):
    return round((np.sum(mask) / mask.size) * 100, 2)


def severity_label(value):
    if value > 6:
        return "High", "high"
    elif value > 3:
        return "Medium", "medium"
    else:
        return "Low", "low"


def predict_yield(pest_area, nutrient_area):
    base = 2500
    loss = min(pest_area * 2.0 + nutrient_area * 0.4, 40)
    return round(base * (1 - loss / 100), 1), round(loss, 1)


def confidence_score(veg, stress):
    if np.sum(veg) == 0:
        return 0
    ratio = np.sum(stress) / np.sum(veg)
    return round((1 - ratio) * 100, 1)


# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("🚜 Drone Image Input")
uploaded = st.sidebar.file_uploader("Upload Crop Field Image", ["jpg","jpeg","png"])
run = st.sidebar.button("🌱 Analyze Field")

# ======================================================
# MAIN PIPELINE
# ======================================================
if uploaded and run:

    image = Image.open(uploaded).convert("RGB")
    processed = preprocess_image(image)

    exg = vegetation_index_exg(processed)
    veg = vegetation_mask(exg)
    stress = stress_mask(exg, veg)
    pest, nutrient, pest_area, nutrient_area = pest_nutrient_segmentation(stress, processed)
    output = overlay(processed, pest, nutrient)

    yield_val, loss = predict_yield(pest_area, nutrient_area)
    confidence = confidence_score(veg, stress)

    # ---------------- RESULTS FIRST ----------------
    st.subheader("📊 Field Analysis Summary")

    s1, s2, s3, s4 = st.columns(4)

    sev_p, cls_p = severity_label(pest_area)
    sev_n, cls_n = severity_label(nutrient_area)

    s1.markdown(f"<div class='metric'><h2>{pest_area}%</h2><p>Pest Risk Area</p><span class='badge {cls_p}'>{sev_p}</span></div>", unsafe_allow_html=True)
    s2.markdown(f"<div class='metric'><h2>{nutrient_area}%</h2><p>Nutrient Stress Area</p><span class='badge {cls_n}'>{sev_n}</span></div>", unsafe_allow_html=True)
    s3.markdown(f"<div class='metric'><h2>{round(pest_area+nutrient_area,2)}%</h2><p>Total Affected</p></div>", unsafe_allow_html=True)
    s4.markdown(f"<div class='metric'><h2>{yield_val}</h2><p>Yield (kg/acre)</p></div>", unsafe_allow_html=True)

    st.progress(int(confidence))
    st.caption(f"📈 Detection Confidence: {confidence}%")

    # ---------------- IMAGES ----------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🛰️ Pixel-Level Field Visualization")

    c1, c2 = st.columns(2)
    with c1:
        st.image(image, caption="Original Drone Image", use_container_width=True)
    with c2:
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                 caption="Red: Pest Risk | Yellow: Nutrient Stress",
                 use_container_width=True)

else:
    st.info("⬅ Upload a drone image to begin advanced pixel-level analysis.")
