"""EcoSort Vision | CSE437, Group 07 | Streamlit deployment."""
import io
from PIL import Image, UnidentifiedImageError
import streamlit as st

from model_utils import EXPECTED_CLASSES, REVIEW_THRESHOLD, load_model, predict

st.set_page_config(
    page_title="EcoSort Vision | Waste Classifier", page_icon="♻️", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 1.5rem;}
.eco-hero {background:linear-gradient(115deg,#123b32,#164f45);padding:25px 30px;border-radius:16px;color:#ffffff;margin-bottom:14px;}
.eco-hero h1 {color:#ffffff;margin:0 0 6px 0;}
.eco-hero p {color:#d8f5e8;margin:0;}
.meta {font-size:0.9rem;opacity:0.88}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="eco-hero">
  <h1>♻️ EcoSort Vision</h1>
  <p>AI-Based Waste Material Classification for Smart Recycling</p>
  <p class="meta">Group 07 · CSE437 Computer Vision · Energy &amp; Environment</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading Group 07's trained MobileNetV2 model...")
def get_model():
    return load_model()

try:
    model, preprocess, classes = get_model()
except Exception as exc:
    st.error("The model could not be loaded. Please check that mobilenet_v2_best.pth is in the GitHub repository next to app.py.")
    with st.expander("Technical details"):
        st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("About this project")
    st.write("**Model:** MobileNetV2 (trained by Group 07)")
    st.write("**Dataset:** RealWaste — 4,752 images, 9 material classes")
    st.write("**Published Milestone 4 test accuracy:** 84.99% on the held-out RealWaste test split")
    st.caption("Test accuracy is a dataset-level measurement and does not guarantee any single prediction.")
    st.link_button("View source dataset", "https://archive.ics.uci.edu/dataset/908/realwaste")
    with st.expander("Supported waste classes"):
        for name in EXPECTED_CLASSES:
            st.write("• " + name)
    st.divider()
    st.caption("Educational demonstration only. Not a physical sorting robot or hazardous-waste guidance.")

st.subheader("Try a waste photo")
st.write("Upload one clear photo showing **one main waste material**. Supported formats: JPG, JPEG or PNG.")

left, right = st.columns([1, 1], gap="large")
with left:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    image = None
    if uploaded is not None:
        if uploaded.size > 15 * 1024 * 1024:
            st.error("Image too large. Please choose a file under 15 MB.")
        else:
            try:
                image = Image.open(io.BytesIO(uploaded.getvalue()))
                image.load()
                st.image(image, caption="Your uploaded waste photo", use_container_width=True)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("This file is not a readable image. Try a different JPG or PNG.")
    analyze = st.button("🔍 Classify Image", type="primary", use_container_width=True, disabled=image is None)

with right:
    st.markdown("#### Prediction results")
    if analyze and image is not None:
        with st.spinner("Analysing your image..."):
            ranked = predict(image, model, preprocess, classes)
        best_class, confidence = ranked[0]
        st.metric("Predicted material", best_class)
        st.metric("Model confidence", f"{confidence:.1%}")
        if confidence < REVIEW_THRESHOLD:
            st.warning("Low-confidence result — please review the image manually.")
        else:
            st.info("Prediction is an estimate. Check manually if the image contains mixed materials.")
        st.write("**Top 3 predictions**")
        for label, probability in ranked[:3]:
            st.write(f"{label} — {probability:.1%}")
            st.progress(min(max(probability, 0), 1))
        with st.expander("Show all nine model scores"):
            for label, probability in ranked:
                st.write(f"{label}: {probability:.2%}")
    else:
        st.caption("Your result will appear here after you upload a photo and click Classify Image.")

st.divider()
st.caption("**Important:** Confidence is a model score, not a guarantee of correctness. The app processes images in memory and does not intentionally save uploaded photos. Keep images free of sensitive personal information.")
