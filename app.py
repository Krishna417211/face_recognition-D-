"""STEP 4: Streamlit UI.   Run:  streamlit run app.py"""
import os
import streamlit as st
from PIL import Image
from common import load_classifier, recognize, draw

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("Face Detection & Recognition")


@st.cache_resource
def load():
    return load_classifier()


model, classes = load()


def run(file):
    img = Image.open(file).convert("RGB")
    img.thumbnail((1280, 1280))
    results = recognize(img, model, classes)
    st.image(draw(img, results), use_container_width=True)
    if not results:
        st.warning("No face found.")
    for _, name, conf in results:
        st.write(f"**{name}** ({conf:.0%} confidence)")


tab1, tab2, tab3 = st.tabs(["Upload image", "Webcam", "Training accuracy"])
with tab1:
    f = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if f:
        run(f)
with tab2:
    cam = st.camera_input("Take a photo")
    if cam:
        run(cam)
with tab3:
    if os.path.exists("accuracy.png"):
        st.image("accuracy.png")
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png")
    st.write("Known people:", ", ".join(classes))
