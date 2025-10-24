import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/best.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.write("### Hasil Prediksi:", class_index)

        st.write("Probabilitas:", np.max(prediction))

import streamlit as st
from streamlit_option_menu import option_menu
import base64

# =================================
# PAGE CONFIG
# =================================
st.set_page_config(
    page_title="Watch n Stopwatch",
    page_icon="📸",
    layout="wide",
)

# =================================
# BACKGROUND ANIMATION (CSS)
# =================================
def load_bg_animation():
    with open("bg.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =================================
# SIDEBAR
# =================================
with st.sidebar:
    st.markdown("<h2 style='color:white;'>📸 Exifa_net</h2>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["File Input", "Model Configuration"],
        icons=["upload", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "icon": {"color": "#00BFFF"},
            "nav-link": {
                "color": "#eee",
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
            },
            "nav-link-selected": {"background-color": "#1f5e7a"},
        }
    )

# =================================
# CONTENT AREA
# =================================
load_bg_animation()

st.markdown(
    """
    <div style='padding: 60px; text-align:center;'>
        <h1 style='color:#7cd3ff;'>Welcome to Exifa_net ❄️</h1>
        <p style='color:#c4c4c4;'>Upload image and configure your model.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if selected == "File Input":
    uploaded_file = st.file_uploader("📂 Upload gambar", type=["jpg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)

elif selected == "Model Configuration":
    st.subheader("⚙️ Model Settings")
    st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
