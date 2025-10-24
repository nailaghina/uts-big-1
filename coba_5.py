import streamlit as st
from streamlit_option_menu import option_menu
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import base64
import random


# ==========================
# Load CSS Stars Background
# ==========================
def load_bg():
    try:
        with open("stars.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        st.warning("stars.css tidak ditemukan! Background tidak muncul.")

import random
import streamlit as st

def generate_stars():
    # Generate 150 twinkling stars
    stars_html = ""
    for _ in range(150):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        stars_html += f'<div class="star" style="top:{y}vh; left:{x}vw;"></div>'

    # Generate 20 falling stars
    for _ in range(20):
        x = random.randint(0, 100)
        stars_html += f'<div class="falling-star" style="left:{x}vw;"></div>'

    # Generate 30 SVG sharp stars
    for _ in range(30):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        stars_html += f'''
        <svg class="svg-star" style="top:{y}vh; left:{x}vw;" viewBox="0 0 24 24">
            <polygon points="12,2 15,9 23,9 17,14 19,21 12,17 
                             5,21 7,14 1,9 9,9"/>
        </svg>
        '''

    st.markdown(f"<div>{stars_html}</div>", unsafe_allow_html=True)


# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/best.h5")
    return yolo, classifier


# Apply Background
load_bg()

# Load Models
yolo_model, classifier_model = load_models()


# ==========================
# Sidebar Navigation
# ==========================
st.sidebar.header("Pilih Mode:")
selected_mode = st.sidebar.selectbox("",
                                     ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

st.sidebar.header("Pilih Menu:")
menu = st.sidebar.radio("", ["Deteksi Gambar", "Konfigurasi Gambar"])


# ==========================
# Image Upload Section
# ==========================
if menu == "Deteksi Gambar":

    st.title("📌 Image Detection System")

    uploaded_file = st.file_uploader("Unggah Gambar...",
                                     type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar berhasil diunggah ✅", use_column_width=True)

        if st.button("Proses Deteksi"):
            if selected_mode == "Deteksi Objek (YOLO)":
                results = yolo_model.predict(np.array(img))
                result_img = results[0].plot()

                st.image(result_img,
                         caption="Hasil Deteksi Objek ✅",
                         use_column_width=True)

            else:
                img_resized = img.resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
                result = classifier_model.predict(img_array)
                class_idx = np.argmax(result)

                st.success(f"✅ Hasil Klasifikasi: Kelas {class_idx}")


# ==========================
# Configuration Section
# ==========================
elif menu == "Konfigurasi Gambar":
    st.title("⚙️ Konfigurasi Gambar")
    st.info("Pengaturan gambar dan model akan ditambahkan di sini ✨")
    st.write("Silahkan request fitur tambahan jika diperlukan ✅")