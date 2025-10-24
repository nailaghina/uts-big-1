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


# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/classifier_model.h5")
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
