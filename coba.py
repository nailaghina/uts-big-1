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
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import cv2

def load_bg_animation():
    with open("stars.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_bg_animation()

# Tambahkan lapisan bintang
st.markdown(
    """
    <div id="stars"></div>
    <div id="stars2"></div>
    <div id="stars3"></div>
    """,
    unsafe_allow_html=True
)
# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model YOLO
    classifier = tf.keras.models.load_model("model/best.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Styling From CSS
# ==========================
try:
    with open("stars.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ File stars.css tidak ditemukan. Background mungkin tidak muncul.")

# ==========================
# UI Layout
# ==========================
st.title("🎯 Image Detection System")

menu = st.radio("Pilih Menu:", ["Deteksi Gambar", "Konfigurasi Gambar"])

# ==========================
# Upload Image Menu
# ==========================
if menu == "Deteksi Gambar":
    st.subheader("Upload Gambar")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar asli", use_column_width=True)

        if st.button("Deteksi Sekarang"):
            # YOLO Detection
            results = yolo_model.predict(np.array(img))
            annotated = results[0].plot()

            st.image(annotated, caption="Hasil YOLO", use_column_width=True)
            st.success("✅ Deteksi selesai!")

# ==========================
# KOnfigurasi Mode
# ==========================
elif menu == "Konfigurasi Gambar":
    st.title("Konfigurasi Gambar")
    st.write("Silakan atur konfigurasi model atau gambar di sini.")


