import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model, Model
from PIL import Image, ImageEnhance, ImageFilter
import os
import random
import base64
import matplotlib.pyplot as plt

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "models/autoencoder_denoising.keras"
TRAIN_DIR = "images/train"
TRAIN_AUG_DIR = "images/train_aug"
PDF_PATH = "digital_docs/presentacion.pdf"

@st.cache_resource
def load_autoencoder():
    return load_model(MODEL_PATH)

def get_bottleneck_model(model, layer_name="enc_pool2"):
    return Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

def visualize_encoding(feature_map):
    fmap = feature_map[0]
    num_channels = fmap.shape[-1]
    channels_to_show = min(num_channels, 6)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    for i in range(6):
        ax = axes[i]
        if i < channels_to_show:
            ax.imshow(fmap[:, :, i], cmap="gray")
            ax.set_title(f"Canal {i+1}", fontsize=8)
        else:
            ax.axis("off")
        ax.axis("off")

    st.pyplot(fig)


def load_matching_pairs(n=3):
    originals = [f for f in os.listdir(TRAIN_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
    noisy = set(os.listdir(TRAIN_AUG_DIR))
    valid = [f for f in originals if f in noisy]

    selected = random.sample(valid, min(n, len(valid)))

    result = []
    for fname in selected:
        orig_img = Image.open(os.path.join(TRAIN_DIR, fname)).convert("RGB")
        noisy_img = Image.open(os.path.join(TRAIN_AUG_DIR, fname)).convert("RGB")
        result.append((orig_img, noisy_img))

    return result

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Autoencoder Completo",
    page_icon="🧠",
    layout="wide"
)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("📑 Navegación")
section = st.sidebar.radio(
    "Selecciona una sección:",
    ["Presentación", "Modelo"]
)

# ==========================
# SECCIÓN 1: PRESENTACIÓN (PDF)
# ==========================
if section == "Presentación":
    st.title("📘 Presentación del Proyecto")

    st.write("Aquí puedes visualizar el PDF con la presentación.")

    try:
        with open(PDF_PATH, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")

        pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="900px"
            type="application/pdf">
        </iframe>
        """

        st.markdown(pdf_display, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error(f"No se encontró el archivo PDF en la ruta: {PDF_PATH}")


# ==========================
# SECCIÓN 2: MODELO
# ==========================
if section == "Modelo":

    st.title("🧠 Autoencoder — Flujo Completo con Bottleneck enc_pool2")

    model = load_autoencoder()
    bottleneck_model = get_bottleneck_model(model)

    # ==========================
    # DATASET EXAMPLES
    # ==========================
    st.markdown("## 📂 Ejemplos del Dataset (Original → Ruido → Codificación → Reconstrucción)")

    pairs = load_matching_pairs(3)

    for orig_img, noisy_img in pairs:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2])

        with col1:
            st.markdown("**Original**")
            st.image(orig_img, use_container_width=True)

        with col2:
            st.markdown("**Con Ruido**")
            st.image(noisy_img, use_container_width=True)

        noisy_arr = np.expand_dims(np.array(noisy_img) / 255.0, axis=0)
        encoding = bottleneck_model.predict(noisy_arr)

        with col3:
            st.markdown("**Codificación (enc_pool2)**")
            visualize_encoding(encoding)

        denoised = (model.predict(noisy_arr)[0] * 255).astype(np.uint8)

        with col4:
            st.markdown("**Reconstrucción**")
            st.image(denoised, use_container_width=True)

    st.markdown("---")

    # ==========================
    # USER IMAGE
    # ==========================
    st.markdown("## 📤 Imagen del Usuario")
    uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        image_proc = image.filter(ImageFilter.DETAIL)
        image_proc = ImageEnhance.Sharpness(image_proc).enhance(1.4)
        image_proc = ImageEnhance.Contrast(image_proc).enhance(1.4)

        arr = np.expand_dims(np.array(image_proc) / 255.0, axis=0)

        encoding = bottleneck_model.predict(arr)
        output = (model.predict(arr)[0] * 255).astype(np.uint8)

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2])

        with col1:
            st.markdown("**Original**")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("**Preprocesada**")
            st.image(image_proc, use_container_width=True)

        with col3:
            st.markdown("**Codificación (enc_pool2)**")
            visualize_encoding(encoding)

        with col4:
            st.markdown("**Reconstrucción Final**")
            st.image(output, use_container_width=True)
