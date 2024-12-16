import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Imports de vos méthodes de débruitage
# from your_denoising_module import pnp_denoising, tv_denoising

def load_image(image_file):
    img = Image.open(image_file)
    return img

def add_noise(image, noise_level):
    noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def pnp_denoising(image, iterations):
    # Implémentez votre méthode PnP ici
    # Exemple : return votre_fonction_pnp(image, iterations)
    return image  # Remplacez par l'image débruitée

def tv_denoising(image, iterations):
    # Implémentez votre méthode TV ici
    # Exemple : return votre_fonction_tv(image, iterations)
    return image  # Remplacez par l'image débruitée

st.title("Application de Débruitage d'Image")

uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    image_np = np.array(image)
    st.image(image, caption="Image Téléchargée", use_column_width=True)

    method = st.selectbox("Choisissez la méthode de débruitage", ("PnP", "TV"))
    
    noise_level = st.slider("Niveau de bruit", min_value=0, max_value=100, value=20)
    iterations = st.slider("Nombre d'itérations", min_value=1, max_value=100, value=10)

    if st.button("Débruiter"):
        noisy_image = add_noise(image_np, noise_level)
        st.image(noisy_image, caption="Image Bruitée", use_column_width=True)

        if method == "PnP":
            denoised_image = pnp_denoising(noisy_image, iterations)
        else:
            denoised_image = tv_denoising(noisy_image, iterations)

        st.image(denoised_image, caption="Image Débruitée", use_column_width=True)