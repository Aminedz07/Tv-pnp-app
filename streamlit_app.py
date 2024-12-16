import streamlit as st
import numpy as np
import cv2
from PIL import Image 
from Proxy_Func import *
from Begin_Func import *
from Variational_Func import * 
from Mix_Func import * 
from Pnp_Algorithms import * 
from Denoisers import * 

# Imports de vos méthodes de débruitage
# from your_denoising_module import pnp_denoising, tv_denoising


denoiser = DRUNet()


st.title("Application de Débruitage d'Image")

uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file)
    image_np = np.array(image)
    st.image(image, caption="Image Téléchargée", use_column_width=True)

    method = st.selectbox("Choisissez la méthode de débruitage", ("TV", "PnP"))
    
    noise_level = st.slider("Niveau de bruit", min_value=0, max_value=100, value=20)
    K = st.slider("Nombre d'itérations", min_value=1, max_value=100, value=10)
    lamb = st.number_input("Paramètre de régularisation (lamb)", min_value=0.01, value=0.1)
    eps = st.number_input("Paramètre pour éviter les divisions par zéro (eps)", min_value=0.01, value=1e-3)
    tau = st.number_input("Pas de mise à jour (tau)", min_value=0.01, value=0.1)
    
    if method == "PnP":
        operator_type = st.selectbox("Type d'opérateur", ("none", "mask", "convolution"))
        operator_params = st.text_input("Paramètres de l'opérateur (JSON)", "{}")
        sigma = st.number_input("Niveau de bruit pour le débruiteur", min_value=0.01, value=0.1)

    if st.button("Débruiter"):
        noisy_image = add_gaussian_noise(image_np, noise_level)
        st.image(noisy_image, caption="Image Bruitée", use_column_width=True)

        if method == "TV":
            denoised_image = Denoise_TV(noisy_image, K, lamb, eps, tau)
            st.image(denoised_image, caption="Image Débruitée par TV", use_column_width=True)
        else:
            # Convertir les paramètres d'opérateur de JSON en dictionnaire
            import json
            operator_params_dict = json.loads(operator_params)
            
            denoised_image, _ = pnp_pgm(noisy_image, operator_type, operator_params_dict, tau, denoiser, sigma=sigma, K=K)
            st.image(denoised_image, caption="Image Débruitée par PnP", use_column_width=True)