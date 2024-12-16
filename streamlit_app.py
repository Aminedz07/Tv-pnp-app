import streamlit as st
import numpy as np
import cv2
import sys
import os
from PIL import Image
import warnings
import json

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Ajoutez le chemin des fonctions personnalisées (si elles existent)
sys.path.append(os.path.join(os.path.dirname(__file__), 'function'))

# Importation des fonctions nécessaires pour le débruitage
from function.Proxy_Func import *
from function.Begin_Func import gradient, div, laplacian, norm, process_image_2
from function.Variational_Func import *
from function.Mix_Func import *
from function.Pnp_Algorithms import *
from function.Denoisers import *

# Instanciation de votre débruiteur
denoiser = DRUNet()

# Titre de l'application
st.title("Application de Débruitage d'Image")

# Fonction pour ajouter du bruit gaussien
def add_gaussian_noise(image, noise_level):
    row, col, ch = image.shape
    mean = 0
    sigma = noise_level
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.array(image, dtype=float) + gauss
    noisy = np.clip(noisy, 0, 1)  # Limite les valeurs entre 0 et 1
    return (noisy * 255).astype(np.uint8)

# Fonction pour calculer le PSNR
def calculate_psnr(original, noisy):
    mse = np.mean((original / 255.0 - noisy ) ** 2)  # Erreur quadratique moyenne
    if mse == 0:
        return 100  # Pas de bruit, PSNR infini
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Chargement de l'image
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Charge l'image en tant qu'objet PIL, puis la convertit en tableau numpy
        image = Image.open(uploaded_file)
        
        # Si l'image est en niveaux de gris, convertissez-la en RGB (3 canaux)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_np = np.array(image) 
        image_np = image_np / 255.0  # Normalisation entre 0 et 1
        
        # Affiche l'image téléchargée sur l'application
        st.image(image, caption="Image Téléchargée", use_column_width=True)

        # Sélection de la méthode de débruitage
        method = st.selectbox("Choisissez la méthode de débruitage", ("TV", "PnP"))
        
        # Paramètres du débruitage
        noise_level = st.slider("Niveau de bruit", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        
        # Définition de la valeur minimale pour les autres paramètres
        K = st.slider("Nombre d'itérations", min_value=1, max_value=1000, value=10)
        lamb = st.number_input("Paramètre de régularisation (lamb)", min_value=0.01, value=0.1, step=0.01)
        eps = st.number_input("Paramètre pour éviter les divisions par zéro (eps)", min_value=0.001, value=0.001, step=0.0001)
        tau = st.number_input("Pas de mise à jour (tau)", min_value=0.01, value=0.1, step=0.01)
        
        if method == "PnP":
            operator_type = st.selectbox("Type d'opérateur", ("none", "mask", "convolution"))
            operator_params = st.text_input("Paramètres de l'opérateur (JSON)", "{}")
            sigma = st.number_input("Niveau de bruit pour le débruiteur", min_value=0.01, value=0.1, step=0.01)

        if st.button("Débruiter"):
            # Ajout du bruit à l'image
            noisy_image = add_gaussian_noise(image_np * 255, noise_level * 255)  # Ajoute du bruit sur l'image non normalisée
            
            # Normalisation de l'image bruitée
            noisy_image = noisy_image / 255.0
            
            # Calcul du PSNR avant le débruitage
            psnr_value = calculate_psnr(image_np, noisy_image)
            
            # Affiche l'image bruitée et le PSNR
            st.image(noisy_image, caption="Image Bruitée", use_column_width=True)
            st.write(f"PSNR avant débruitage : {psnr_value:.2f} dB")

            if method == "TV":
                # Applique le débruitage Total Variation (TV)
                denoised_image = Denoise_TV(noisy_image * 255, K, lamb, eps, tau)  # Passe l'image non normalisée
                denoised_image = denoised_image / 255.0  # Normalise l'image débruitée
                st.image(denoised_image, caption="Image Débruitée par TV", use_column_width=True)
            else:
                try:
                    # Conversion des paramètres JSON en dictionnaire
                    operator_params_dict = json.loads(operator_params)
                    
                    # Applique le débruitage par PnP (Proximal and Non-Linear)
                    denoised_image, _ = pnp_pgm(noisy_image * 255, operator_type, operator_params_dict, tau, denoiser, sigma=sigma, K=K)
                    denoised_image = denoised_image / 255.0  # Normalise l'image débruitée
                    st.image(denoised_image, caption="Image Débruitée par PnP", use_column_width=True)
                except json.JSONDecodeError as e:
                    st.error(f"Erreur de parsing JSON : {e}")

    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image : {e}")
else:
    st.warning("Veuillez télécharger une image pour commencer.")
