import src.config as config
import pandas as pd
from cv2 import imread, resize, cvtColor, IMREAD_COLOR, COLOR_BGR2RGB
from numpy import argmax
import streamlit as st
from numpy import expand_dims
from keras.saving import load_model
from numpy import zeros, float32
import fsspec  # Importer fsspec
import s3fs
import os
import joblib
import nltk
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input


# Chemin relatif au fichier CSV
dataframe_path = "data/processed/cleaned_text.csv"


@st.cache_data
def load_processed_df():
    """Load the processed dataframe"""
    return pd.read_csv(
        config.processing_df_path,
        storage_options={
            "key": config.SCW_ACCESS_KEY,
            "secret": config.SCW_SECRET_KEY,
            "client_kwargs": {"endpoint_url": config.SCW_ENDPOINT_URL},
        },
    )


@st.cache_data
def load_raw_df():
    """Load the unprocessed dataframe"""
    return pd.read_csv(
        config.raw_df_path,
        sep=",",
        index_col="Unnamed: 0",
        storage_options={
            "key": config.SCW_ACCESS_KEY,
            "secret": config.SCW_SECRET_KEY,
            "client_kwargs": {"endpoint_url": config.SCW_ENDPOINT_URL},
        },
    )


@st.cache_data
def load_y_df():
    """Returns y"""
    return pd.read_csv(
        config.y_path,
        sep=",",
        index_col="Unnamed: 0",
        storage_options={
            "key": config.SCW_ACCESS_KEY,
            "secret": config.SCW_SECRET_KEY,
            "client_kwargs": {"endpoint_url": config.SCW_ENDPOINT_URL},
        },
    )


@st.cache_data
def load_df_no_processing():
    """Returns the unprocessed dataframe joined to y"""
    raw_df = load_raw_df()
    y_df = load_y_df()
    df = raw_df.join(y_df)
    return df


@st.cache_data
def load_label_encoder():
    """Returns the label encoder"""
    return joblib.load(config.label_encoder_path)


@st.cache_resource
def load_multimodal_classifier():
    """Returns the multimodal classifier"""
    return load_model(config.multimodal_model_path)


@st.cache_resource
def load_lstm():
    return load_model(config.lstm_model_path)


@st.cache_resource
def load_linearsvc_classifier():
    """For caching purpose. Do not call."""
    return joblib.load(config.linearsvc_model_path)


@st.cache_data
def load_tfidf_vectorizer():
    """For caching purpose. Do not call."""
    return joblib.load(config.tfidf_path)


@st.cache_data
def load_vectorizer():
    """Returns the LSTM trained vectorizer"""
    return joblib.load(config.vectorizer_path)


def predict_with_multimodal(processed_image, processed_text):
    try:
        # Assurez-vous que l'image est traitée correctement pour le modèle
        if isinstance(processed_image, np.ndarray):
            img_array = preprocess_input(
                processed_image
            )  # Appliquez le prétraitement nécessaire
        else:
            raise TypeError("L'image traitée doit être un tableau numpy")

        # Charger le modèle
        model = load_multimodal_classifier()

        # Préparez le texte si nécessaire
        # Exemple: processed_text = preprocess_text(text_input) où `preprocess_text` est votre fonction de traitement de texte

        # Faire la prédiction
        preds = model.predict([img_array, processed_text])
        decoded_preds = load_label_encoder().inverse_transform([np.argmax(preds)])
        return decoded_preds
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        raise


@st.cache_data
def predict_with_lstm(text):
    return load_lstm().__call__(pad_text(vectorize_text(text)))


def clean_text(text):
    if isinstance(text, float):
        return ""
    return text


def vectorize_text(texts):
    texts = [clean_text(text) for text in texts]
    vectorizer = load_vectorizer()
    if vectorizer is not None:
        return vectorizer.texts_to_sequences(texts)
    return None


@st.cache_data
def pad_text(text):
    """Pad a tokenized text sequence. Cached"""
    return pad_sequences(text, maxlen=527)


# Caching gives out some scope error, too late to fix so I disabled it
def fetch_unprocessed_images(namelist):
    output = zeros((len(namelist), 500, 500, 3))
    erreur_survenue = False

    for i in range(len(namelist)):
        try:
            if os.path.exists(config.raw_img_folder + "/" + namelist[i]):
                output[i] = imread(
                    config.raw_img_folder + "/" + namelist[i], IMREAD_COLOR
                )
            else:
                erreur_survenue = True
                print(
                    "ERREUR 1 : Fichier introuvable - "
                    + str(config.raw_img_folder + "/" + namelist[i])
                )
        except Exception as e:
            erreur_survenue = True
            print("ERREUR 2 : " + str(e))

    if output.max() > 1:
        output /= 255

    if erreur_survenue:
        st.toast("Un problème est survenu lors du chargement des images...", icon="❓")

    return float32(output)


# These ones most likely won't ever be called on the same arguments, so caching will
# only grow the memory stack without giving any performance improvements
def fetch_processed_images(namelist):
    """Read images in the input from the preprocessed images folder. Images class folder need to be appended before the image name. Returned images are in 500x500 BGR float32 format, with values in [0,1]"""
    output = zeros((len(namelist), 500, 500, 3))
    error_happened = False
    for i in range(len(namelist)):
        try:
            if os.path.exists(
                config.preprocessed_img_folder.format("train") + "/" + namelist[i]
            ):
                output[i] = imread(
                    config.preprocessed_img_folder.format("train") + "/" + namelist[i],
                    IMREAD_COLOR,
                )
            elif os.path.exists(
                config.preprocessed_img_folder.format("test") + "/" + namelist[i]
            ):
                output[i] = imread(
                    config.preprocessed_img_folder.format("test") + "/" + namelist[i],
                    IMREAD_COLOR,
                )
            elif os.path.exists(
                config.preprocessed_img_folder.format("discarded") + "/" + namelist[i]
            ):
                output[i] = imread(
                    config.preprocessed_img_folder.format("discarded")
                    + "/"
                    + namelist[i],
                    IMREAD_COLOR,
                )
            else:
                error_happened = True
        except:
            error_happened = True
    if output.max() > 1:
        output /= 255
    if error_happened:
        st.toast("Un problème est survenu lors du chargement des images", icon="❓")
    return float32(output)


def df_to_processed_images(df):
    """Returns the image names of the input dataframe while appending the class folder before"""
    return (
        df["prdtypecode"]
        + "/image_"
        + df["imageid"].astype(str)
        + "_product_"
        + df["productid"].astype(str)
        + ".jpg"
    )


def df_to_raw_images(df):
    """Returns the image names of the input dataframe without appending the class folder before"""
    return (
        "image_"
        + df["imageid"].astype(str)
        + "_product_"
        + df["productid"].astype(str)
        + ".jpg"
    )


# Variable declarations, poor man singleton
df = load_processed_df()
# nltk.download('punkt')
