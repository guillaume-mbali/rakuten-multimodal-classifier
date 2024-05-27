import streamlit as st
from src.sidebar import sidebar
import src.backend as backend
from numpy.random import randint
import numpy as np
from src.utils import process_image, fetch_image

if "is_multimodal_loaded" not in st.session_state:
    st.session_state.is_multimodal_loaded = False
if "selected_line" not in st.session_state:
    st.session_state.selected_line = 46942


def set_random_line():
    st.session_state.selected_line = randint(0, len(backend.df))


df_line = backend.load_raw_df().loc[st.session_state.selected_line, :]
df_preprocessed_line = backend.load_processed_df().loc[
    st.session_state.selected_line, :
]

# Nettoyer les valeurs de description et designation
description = backend.clean_text(df_line["description"])
designation = backend.clean_text(df_line["designation"])

image_name = (
    "image_"
    + str(df_line["imageid"])
    + "_product_"
    + str(df_line["productid"])
    + ".jpg"
)
(
    img_contrasted,
    bounding_box_image,
    processed_image,
) = process_image(image_name)

original_image = fetch_image(image_name)
vectorized_text = backend.vectorize_text([description])

processed_image_np = np.array(processed_image)
processed_image_batch = np.expand_dims(processed_image_np, axis=0)
pred = backend.predict_with_multimodal(
    processed_image_batch, backend.pad_text(vectorized_text)
)

sidebar()
st.title("Classifieur multimodal")

tab1, tab2 = st.tabs(["Description du modèle", "Démo"])
with tab1:
    st.subheader("Architecture")
    st.markdown(
        """
        Le modèle repose sur le **CNN basé sur EfficientNetV2 B3** pour traiter les images, 
        et un modèle **LSTM** pour le texte. Chaque modèle fait une prédiction indépendante, 
        renvoyant une matrice de probabilités qui sont ensuite fusionnées.
    """
    )
    st.image("images/multimodal.png")

    st.subheader("Résultats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "0.95")
    col2.metric("Recall", "0.96")
    col3.metric("F1-score", "0.96")

with tab2:
    st.subheader("Démo")
    st.write(
        "Ci-dessous un widget permettant de tester le classifieur multimodal, et de voir en action toute la pipeline de traitement des données."
    )
    with st.container(border=True):
        if not st.session_state.is_multimodal_loaded:
            if st.button("Charger le modèle"):
                with st.spinner("Chargement du modèle..."):
                    multimodal_model = backend.load_multimodal_classifier()
                    st.session_state.is_multimodal_loaded = True
                    st.rerun()
        else:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.session_state.selected_line = st.number_input(
                    "Ligne",
                    min_value=0,
                    max_value=len(backend.df),
                    value=st.session_state.selected_line,
                )
            with col2:
                st.markdown(
                    """<div style="margin-bottom:3px">&emsp;</div>""",
                    unsafe_allow_html=True,
                )
                st.button("Aléatoire", on_click=set_random_line)
            st.divider()

            st.markdown("""<h4>Données brutes</h4>""", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""<p>Image</p>""", unsafe_allow_html=True)
                with st.container(border=True):
                    st.image(original_image, channels="BGR")
            with col2:
                st.markdown("""<p>Texte</p>""", unsafe_allow_html=True)
                with st.container(border=False):
                    with st.expander(
                        f"Longeur totale : {len(description + designation)} caractères"
                    ):
                        st.markdown("**Titre :** " + designation)
                        st.markdown("**Description :** " + description)
            st.divider()

            st.markdown("""<h4>Données traitées</h4>""", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""<p>Image</p>""", unsafe_allow_html=True)
                with st.container(border=True):
                    st.image(processed_image, channels="BGR")
            with col2:
                st.markdown(f"""<p>Texte</p>""", unsafe_allow_html=True)
                with st.container(border=False):
                    with st.expander(
                        f"Longeur totale : {len(str(df_preprocessed_line['description']))} caractères"
                    ):
                        st.markdown(
                            "**Texte nettoyé :** " + df_preprocessed_line["description"]
                        )
                        st.markdown("**Texte vectorisé :** ")
                        st.table(vectorized_text)
            st.divider()

            st.markdown("""<h4>Résultat final</h4>""", unsafe_allow_html=True)
            with st.container(border=True):
                col1, col2, col3 = st.columns([1, 1, 1.2])
                with col1:
                    st.metric("Prédiction", pred)
                with col2:
                    st.metric("Classe réelle", df_preprocessed_line["prdtypecode"])
                with col3:
                    st.markdown(
                        """<p style="height:0px"></p>""", unsafe_allow_html=True
                    )
                    if pred == df_preprocessed_line["prdtypecode"]:
                        st.success("&ensp;Prédiction correcte ✔")
                    else:
                        st.error("&ensp;Prédiction incorrecte ❌")
