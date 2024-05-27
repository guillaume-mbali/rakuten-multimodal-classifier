import streamlit as st
from src.sidebar import sidebar

sidebar()
st.title("Modèle de traitement des images")

st.subheader("Réseau convolutif basé sur EfficientNet V2")
st.markdown(
    "Pour classifier les images, j'ai décidé d’utiliser un **CNN**, en basant notre architecture sur l’un des modèles de la littérature pour l'extraction des features avec plusieurs couches denses par-dessus. "
)
st.markdown(
    "Après avoir testé différents modèles comme VGG16 ou Xception, j'ai choisi d'utiliser un modèle **EfficientNet** comme backbone. Ces modèles sont non seulement **plus rapides à entraîner**, mais ils offrent dans ce contexte des **performances** tout aussi bonnes que les autres modèles."
)
st.write("Voici notre architecture finale :")
_, col = st.columns([1, 2])
with col:
    st.image("images/cnn.png", width=250)


st.subheader("Résultats")

st.markdown("""<h5>Ensemble d'entrainement</h5>""", unsafe_allow_html=True)
with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", 0.67)
    with col2:
        st.metric("Recall", 0.66)
    with col3:
        st.metric("F1-score", 0.65)

st.markdown("""<h5>Ensemble de test</h5>""", unsafe_allow_html=True)
with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", 0.56, delta=-0.11)
    with col2:
        st.metric("Recall", 0.57, delta=-0.09)
    with col3:
        st.metric("F1-score", 0.57, delta=-0.08)


st.write(
    "La qualité des images étant fortement variable dans ce jeu de données, les résultats de notre modèle sont satisfaisants pour cette tâche."
)
