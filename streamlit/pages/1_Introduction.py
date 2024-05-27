import streamlit as st
from src.sidebar import sidebar
import src.backend as backend


sidebar()
st.title("Introduction")
intro, toc = st.tabs(["Introduction", "Table des matières"])

with intro:

    st.header("Contexte")
    st.markdown(
        "Ce projet s’inscrit dans le **challenge Rakuten France Multimodal Product Data Classification**. Disponible à l'adresse https://challengedata.ens.fr/challenges/35, il met au défi les participants de réaliser le meilleur modèle possible pour classifier des produits à partir d'une photo et d'un texte."
    )

    st.markdown(
        "**Rakuten** est une plateforme de commerce en ligne japonaise, offrant une vaste gamme de produits allant de l'électronique aux vêtements. Rakuten possède plusieurs **millions de produits** dans son catalogue, ce qui fait du catalogage et du tri de ces produits une **problématique cruciale** pour cette entreprise."
    )

    st.header("Description des données")
    st.markdown(
        f"""
        - **{len(backend.df)}** observations
        - **27** classes différentes
        - Chaque produit est représenté par une **image**, une **désignation** et une **description optionnelle**. Les données sont **multimodales**.
        - **~60 Mb** de données textuelles
        - **~2.2 Gb** d'images
        """
    )

    st.header("Objectif")
    st.markdown(
        "Mon objectif sera de réaliser un **classifieur multimodal** capable de classifier un produit en se basant sur sa miniature et une description textuelle."
    )
    st.markdown(
        "Sur le site du challenge, le dixième meilleur score appartient à **FEEEScientest**, une équipe venant de DataScientest. Mon objectif est d'approcher leur score de **0.8660**, et si possible de les dépasser."
    )


with toc:
    st.markdown(
        """
    <h4 class="big-title" style="font-weight:900">Introduction</h4>
    <h4 class="big-title" style="font-weight:900">Data exploration</h4>
        <h5 class="less-big-title">&emsp;Données textuelles</h5>
        <h5 class="less-big-title">&emsp;Images</h5>
    <h4 class="big-title" style="font-weight:900">Préprocessing</h4>
        <h5 class="less-big-title">&emsp;Données textuelles</h5>
        <h5 class="less-big-title">&emsp;Images</h5>
    <h4 class="big-title" style="font-weight:900">Modèles de traitement de texte</h4>
        <h5 class="less-big-title">&emsp;Modèle de Machine Learning</h5>
        <h5 class="less-big-title">&emsp;Modèle de Deep Learning</h5>
    <h4 class="big-title" style="font-weight:900">Modèle de traitement des images</h4>
        <h5 class="less-big-title">&emsp;CNN</h5>
    <h4 class="big-title" style="font-weight:900">Classifieur multimodal</h4>
    <h4 class="big-title" style="font-weight:900">Conclusion</h4>""",
        unsafe_allow_html=True,
    )
