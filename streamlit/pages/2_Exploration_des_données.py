import streamlit as st
import src.backend as backend
from src.sidebar import sidebar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.express import bar, pie
from src.utils import display_images, load_new_images
import os

sidebar()
st.title("Exploration des données")


def next_image():
    st.session_state.img_id = (st.session_state.img_id + 1) % 10


def prev_image():
    st.session_state.img_id = (st.session_state.img_id - 1) % 10


@st.cache_data
def load_data():
    try:
        return backend.load_df_no_processing()
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()


@st.cache_data
def graph1():
    colors = ["Small"] * 6 + ["Med"] * 20 + ["Big"]
    annotations = (
        ["Cette variable est sous-représentée"] * 6
        + ["L'effectif de cette variable est dans la norme"] * 20
        + ["Cette variable est sur-représentée"]
    )
    plotly_df = backend.df["prdtypecode"].value_counts(ascending=True).reset_index()
    plotly_df["colors"] = colors
    plotly_df["effectif"] = annotations
    fig = bar(
        plotly_df,
        x="prdtypecode",
        y="count",
        color="colors",
        color_discrete_map={"Med": "#123c60", "Big": "#f0ca9d", "Small": "#ef3b6e"},
        hover_name="prdtypecode",
        hover_data={
            "count": True,
            "prdtypecode": False,
            "effectif": False,
            "colors": False,
        },
    )
    fig.update_layout(
        xaxis_type="category",
        showlegend=False,
        xaxis_tickangle=-45,
        title_text="Répartition des classes",
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Effectif : %{y}<br>%{customdata[0]}<extra></extra>"
    )
    fig.update_xaxes(title_text="Codes produits")
    return fig


@st.cache_data
def graph2(description):
    description_counts = pd.DataFrame(description.isna().value_counts())
    description_counts["name"] = ["Avec description", "Sans description"]
    fig = pie(
        description_counts,
        values="count",
        title="Pourcentage de produits avec et sans description",
        hole=0.5,
        hover_name="name",
        color="name",
        color_discrete_map={
            "Avec description": "#123c60",
            "Sans description": "#ef3b6e",
        },
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def chart1():
    sns.set(style="whitegrid")
    df_sorted = df_no_processing["prdtypecode"].value_counts().reset_index()
    df_sorted.columns = ["prdtypecode", "count"]
    plt.figure(figsize=(7, 5))
    palette = sns.color_palette("Reds_r", len(df_sorted))
    sns.barplot(
        x="prdtypecode",
        y="count",
        data=df_sorted,
        order=df_sorted["prdtypecode"],
        palette=palette,
    )
    plt.title("Distribution des types de produits")
    plt.xlabel("Code de type de produit")
    plt.ylabel("Nombre de produits")
    plt.xticks(rotation=90)
    st.pyplot(plt)


def chart2():
    df_no_processing["description_length"] = df_no_processing["description"].apply(
        lambda x: len(x) if pd.notna(x) else 0
    )
    groupe_par_prdtype_description = (
        df_no_processing.groupby("prdtypecode")["description_length"]
        .mean()
        .reset_index()
    )
    palette = sns.color_palette("magma", n_colors=len(groupe_par_prdtype_description))
    plt.figure(figsize=(7, 5))
    sns.barplot(
        x="prdtypecode",
        y="description_length",
        data=groupe_par_prdtype_description,
        palette=palette,
    )
    plt.xlabel("Code de produit (prdtypecode)")
    plt.xticks(rotation=90)
    plt.ylabel("Taille moyenne des descriptions")
    plt.title("Taille moyenne des descriptions par prdtypecode")
    st.pyplot(plt)


def chart3():
    df_no_processing["designation_length"] = df_no_processing["designation"].apply(len)
    groupe_par_prdtype_designation = (
        df_no_processing.groupby("prdtypecode")["designation_length"]
        .mean()
        .reset_index()
    )
    palette = sns.color_palette("magma", n_colors=len(groupe_par_prdtype_designation))
    plt.figure(figsize=(7, 5))
    sns.barplot(
        x="prdtypecode",
        y="designation_length",
        data=groupe_par_prdtype_designation,
        palette=palette,
    )
    plt.xlabel("Code de produit (prdtypecode)")
    plt.xticks(rotation=90)
    plt.ylabel("Taille moyenne des désignations")
    plt.title("Taille moyenne des désignations par prdtypecode")
    st.pyplot(plt)


def chart4():
    df_no_processing["a_description"] = df_no_processing["description"].isna()
    description_counts = (
        df_no_processing["a_description"]
        .value_counts(normalize=True)
        .reindex([True, False])
    )
    plt.figure(figsize=(8, 8))
    colors = ["#8B0000", "#F08080"]
    plt.pie(
        description_counts,
        labels=["Avec Description", "Sans Description"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
    )
    plt.title("Pourcentage de produits avec et sans description")
    st.pyplot(plt)


if "img_id" not in st.session_state:
    st.session_state.img_id = 0
if "img_exploration_data" not in st.session_state:
    st.session_state.img_exploration_data = backend.fetch_unprocessed_images(
        backend.df_to_raw_images(backend.df.sample(n=10)).to_list()
    )
df_no_processing = load_data()

text_exploration, images_exploration = st.tabs(
    ["Exploration du texte", "Exploration des images"]
)

with text_exploration:
    st.subheader("Exploration du texte")
    st.subheader("*Jeu de données*")
    st.write(
        f"Le dataset contient **{len(backend.df)}** lignes. Voici les premières lignes."
    )
    st.dataframe(backend.load_raw_df().head())
    st.write("Le dataframe contient quatre colonnes :")
    st.markdown(
        """
        - **designation** : Libellé du produit
        - **description** : Description du produit
        - **productid** : Identifiant unique du produit
        - **imageid** : ID de l'image correspondante
    """
    )

    st.subheader("*Variable cible*")
    st.write(
        "**prdtypecode** : La variable cible est une variable catégorielle représentant le code produit de chaque observation. Le code est un entier positif et la variable a 27 modalités."
    )
    st.dataframe(
        [[backend.load_y_df().prdtypecode.unique()]],
        column_config={
            "0": st.column_config.ListColumn("Valeurs uniques de la variable cible")
        },
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Distribution",
            "Quantité de texte par classe",
            "Valeur manquantes",
            "Boxplot",
            "Exploration des langues du Dataframe",
        ]
    )
    with tab1:
        st.plotly_chart(graph1())
    with tab2:
        chart2()
        chart3()
    with tab3:
        st.plotly_chart(graph2(df_no_processing["description"]))
    with tab4:
        st.image("./images/statistiques_descriptions_designations.png")
        st.markdown(
            """
            Statistiques descriptives de la longueur des descriptions :
            - count    84916.000000
            - mean       524.555926
            - std        754.893905
            - min          0.000000
            - 25%          0.000000
            - 50%        231.000000
            - 75%        823.000000
            - max      12451.000000

            Statistiques descriptives de la longueur des désignations :
            - count    84916.000000
            - mean        70.163303
            - std         36.793383
            - min         11.000000
            - 25%         43.000000
            - 50%         64.000000
            - 75%         90.000000
            - max        250.000000
        """
        )
    with tab5:
        st.image("./images/langues.png")
        st.markdown(
            """
            Les langues présentes dans le jeu de données sont :
            ['fr', 'en', 'it', 'ca', 'pt', 'pl', 'de', 'ro', 'id', 'vi', 'tl', 'es', 'nl', 'fi', 'hr', 'et', 'sk', 'sl', 'sv', 'no', 'af', 'da', 'sw', 'so', 'cy', 'cs', 'tr', 'hu', 'lt', 'lv']
        """
        )
        st.markdown(
            "Le taux de correspondance entre les deux colonnes langage_description et langage_designation est de 70%."
        )

with images_exploration:
    st.subheader("Exploration des images")
    st.write(
        f"Le dataset contient **{len(backend.df)}** images. En voici quelques-unes."
    )
    st.markdown("<p>10 images du jeu de données</p>", unsafe_allow_html=True)
    with st.container(border=True):
        left_col, right_col = st.columns([5, 2.5])
        with left_col:
            display_images()
        with right_col:
            st.markdown(
                f"<p style='text-align: center;'><strong>{st.session_state.img_id + 1}/10</strong></p>",
                unsafe_allow_html=True,
            )
            st.button("Précédente", on_click=prev_image)
            st.button("Suivante", on_click=next_image)

    st.write(
        "La majorité des images ont des bordures blanches autour du sujet. Sur certaines images, ces bordures occupent la majorité de l'espace."
    )
    st.write(
        "Pour améliorer les performances du modèle, je vais essayer de réduire la quantité de ces pixels vides."
    )
