import streamlit as st
import src.backend as backend
from src.sidebar import sidebar
from numpy import uint8
from cv2 import (
    cvtColor,
    threshold,
    findContours,
    resize,
    boundingRect,
    rectangle,
    contourArea,
    COLOR_BGR2GRAY,
    THRESH_BINARY_INV,
    RETR_EXTERNAL,
    CHAIN_APPROX_SIMPLE,
)
import src.functions as functions
from src.utils import process_image, fetch_image


if "selected_image_key" not in st.session_state:
    st.session_state["selected_image_key"] = None


original_image = fetch_image("image_482917_product_928735.jpg")
img_contrasted, bounding_box_image, processed_image = process_image(
    "image_482917_product_928735.jpg"
)


@st.cache_data
def run_pipeline(df):
    df_processed = df.head(13).copy()
    df_processed["description"].fillna("", inplace=True)
    df_processed["designation"].fillna("", inplace=True)

    # Suppression des balises HTML
    df_processed["description"] = df_processed["description"].apply(
        functions.delete_html_tags
    )
    df_processed["designation"] = df_processed["designation"].apply(
        functions.delete_html_tags
    )

    # Fusion des titres et des descriptions
    df_processed["identification"] = (
        df_processed["designation"].astype(str)
        + " . "
        + df_processed["description"].astype(str)
    )

    # Suppression des caractères spéciaux et de la ponctuation
    df_processed["identification"] = df_processed["identification"].apply(
        functions.remove_special_characters
    )
    df_processed["identification"] = df_processed["identification"].apply(
        functions.remove_punctuation
    )

    # Détection et traduction des langues
    df_processed["lang"] = df_processed["identification"].apply(
        functions.detect_language
    )
    df_processed["identification"] = df_processed.apply(
        lambda row: (
            functions.translate_text_to_french(
                row["identification"], source_lang=row["lang"]
            )
            if row["lang"] != "fr"
            else row["identification"]
        ),
        axis=1,
    )

    # Suppression des mots vides et correction orthographique
    df_processed["identification"] = df_processed["identification"].apply(
        lambda x: functions.remove_stopwords(x, lang="french")
    )
    df_processed["identification"] = df_processed["identification"].apply(
        lambda x: functions.correct_spelling(x, lang="fr")
    )

    # Tokenisation
    df_processed["tokens"] = df_processed["identification"].apply(
        functions.word_tokenize
    )

    return df_processed


# Chargement du DataFrame
df = backend.load_df_no_processing()

# Interface utilisateur
sidebar()
st.title("Preprocessing")
images_preprocessing, text_preprocessing = st.tabs(
    ["Traitement des images", "Traitement du texte"]
)

with text_preprocessing:
    st.write("DataFrame Original:")
    st.dataframe(df.head(13))
    st.title("Pipeline")
    st.code(
        """
        df["description"].fillna("", inplace=True)
        df.description = df.description.apply(functions.deleteHTML)
        df.designation = df.designation.apply(functions.deleteHTML)
        df["identification"] = df["designation"].astype(str) + " . " + df["description"].astype(str)
        df = df.drop(["designation", "description"], axis=1)
        df['langage_identification_avant_traduction'] = df['identification'].apply(detect)
        df["identification"] = df["identification"].apply(functions.delete_emoji)
        df['identification'] = df.apply(lambda row: functions.correct_text(row['identification'], row['langage_identification_avant_traduction']), axis=1)
        df["identification"] = df.apply(lambda row: functions.traduire_vers_françaisv3(row["identification"], row['langage_identification_avant_traduction']), axis=1)
        df['langage_identification_apres_traduction'] = df['identification'].apply(detect)
        df['identification'] = functions.delete_punctuation(df['identification'])
        df['identification'] = df['identification'].apply(functions.remove_stopwords)
        df['identification'] = df['identification'].str.lower()
        df['identification'] = df['identification'].apply(functions.tokenize_text)
        df.to_csv('dataprepreprocessing.csv',index=False)
    """,
        language="python",
    )

    if not st.session_state.get("is_pipeline_loaded", False):
        if st.button("Lancer la Pipeline"):
            with st.spinner("Exécution de la pipeline en cours..."):
                df_processed = run_pipeline(df)
                st.session_state.is_pipeline_loaded = True
            st.success("Succès ! Pipeline terminée.")
            st.dataframe(df_processed.head(13))

with images_preprocessing:
    st.subheader("Agrandissement du sujet")
    st.markdown(
        "La première étape du preprocessing des images consiste à **maximiser la taille du sujet**."
    )

    st.markdown(
        "Prenons l'image avec ID **84 312** comme exemple. La majorité de l'image est occupée par des **bordures blanches**."
    )
    st.dataframe(backend.df[backend.df["imageid"] == 482917])
    st.image(
        original_image, caption="Image Originale", channels="BGR", use_column_width=True
    )

    with st.expander("Traitement des images pour l'agrandissement du sujet"):
        st.subheader("Agrandissement du sujet")
        st.markdown(
            "La première étape du preprocessing des images consiste à **maximiser la taille du sujet**."
        )
        st.markdown(
            "Prenons l'image avec ID **84 312** comme exemple. La majorité de l'image est occupée par des **bordures blanches**."
        )
        st.image(original_image, caption="Image Originale", use_column_width=True)

        st.subheader("Étapes de traitement")
        tab1, tab2, tab3 = st.tabs(["Thresholding", "Bounding box", "Résultat"])

        with tab1:
            st.image(
                img_contrasted,
                caption="Image après Thresholding",
                use_column_width=True,
            )

        with tab2:
            st.image(
                bounding_box_image,
                caption="Image avec Bounding Box",
                use_column_width=True,
            )

        with tab3:
            st.image(processed_image, caption="Image Traitée", use_column_width=True)

        st.subheader("Equilibrage des classes")
        st.write(
            """
            La classe 2583 est largement majoritaire dans notre dataset. J'applique donc un undersampling en retirant 2000 images aléatoires de cette classe du jeu de données. Puis, je calcule également les poids de classes inversément proportionnels à leur effectif pour aider le modèle à classifier correctement les classes minoritaires.
        """
        )
