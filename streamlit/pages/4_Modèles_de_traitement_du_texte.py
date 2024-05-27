###############################################################################
#                                  FUNCTIONS                                  #
###############################################################################

import streamlit as st
import src.backend as backend
from src.sidebar import sidebar
from numpy.random import randint
from nltk.tokenize import word_tokenize
import numpy as np


def set_random_line():
    st.session_state.selected_line = randint(0, len(backend.df))


###############################################################################
#                            VARIABLES DECLARATIONS                           #
###############################################################################


if "is_linearsvc_loaded" not in st.session_state:
    st.session_state.is_linearsvc_loaded = False
if "is_lstm_loaded" not in st.session_state:
    st.session_state.is_lstm_loaded = False
if "selected_line" not in st.session_state:
    st.session_state.selected_line = 46942

###############################################################################
#                             SCRIPT INSTRUCTIONS                             #
###############################################################################

sidebar()
st.title("Modèles de traitement du texte")

ml_models, dl_models, demo, lstm = st.tabs(
    ["Machine learning", "Deep learning", "Démo", "Modèle retenu"]
)

with ml_models:
    st.markdown(
        """Pour classifier le texte, j'ai d'abord testé plusieurs **modèles de Machine Learning** : """
    )
    st.markdown(
        """ 
                - LinearSVC
                - SVC
                - Logistic Regression
                - SGD Classifier
                - MLP Classifier
                - K-Neighbors Classifier
                - RandomForest Classifier
                """
    )
    st.subheader("Vectorisation")
    st.markdown(
        """
        Ce texte est ensuite véctorisé à l'aide de la classe TfidfVectorizer plutôt que CountVectorizer. Contrairement à CountVectorizer
        la classeTfidfVectorizer prend en compte à la fois la fréquence du terme dans un document et son importance globale dans l'ensemble des documents.
    """
    )
    st.subheader("Recherche des hyperparamètres")
    st.write("Recherche des hyperparamètres sur 50% du Dataframe.")
    code = """
        param_distributions = {
            "SVC": {
                "C": [10, 100],
                "kernel": ["linear", "rbf", "poly"],
            },
            "LinearSVC": {
                "C": [1, 10, 100],
                "penalty": ["l1", "l2"],
            },
            "LogisticRegression": {
                "C": [1, 10, 100],
                "max_iter": [100, 200, 300],
            },
        }
        for model_name, param_dist in tqdm(param_distributions.items(), desc="Training models"):
            model = models_dict[model_name]
            random_search = RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=10, scoring="accuracy", cv=5
            )
    """
    st.code(code, language="python")
    st.subheader("Résultats")
    st.write(
        "Voici les résultats obtenu par les meilleurs modèles. Les modèles souffrent un petit peu d'overfitting, mais obtiennent globalement un bon score sur le texte nettoyé."
    )
    st.markdown("""<h5>Accuracy Ensemble d'entrainement</h5>""", unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SVC", 0.99)
        with col2:
            st.metric("LinearSVC", 0.98)
        with col3:
            st.metric("LogisticRegression", 0.88)

    st.markdown("""<h5>Accuracy Ensemble de test</h5>""", unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("SVC", 0.81, delta=-0.18)
        with col2:
            st.metric("LinearSVC", 0.82, delta=-0.16)
        with col3:
            st.metric("LogisticRegression", 0.79, delta=-0.09)

    st.subheader("Interpétation des résultats")
    st.write("Les classes les moins bien prédites sont les suivantes :")
    st.write("**Accuracy**")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("10", 0.47)
        with col2:
            st.metric("1280", 0.60)
        with col3:
            st.metric("1281", 0.58)

with demo:
    # Live testing
    st.subheader("Démo")
    st.write(
        "Voici un widget permettant d'afficher les différentes étapes de la **pipeline de nettoyage et traitement du texte**, puis essayant de prédire la classe du produit à l'aide du classifieur **LinearSVC** et du modèle **LSTM**."
    )
    # Widget
    with st.container(border=True):
        # Load model button
        if not st.session_state.is_linearsvc_loaded:
            if st.button("Charger le modèle", key="load_ml_model_button"):
                with st.spinner("Chargement du modèle..."):
                    linearsvc_model = backend.load_linearsvc_classifier()
                    st.session_state.is_linearsvc_loaded = True
                    st.rerun()
        # If model is loaded
        elif st.session_state.is_linearsvc_loaded:
            # Choose line
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

            # Display raw data
            st.markdown("""<h4>Données brutes</h4>""", unsafe_allow_html=True)

            line = backend.load_raw_df().loc[
                st.session_state.selected_line, :
            ]  # Get line from raw dataframe
            line = line.fillna(value="")
            with st.container():
                with st.expander(
                    f"Longeur totale : {len(str(line['description'])+str(line['designation']))} caractères"
                ):
                    st.markdown("**Titre :** " + line["designation"])
                    st.markdown("**Description :** " + line["description"])
            # Display preprocessed data
            line = backend.df.loc[st.session_state.selected_line, :]
            tokenized_text = word_tokenize(line["description"])
            vectorized_text = backend.load_tfidf_vectorizer().transform(tokenized_text)
            st.markdown("""<h4>Données nettoyées</h4>""", unsafe_allow_html=True)
            with st.container():
                with st.expander(
                    f"Longeur totale : {len(str(line['description']))} caractères"
                ):
                    st.markdown(tokenized_text)
                    st.markdown("**Texte vectorisé (format Sparse Matrix) :** ")
                    st.write(vectorized_text)
                    st.divider()

            # # Display model prediction
            st.markdown("""<h4>Prédictions</h4>""", unsafe_allow_html=True)
            # Linear SVC
            st.markdown("**LinearSVC**")
            linearsvc_model = backend.load_linearsvc_classifier()
            pred = linearsvc_model.predict(vectorized_text)[0]
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prédiction", str(pred))
            with col2:
                st.metric("Classe réelle", line["prdtypecode"])
            with col3:
                st.markdown("""<p style="height:0px"></p>""", unsafe_allow_html=True)
                if pred == line["prdtypecode"]:
                    st.success("&ensp;Prédiction correcte ✔")
                else:
                    st.error("Prédiction incorrecte ❌")
            # LSTM
            st.markdown("**LSTM**")
            pred = np.argmax(backend.predict_with_lstm([line["description"]]), axis=1)
            pred = backend.load_label_encoder().inverse_transform(pred)[0]
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prédiction", str(pred))
            with col2:
                st.metric("Classe réelle", line["prdtypecode"])
            with col3:
                st.markdown("""<p style="height:0px"></p>""", unsafe_allow_html=True)
                if pred == line["prdtypecode"]:
                    st.success("&ensp;Prédiction correcte ✔")
                else:
                    st.error("Prédiction incorrecte ❌")

with dl_models:
    st.markdown(
        "J'ai aussi testé **deux modèles de deep learning** pour classifier le texte."
    )

    st.markdown(
        "Les **LSTM (Long Short-Term Memory)** sont des réseaux de neurones **récurrents** dotés de mécanismes spéciaux pour **conserver et réguler** l'information sur de longues séquences, permettant ainsi de capturer efficacement les **dépendances temporelles complexes** dans les données. Ils comprennent des portes d'entrée, de sortie et d'oubli qui contrôlent le flux d'information, rendant les LSTM particulièrement adaptés à la modélisation des séquences dans le domaine du deep learning."
    )
    st.markdown(
        "Les **GRU (Gated Recurrent Unit**) sont des réseaux de neurones récurrents **simplifiés par rapport aux LSTM** (Long Short-Term Memory), avec des mécanismes de **portes d'update et de reset** pour contrôler le flux d'information dans la séquence temporelle. Leur **architecture plus compacte** facilite l'entraînement et les rend efficaces pour la modélisation des dépendances à long terme dans les données séquentielles."
    )

    st.subheader("Résultats")
    st.markdown("""<h5>Accuracy</h5>""", unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("LSTM", 0.78)
        with col2:
            st.metric("GRU", 0.79)

with lstm:
    st.markdown(
        "Les modèles de machine learning affichaient des résultats supérieurs, mais ils n'étaient **pas compatibles** avec l'utilisation de la classe `tf.data.Dataset`. Cette classe étant nécessaire pour pouvoir entraîner le classifieur multimodal, nous avons fait le choix d'un modèle **un peu moins performant initialement**, pour pouvoir construire un **modèle final plus performant** que le LinearSVC, le meilleur modèle de machine learning."
    )
    st.markdown(
        "Nous avons donc finalement retenu le modèle **LSTM** pour comme modèle final."
    )
    st.markdown("Voici l'architecture retenue :")
    _, col = st.columns([1, 2])
    with col:
        st.image("./images/LSTM.png")
