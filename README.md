Challenge Rakuten
==============================
### Promotion Data Scientist Bootcamp - Septembre 2023

Ce projet s’inscrit dans le challenge Rakuten France Multimodal Product Data Classification disponible à l'adresse https://challengedata.ens.fr/challenges/35. Il s’agit de prédire le code type de produits (tel que défini dans le catalogue Rakuten France) à partir d’une description texte et d’une image.

Déroulement
-----------
J'ai conçu et entrainés 3 modèles :
- Un CNN basé sur Efficient Net V2 B3 pour extraire les features des images
- Un LSTM RNN pour traiter le texte
- Un classifieur multimodal fusionnant les deux modèles précédents pour effectueur une prédiction finale

Le classifieur multimodal final obtient une **accuracy de 0.95 sur l'ensemble de validation.**

Organisation du repository
------------
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    │
    ├── data
    │    ├── processed
    │    │    └── ...
    │    └── raw
    │         └── ...
    │
    ├── models
    │    ├── multimodal_classifier.keras
    │    ├── image_classifier.keras
    │    ├── lstm_classifier.keras
    │    ├── trained_processing_utils
    │    │    ├── label_encoder.joblib
    │    │    ├── onehot_encoder.joblib
    │    │    └── vectorizer.joblib
    │    │
    │    └── training_history
    │         └── ...
    │
    ├── src
    │   ├── features
    │   │    ├── preprocess_text.py  // TO DO
    │   │    └── preprocess_images.py
    │   │
    │   └── models
    │        ├── predict_model.py
    │        ├── train_model.py
    │        ├── multimodal
    │        │    └── train_multimodal_model.py
    │        ├── images
    │        │    └── train_image_model.py
    │        └── text
    │             └── train_text_model.py
    │
    ├── streamlit
    │    ├── Home.py
    │    └── ...
    │ 
    └── notebooks
--------
