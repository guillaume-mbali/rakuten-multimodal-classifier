import pandas as pd
import re
from langdetect import detect, LangDetectException
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from spellchecker import SpellChecker
from deep_translator import GoogleTranslator
import string

# Téléchargements NLTK nécessaires
import nltk

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")

##########################################################################################################


def remove_special_characters(text):
    """Supprime les caractères spéciaux, ne garde que les lettres et les nombres."""
    return re.sub(r"[^A-Za-z0-9\s]", "", text)


def remove_stopwords(text, lang="french"):
    """Supprime les mots vides basés sur la langue spécifiée."""
    stop_words = set(stopwords.words(lang))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


def delete_html_tags(text):
    """Supprime les balises HTML du texte."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_punctuation(text):
    """Supprime toute la ponctuation du texte."""
    return text.translate(str.maketrans("", "", string.punctuation))


def translate_text_to_french(text, source_lang="auto"):
    """Traduit le texte vers le français en utilisant deep_translator."""
    try:
        return GoogleTranslator(source=source_lang, target="fr").translate(text)
    except Exception as e:
        print(f"Erreur lors de la traduction : {e}")
        return text


def correct_spelling(text, lang="en"):
    """Corrige l'orthographe des mots dans le texte basé sur la langue spécifiée."""
    spell_checker = SpellChecker(language=lang)
    corrected_text = []

    # Tokenise le texte en mots
    words = word_tokenize(text)

    for word in words:
        # Utilisez le correcteur orthographique pour obtenir le mot corrigé
        corrected_word = spell_checker.correction(word)

        # Assurez-vous que corrected_word n'est pas None avant de l'ajouter à la liste
        if corrected_word is not None:
            corrected_text.append(corrected_word)
        else:
            # Si corrected_word est None, utilisez le mot original
            corrected_text.append(word)

    return " ".join(corrected_text)


def detect_language(text):
    """Détecte la langue du texte donné."""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


# Exemple d'utilisation des fonctions de nettoyage
text_example = "<p>Hello, this is a <b>test</b>.</p>"
cleaned_text = delete_html_tags(text_example)

# Traduction, si nécessaire
translated_text = translate_text_to_french(cleaned_text)

# Correction orthographique en anglais (pour exemple)
corrected_text = correct_spelling(cleaned_text, lang="en")
