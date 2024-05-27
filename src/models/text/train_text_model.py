from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from datetime import datetime
import joblib

# Constantes
CLEANED_DATA_PATH = "../../../data/processed/cleaned_text.csv"
LSTM_MODEL_EXPORT_PATH = "../../../models/lstm_classifier.keras"
VECTORIZER_EXPORT_PATH = "../../../models/trained_processing_utils/vectorizer.joblib"
LABEL_ENCODER_EXPORT_PATH = (
    "../../../models/trained_processing_utils/label_encoder.joblib"
)
ONEHOT_ENCODER_EXPORT_PATH = (
    "../../../models/trained_processing_utils/onehot_encoder.joblib"
)

OPTIONS = {
    "PADDED_SEQUENCES_LEN": 527,
    "EPOCHS": 5,  # Vous pouvez ajuster le nombre d'époques en fonction de votre besoin
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
}


def preprocess_data(df, padded_sequences_len, test_size, random_state):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["description"])
    total_words = len(tokenizer.word_index) + 1

    sequences = tokenizer.texts_to_sequences(df["description"])
    padded_sequences = pad_sequences(sequences, maxlen=padded_sequences_len)

    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(
        label_encoder.fit_transform(df["prdtypecode"]).reshape(-1, 1)
    )

    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, onehot_encoded, test_size=test_size, random_state=random_state
    )

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        total_words,
        label_encoder,
        onehot_encoded,
        tokenizer,
    )


def create_lstm_model(input_length, total_words, output_dim):
    model = Sequential(
        [
            Embedding(total_words, output_dim, input_length=input_length),
            Bidirectional(LSTM(output_dim)),
            Dense(output_dim, activation="softmax", name="dense_lstm"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs):
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
    return history


def export_dependencies(model, label_encoder, onehot_encoder, tokenizer):
    model.save(LSTM_MODEL_EXPORT_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_EXPORT_PATH)
    joblib.dump(onehot_encoder, ONEHOT_ENCODER_EXPORT_PATH)
    joblib.dump(tokenizer, VECTORIZER_EXPORT_PATH)


def main():
    date = datetime.now()

    # Chargement et prétraitement des données
    data_frame = pd.read_csv(CLEANED_DATA_PATH).dropna(subset=["description"])
    (
        X_train,
        X_val,
        y_train,
        y_val,
        total_words,
        label_encoder,
        onehot_encoded,
        tokenizer,
    ) = preprocess_data(
        data_frame,
        OPTIONS["PADDED_SEQUENCES_LEN"],
        OPTIONS["TEST_SIZE"],
        OPTIONS["RANDOM_STATE"],
    )

    # Création du modèle LSTM
    lstm_model = create_lstm_model(
        OPTIONS["PADDED_SEQUENCES_LEN"], total_words, onehot_encoded.shape[1]
    )

    # Entraînement du modèle LSTM
    history = train_lstm_model(
        lstm_model, X_train, y_train, X_val, y_val, OPTIONS["EPOCHS"]
    )

    # Export des dépendances
    export_dependencies(lstm_model, label_encoder, onehot_encoded, tokenizer)

    train_accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    print(
        f"Train accuracy: {train_accuracy}, Val accuracy: {val_accuracy}, Train loss: {train_loss}, Val loss: {val_loss}"
    )
    print("Saved LSTM model and dependencies to disk.")
    print("Training time:", datetime.now() - date)
    print("Success!")


if __name__ == "__main__":
    main()
