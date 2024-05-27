# This script initialize the LSTM text classifier and its dependencies.
# Requires :
# - Trained text model weights
# - Trained label encoder
# - Trained tokenizer
# - Trained OneHot encoder
# Last revision : Mathis Doyon - 08/12/2023


###############################################################################
#                                   OPTIONS                                   #
###############################################################################

root_folder = "../../../models"
model_weights_path = f"{root_folder}/lstm_classifier.h5"
vectorizer_path = f"{root_folder}/trained_processing_utils/vectorizer.joblib"
label_encoder_path = f"{root_folder}/trained_processing_utils/label_encoder.joblib"
onehot_encoder_path = f"{root_folder}/trained_processing_utils/onehot_encoder.joblib"
padded_sequences_len = 527

###############################################################################
#                       DECLARATIONS AND MODEL LOADING                        #
###############################################################################

import joblib
from keras.saving import load_model

label_encoder = joblib.load(label_encoder_path)
onehot_encoder = joblib.load(onehot_encoder_path)
vectorizer = joblib.load(vectorizer_path)
total_words = len(vectorizer.word_index) + 1
model = load_model(model_weights_path)

###############################################################################
#                              MODEL ARCHITECTURE                             #
###############################################################################

# from keras.models import Sequential
# from keras.layers import Embedding, LSTM, Dense, Bidirectional

# model = Sequential()
# model.add(Embedding(total_words, 100, input_length=padded_sequences_len))
# model.add(Bidirectional(LSTM(100)))
# model.add(Dense(27, activation="softmax"))
# model.load_weights(model_weights_path)
