# This script trains the multimodal classifier. Expects the image and text classifiers to be defined in the same folder.
# Requires :
# - Image model weights
# - Text model weights
# - Trained label encoder
# - Trained vectorizer
# - Trained OneHot encoder
# Last revision : Mathis Doyon - 08/12/2023


###############################################################################
#                                   OPTIONS                                   #
###############################################################################

# The image path should point to the directory containing the test and train images folders
images_path = "../../../data/processed/images"
save_path = "../../../models/multimodal_classifier.keras"
csv_path = "../../../data/processed/cleaned_text.csv"
batch_size = 32
validation_split = 0.2

###############################################################################
#                                DECLARATIONS                                #
###############################################################################

from datetime import datetime
import pandas as pd
import os
from tensorflow import io, image, data
import numpy as np
from image_classifier import model as image_model
from text_classifier_lstm import model as text_model
from text_classifier_lstm import vectorizer, padded_sequences_len
from keras.layers import Dense, Concatenate, BatchNormalization, Dropout
from keras.models import Model
from keras.metrics import Recall, Precision
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def generate_filenames(df):
    names = []
    for i in df.values:
        filename = f"{i[2]}/{i[1]}_product_{i[0]}.jpg"
        if os.path.exists(os.path.join(images_path, "test", filename)):
            names.append(os.path.join(images_path, "test", filename))
        elif os.path.exists(os.path.join(images_path, "train", filename)):
            names.append(os.path.join(images_path, "train", filename))
        else:
            names.append(np.nan)
    return names


def load_and_preprocess(image_filename, text, target):
    img = io.read_file(image_filename)
    img = image.decode_jpeg(img, channels=3)
    img = image.resize(img, [200, 200])
    return (img, text), target


###############################################################################
#                                   TRAINING                                  #
###############################################################################

# Start clock
start = datetime.now()

# Data ingestion
df = pd.read_csv(csv_path)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data before training

# Images
df["filenames"] = generate_filenames(df)
df.dropna(inplace=True)
df["description"].fillna(" ", inplace=True)

# Target encoding
print("Unique prdtypecode values:", df["prdtypecode"].unique())

# Ajoutez également cette vérification
if df["prdtypecode"].nunique() == 0:
    raise ValueError("No valid prdtypecode values found in the dataset.")

# Target encoding
integer_encoded = label_encoder.transform(df["prdtypecode"])
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)


# Text
processed_text = vectorizer.texts_to_sequences(df["description"])
processed_text = pad_sequences(processed_text, maxlen=padded_sequences_len)

# Pipeline
print("Generating dataset...")
dataset = data.Dataset.from_tensor_slices(
    (df["filenames"], processed_text, onehot_encoded)
)
dataset = dataset.map(load_and_preprocess).batch(batch_size)
dataset = dataset.prefetch(data.AUTOTUNE)

validation_samples = int(validation_split * dataset.cardinality().numpy())

validation_dataset = dataset.take(validation_samples)
train_dataset = dataset.skip(validation_samples)
print("Dataset generated.")

# Multimodal classifier declaration
for layer in image_model.layers:
    layer.trainable = False
for layer in text_model.layers:
    layer.trainable = False

image_norm = BatchNormalization()(image_model.output)
text_norm = BatchNormalization()(text_model.output)
x = Concatenate(axis=1)([image_norm, text_norm])
x = Dense(512, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(27, activation="softmax")(x)

multimodal_model = Model(
    inputs=[image_model.inputs, text_model.inputs], outputs=outputs
)

multimodal_model.compile(
    optimizer="nadam",
    loss="categorical_crossentropy",
    metrics=["accuracy", Recall(), Precision()],
)
multimodal_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    batch_size=batch_size,
    epochs=3,
)

multimodal_model.save(save_path)
print("Training time :", datetime.now() - start)
