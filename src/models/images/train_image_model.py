import joblib
import numpy as np
from keras.applications.efficientnet_v2 import EfficientNetV2B3
from keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    BatchNormalization,
    Dropout,
)
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_sample_weight
from datetime import datetime

# Options
training_history_saving_path = (
    "../../../models/training_history/CNN_training_history.joblib"
)
preprocessed_images_folder = "../../../data/processed/images"
class_weights_path = "../../../models/trained_processing_utils/class_weights.joblib"
model_export_path = "../../../models/image_classifier.h5"
batch_size = 32
epochs = 1

# Declarations
date = datetime.now()

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    rotation_range=30,
    validation_split=0.2,
    fill_mode="nearest",
    height_shift_range=0.2,
)


# Utilisation de l'API fonctionnelle de Keras
def create_image_model(input_shape, num_classes):
    base_model = EfficientNetV2B3(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)

    # Ajoutez des noms spécifiques aux couches Dense
    x = Dense(
        units=1024, activation="relu", kernel_regularizer="l2", name="dense_image_1"
    )(x)
    x = BatchNormalization(name="batch_norm_image_1")(x)
    x = Dropout(rate=0.5, name="dropout_image_1")(x)
    x = Dense(
        units=512, activation="relu", kernel_regularizer="l2", name="dense_image_2"
    )(x)
    x = BatchNormalization(name="batch_norm_image_2")(x)
    x = Dropout(rate=0.5, name="dropout_image_2")(x)

    predictions = Dense(units=num_classes, activation="softmax", name="output_image")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


# Chargement des données
training_set = train_datagen.flow_from_directory(
    directory=f"{preprocessed_images_folder}/train",
    class_mode="categorical",
    target_size=(200, 200),
    keep_aspect_ratio=True,
    batch_size=batch_size,
    subset="training",  # Utilisation de la partie d'entraînement
)

# Calcul des poids de classe
sample_weights = compute_sample_weight("balanced", y=training_set.classes)
class_weights = dict(enumerate(sample_weights))
joblib.dump(class_weights, class_weights_path)

# Exploration du jeu de données de validation
validation_set = train_datagen.flow_from_directory(
    directory=f"{preprocessed_images_folder}/train",
    class_mode="categorical",
    target_size=(200, 200),
    keep_aspect_ratio=True,
    batch_size=batch_size,
    subset="validation",  # Utilisation de la partie de validation
)

# Création du modèle image
image_model = create_image_model(input_shape=(200, 200, 3), num_classes=27)

# Entraînement du modèle image avec l'optimiseur Adam
image_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Callbacks
callbacks = [
    ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.2, patience=3, min_lr=0.001, min_delta=0.04
    )
]

# Entraînement du modèle
history = image_model.fit(
    training_set,
    epochs=epochs,
    validation_data=validation_set,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Export du modèle image
image_model.save(model_export_path)

print("Saved image model to disk.")
print("Success!")
print("Training time: {}".format(datetime.now() - date))
