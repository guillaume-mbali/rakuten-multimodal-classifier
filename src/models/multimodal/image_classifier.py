# This script initialize the image classifier and its dependencies.
# Requires :
# - Trained image model weights
# Last revision : Mathis Doyon - 08/12/2023


###############################################################################
#                                   OPTIONS                                   #
###############################################################################

# model_weights_path = "../../../models/trained_processing_utils/class_weights.joblib"

###############################################################################
#                                MODEL LOADING                                #
###############################################################################

# from keras.saving import load_model

# model = load_model(model_weights_path)
import os
from keras.models import load_model

model = os.path.abspath("../../../models/image_classifier.h5")
text_model = os.path.abspath("../../../models/lstm_classifier.h5")

# try:
model = load_model(model)
print("Modèle d'image chargé avec succès.")
# except Exception as e:
#     print(f"Erreur lors du chargement du modèle d'image : {e}")


###############################################################################
#                              MODEL ARCHITECTURE                             #
###############################################################################

# from keras.applications.efficientnet_v2 import EfficientNetV2B3
# from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
# from keras.models import Sequential

# base_model = EfficientNetV2B3(
#     weights="imagenet", include_top=False, input_shape=(200, 200, 3)
# )
# for layer in base_model.layers:
#     layer.trainable = False
# model = Sequential()
# model.add(base_model)
# model.add(GlobalAveragePooling2D())
# model.add(Dense(units=1024, activation="relu"))
# model.add(BatchNormalization())
# model.add(Dropout(rate=0.2))
# model.add(Dense(units=512, activation="relu"))
# model.add(BatchNormalization())
# model.add(Dropout(rate=0.2))
# model.add(Dense(units=27, activation="softmax"))
# model.load_weights(model_weights_path)
