"""Config file. Stores all the filepaths"""

import os
from dotenv import load_dotenv

load_dotenv()

# Bucket config

RAW_BUCKET_NAME = os.getenv("RAW_BUCKET_NAME")
PROCESSED_BUCKET_NAME = os.getenv("PROCESSED_BUCKET_NAME")
VALIDATION_SIZE = os.getenv("VALIDATION_SIZE")

SCW_ACCESS_KEY = os.getenv("SCW_ACCESS_KEY")
SCW_SECRET_KEY = os.getenv("SCW_SECRET_KEY")
SCW_ENDPOINT_URL = os.getenv("SCW_ENDPOINT_URL")

# Images
preprocessed_img_folder = "s3://{}/processed_images/{{}}".format(
    PROCESSED_BUCKET_NAME
)  # Using bucket path
raw_img_folder = "s3://{}/raw_images/".format(RAW_BUCKET_NAME)

# Dataframes
processing_df_path = "s3://rakuten-files/processed/cleaned_text.csv"
raw_df_path = "s3://rakuten-files/raw/X_train.csv"
y_path = "s3://rakuten-files/raw/Y_train.csv"

# Model weights
multimodal_model_path = "../models/multimodal_classifier.keras"
lstm_model_path = "../models/lstm_classifier.keras"
linearsvc_model_path = "data/linearsvc_model.joblib"

# Processing objects
vectorizer_path = "../models/trained_processing_utils/vectorizer.joblib"
label_encoder_path = "../models/trained_processing_utils/label_encoder.joblib"
tfidf_path = "data/tfidf_vectorizer.joblib"
