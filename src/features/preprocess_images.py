import boto3
import streamlit.src.config as config
from botocore.exceptions import NoCredentialsError
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


s3_client = boto3.client(
    "s3",
    endpoint_url=config.SCW_ENDPOINT_URL,
    aws_access_key_id=config.SCW_ACCESS_KEY,
    aws_secret_access_key=config.SCW_SECRET_KEY,
    region_name="fr-par",
)


def preprocess_image_from_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, img_contrasted = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        img_contrasted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img_cropped = img_bgr[y : y + h, x : x + w]
        return cv2.resize(img_cropped, (500, 500))
    return img_bgr


def upload_image(bucket_name, img_data, object_name):
    try:
        # Create a BytesIO object
        buffer = BytesIO()

        # Encode the image data to JPEG format and write it to the buffer
        is_success, encoded_image = cv2.imencode(".jpg", img_data)
        if not is_success:
            raise ValueError("Could not encode image to JPEG")

        # Write the encoded image to the BytesIO buffer
        buffer.write(encoded_image)

        # IMPORTANT: Reset the buffer's position to the beginning after writing
        buffer.seek(0)

        # Upload the image from the buffer
        s3_client.upload_fileobj(buffer, bucket_name, object_name)
        print(f"Uploaded {object_name} to {bucket_name}")
    except Exception as e:
        print(f"Failed to upload {object_name}: {e}")


def main():
    start_time = datetime.now()
    df = pd.read_csv(config.raw_df_path)
    Y = np.genfromtxt(config.y_path, delimiter=",", skip_header=1, usecols=1, dtype=int)
    df["product_code"] = Y

    process_images()

    print(f"Preprocessing completed in {datetime.now() - start_time}")


def process_images():
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=config.RAW_BUCKET_NAME)

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    object_key = obj["Key"]
                    print(f"Processing {object_key}")

                    response = s3_client.get_object(
                        Bucket=config.RAW_BUCKET_NAME, Key=object_key
                    )

                    image_data = response["Body"].read()
                    processed_image = preprocess_image_from_bytes(image_data)

                    upload_image(
                        config.PROCESSED_BUCKET_NAME,
                        processed_image,
                        f"processed_{object_key}",
                    )
            else:
                print("No more contents found in bucket.")
    except Exception as e:
        print(f"Error processing images: {e}")


if __name__ == "__main__":
    main()
