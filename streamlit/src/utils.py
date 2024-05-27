import os
import boto3
from botocore.exceptions import NoCredentialsError
from io import BytesIO
from PIL import Image
import streamlit as st
import random
import cv2
import numpy as np
from PIL import Image


# AWS S3 Configuration setup using environment variables
ACCESS_KEY = "SCWDNQQSZNC6Y41QQ9YH"
SECRET_KEY = "b1fedb95-8553-4cba-b5d6-5407cce19630"
REGION = "fr-par"
ENDPOINT_URL = f"https://s3.{REGION}.scw.cloud"
RAW_BUCKET_NAME = "rakuten"


# Singleton pattern for S3 client
def get_s3_client():
    session = boto3.session.Session()
    return session.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
        region_name=REGION,
    )


s3 = get_s3_client()


def get_sample_image_keys(bucket_name, n=10):
    paginator = s3.get_paginator("list_objects_v2")
    all_keys = []
    for page in paginator.paginate(Bucket=bucket_name):
        all_keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return random.sample(all_keys, min(n, len(all_keys))) if all_keys else []


def upload_to_scaleway(local_folder, bucket_name):
    try:
        for filename in os.listdir(local_folder):
            if filename.endswith((".jpg", ".png")):
                file_path = os.path.join(local_folder, filename)
                with open(file_path, "rb") as file_data:
                    s3.upload_fileobj(file_data, bucket_name, filename)
                    print(f"{filename} uploaded successfully.")
    except NoCredentialsError:
        st.error("Credentials not available")
    except Exception as e:
        st.error(f"An error occurred: {e}")


def get_image_from_s3(bucket, key):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return Image.open(BytesIO(response["Body"].read()))
    except Exception as e:
        st.error(f"Failed to retrieve image {key} from bucket {bucket}: {e}")
        return None


def load_new_images():
    if "img_exploration_data" not in st.session_state:
        st.session_state.img_exploration_data = []
    sample_keys = get_sample_image_keys(RAW_BUCKET_NAME, n=10)
    for key in sample_keys:
        img = get_image_from_s3(RAW_BUCKET_NAME, key)
        if img:
            st.session_state.img_exploration_data.append(img)


def display_images():
    if len(st.session_state.img_exploration_data) > 0:
        image = st.session_state.img_exploration_data[st.session_state.img_id]
        st.image(image, caption=f"Image {st.session_state.img_id + 1}")
    else:
        st.warning("Aucune image à afficher.")


def process_image(image):
    """
    Process an input image for feature extraction, focusing and standardizing the subject.

    Parameters:
    - image (PIL Image): The original image input.

    Returns:
    - original_image (PIL Image): The original image after conversion from BGR to RGB.
    - img_contrasted_pil (PIL Image): The thresholded image highlighting features.
    - bounding_box_image (PIL Image): The original image with a bounding box drawn around the largest contour.
    - processed_image (PIL Image): The cropped and resized image centered on the subject.
    """
    image = get_image_from_s3("rakuten", image)
    # Convert PIL Image to NumPy array for OpenCV processing
    image_np = np.array(image)
    # Convert RGB (PIL) to BGR (OpenCV)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    img_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding for dynamic contrast adjustment
    img_contrasted = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours from the thresholded image
    contours, _ = cv2.findContours(
        img_contrasted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Find the largest contour assumed to be the subject
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw a bounding box around the largest contour on the original image
        bounding_box_image = image_np.copy()
        cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop and resize the image to a standard size (e.g., 200x200 pixels)
        processed_image = image_np[y : y + h, x : x + w]
        processed_image = cv2.resize(
            processed_image, (200, 200), interpolation=cv2.INTER_AREA
        )
    else:
        # Default to the original image if no contours are found
        bounding_box_image = image_np.copy()
        processed_image = image_np

    # Convert the images back to PIL for compatibility with Streamlit
    img_contrasted_pil = Image.fromarray(img_contrasted)
    bounding_box_image = Image.fromarray(
        cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB)
    )
    processed_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

    return img_contrasted_pil, bounding_box_image, processed_image


if "img_exploration_data" not in st.session_state:
    load_new_images()


def fetch_image(image):
    image = get_image_from_s3("rakuten", image)
    # Convert PIL Image to NumPy array for OpenCV processing
    image_np = np.array(image)
    # Convert RGB (PIL) to BGR (OpenCV)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    return image
