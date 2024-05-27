import os
import boto3

ACCESS_KEY = "SCWDNQQSZNC6Y41QQ9YH"
SECRET_KEY = "b1fedb95-8553-4cba-b5d6-5407cce19630"
BUCKET_NAME = "rakuten"


REGION = "fr-par"

session = boto3.session.Session()
s3 = session.client(
    service_name="s3",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    endpoint_url=f"https://s3.{REGION}.scw.cloud",
    region_name=REGION,
)

local_folder = "/Users/guimb/Documents/Data/raw/images/image_train"


for filename in os.listdir(local_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        file_path = os.path.join(local_folder, filename)
        with open(file_path, "rb") as file_data:
            s3.upload_fileobj(file_data, BUCKET_NAME, filename)
            print(f"{filename} a été téléchargé avec succès.")
