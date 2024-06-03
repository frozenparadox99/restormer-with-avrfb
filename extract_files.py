import os
import zipfile
import gdown

def download_file_from_google_drive(file_id, dest):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, dest, quiet=False)

def extract_zip_file(unzip_dir, zip_file_path):
    """
    Extracts the zip file into the specified directory.
    """
    os.makedirs(unzip_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)

# File ID from the Google Drive link
file_id = '1Y35DS9atsaE_ZH1mVnSApVEXVMQQS6jg'
destination = './WIPERNET.zip'

# Download the zip file
download_file_from_google_drive(file_id, destination)

# Extract the zip file
extract_zip_file('./Restormer', destination)
