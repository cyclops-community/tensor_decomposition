from PIL import Image
import os
import numpy as np
from array import array
from os.path import join
import wget
from zipfile import ZipFile


def download_unzip_data(urls, zip_names, data_dir):
    try:
        os.stat(data_dir)
    except:
        os.mkdir(data_dir)

    for zip_name, url in zip(zip_names, urls):
        file_dir = join(data_dir, zip_name)
        if not os.path.isfile(file_dir):
            wget.download(url, data_dir)
            zf = ZipFile(file_dir, 'r')
            zf.extractall(data_dir)
            zf.close()


def load_images_from_folder(folder):
    print("Loading images from the dataset ......")
    pix_val = []
    for filename in os.listdir(folder):
        img = Image.open(folder + filename)
        if img is not None:
            pix = list(img.getdata())
            pix_val.append(pix)
        img.close()
    return np.asarray(pix_val)
