import random
import glob
import numpy as np
from PIL import Image
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import numpy


def generate_output_images(output, path):
    r = output[:, :, 0]
    g = output[:, :, 1]
    b = output[:, :, 2]

    r = np.clip(r, 0, 2000)
    g = np.clip(g, 0, 2000)
    b = np.clip(b, 0, 2000)

    rgb = np.dstack((r, g, b))
    rgb = rgb - np.nanmin(rgb)

    if np.nanmax(rgb) == 0:
        rgb = 255 * np.ones_like(rgb)
    else:
        rgb = 255 * (rgb / np.nanmax(rgb))

    rgb[np.isnan(rgb)] = np.nanmean(rgb)
    rgb = rgb.astype(np.uint8)
    save_image = Image.fromarray(rgb)
    save_image.save(path)


def load_file(path, dtype=np.float32):
    if (path[-4:] == '.tif'):

        return np.array(tiff.imread(path), dtype)


path = './Sen2_MTC'

for tile_name in os.listdir(path):
    tile_path = os.path.join(path, tile_name)
    for type in os.listdir(tile_path):
        type_path = os.path.join(tile_path, type)
        save_path = os.path.join('./Sen2_MTC_RGB', tile_name, type)
        os.makedirs(save_path, exist_ok=True)
        for image_name in os.listdir(type_path):
            image_path = os.path.join(type_path, image_name)

            image = load_file(image_path)
            r = image[:, :, 0]
            g = image[:, :, 1]
            b = image[:, :, 2]
            rgb = np.dstack((r, g, b))
            generate_output_images(rgb, os.path.join(
                save_path, image_name[:-4] + '.png'))
