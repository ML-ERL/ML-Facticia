import cv2
import numpy as np
from PIL import Image


def extract_image(image_path, xywhr_coordinates):
    img = Image.open(image_path)
    x, y, w, h, r = xywhr_coordinates[0].cpu().numpy()
    img.rotate(-r)
    cropped_img = img.crop((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return cropped_img
