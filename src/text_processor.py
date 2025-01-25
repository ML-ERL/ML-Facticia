import os
import re
import cv2
import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def image_preprocessing(image):
    
    # grayscale image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # noise removal
    image = cv2.medianBlur(image, 5)
    
    # thresholding
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return image    

def osd(image):
    
    osd = pytesseract.image_to_osd(image)
    
    angle = re.search('(?<=Rotate: )\d+', osd).group(0)
    script = re.search('(?<=Script: )\d+', osd).group(0)
    
    return angle, script


def extract_text(image, lenguage="spa", verbose=False):
    
    image = cv2.imread(image)
    
    image = image_preprocessing(image)
    
    pil_image = Image.fromarray(image)
    
    # angle, script = osd(image)
    
    config = "--psm 6 --oem 1"
    
    text = pytesseract.image_to_string(pil_image, lang=lenguage, config=config)
    
    if verbose:
        print(text)
    
    return text