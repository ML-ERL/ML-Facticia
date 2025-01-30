import os
import re
import cv2
import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def img_inversion(image):
    return cv2.bitwise_not(image)

def img_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def img_threshold(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def img_blur(image):
    return cv2.medianBlur(image, 5)

def img_noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 5)
    
    return image

def img_dilation_and_erosion(image):
    image = cv2.bitwise_not(image)
    
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    image = cv2.bitwise_not(image)
    return image

def img_remove_borders(image):
    contours, heirarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    
    return image[y:y+h, x:x+w]
 

def image_preprocessing(image):
    
    # grayscale image
    image = img_grayscale(image)
    
    # noise removal
    image = img_blur(image)
    
    # thresholding
    image = img_threshold(image)
    
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
    
    return 


def parametric_search(images):
    
    results = []
    
    for image in images:
        results.append(parametric_preprocessing(image))
        
    return results


def parametric_preprocessing(image, lenguage="spa"):
    
    preprocessing_techniques = [
        img_inversion,
        img_grayscale,
        img_threshold,
        img_blur,
        img_noise_removal,
        img_dilation_and_erosion,
        img_remove_borders,
    ]
    
    results = []
    
    for i in range(2 ** len(preprocessing_techniques)):
        
        i_image = cv2.imread(image)
        
        for j in range(len(preprocessing_techniques)):
            if i & (1 << j):
                image = preprocessing_techniques[j](image)
                
        pil_image = Image.fromarray(i_image)
        
        config = "--psm 6 --oem 1"
    
        text = pytesseract.image_to_string(pil_image, lang=lenguage, config=config)
        
        results.append(text)
        
    return results