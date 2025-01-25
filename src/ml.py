import os
import cv2
import numpy as np
from PIL import Image
import pytesseract

def extract_text(cropped_image):
    
    cropped_image.save('temp.png')
    
    text = pytesseract.image_to_string('temp.png')
    
    os.remove('temp.png')
    
    return text