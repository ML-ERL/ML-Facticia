import json, os
from PIL import Image
import cv2
import numpy as np
from src.data_format import DataFormat
from src.text_processor import extract_text
from src.utils import preprocess_photography_image


def extract_image(original_image, bbox):
    print(bbox)
    points = bbox.cpu().numpy()
    x, y, w, h = cv2.boundingRect(points)
    cropped_img = original_image[y : y + h, x : x + w]
    return cropped_img


def crop_image(yolo_model, export_path: str, data_path: str, image: str, llm):

    image_path = f"{data_path}/{image}.jpg"
    filename = image

    preprocess_photography_image(image_path, f"{data_path}/gray_scale.jpg")

    result = yolo_model(f"{data_path}/gray_scale.jpg")

    os.remove(f"{data_path}/gray_scale.jpg")

    obb = result[0].obb
    bbox = obb.xyxyxyxy
    cls = obb.cls

    data = []

    os.makedirs(f"{export_path}/{filename}", exist_ok=True)

    for i in range(len(bbox)):
        image = cv2.imread(image_path)
        data.append(DataFormat(f"{filename}_{i}", bbox[i], cls[i]))
        image = extract_image(image, bbox[i])
        cv2.imwrite(f"{export_path}/{filename}/{filename}_{i}.jpg", image)

    for d in data:
        if d.type == 3 or d.type == 0:
            d.text = extract_text(f"{export_path}/{filename}/{d.filename}.jpg")
            d.text = llm.fix_text(d.text)

    with open(f"{export_path}/{filename}/{filename}.json", "w") as f:
        json.dump({d.filename: d.to_dict() for d in data}, f, indent=4)
