import json, os
from PIL import Image

from src.data_format import DataFormat
from src.text_processor import extract_text


def extract_image(image_path, xywhr_coordinates):
    img = Image.open(image_path)
    x, y, w, h, r = xywhr_coordinates.cpu().numpy()
    img.rotate(-r)
    cropped_img = img.crop((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return cropped_img

def crop_image(yolo_model, export_path: str, data_path: str, image: str):

        image_path = f"{export_path}/{image}"
        filename = image

        result = yolo_model(image_path)

        obb = result[0].obb

        xywhr = obb.xywhr
        cls = obb.cls

        data = []

        os.makedirs(f"{export_path}/{filename}", exist_ok=True)

        # Save crops and create metadata
        for i in range(len(xywhr)):
            data.append(DataFormat(f"{filename}_{i}", xywhr[i], cls[i]))
            image = extract_image(image_path, xywhr[i])

            image.save(f"{export_path}/{filename}/{filename}_{i}.jpg")

        for d in data:
            if d.type == 3 or d.type == 0:
                d.text = extract_text(f"{export_path}/{filename}/{d.filename}.jpg")

        with open(f"{export_path}/{filename}/{filename}.json", "w") as f:
            json.dump({d.filename: d.to_dict() for d in data}, f, indent=4)