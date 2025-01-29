from ultralytics import YOLO
import numpy as np

from src.clip import CLIPInstance
from src.data_format import DataFormat, from_json
from src.dataset_loader import get_dataset_from_file, clear
from src.image_processor import extract_image
from src.text_processor import extract_text

import json, os


class FullModel:

    def __init__(self, model="yolo11n-obb.pt"):

        self.yolo_model = YOLO(model)
        self.clip_model = CLIPInstance()

    def train(self, data_path="./dataset/facticia", dataset="./dataset/data.yaml"):

        clear()
        get_dataset_from_file(data_path)

        self.model.train(
            data=dataset,
            project="C:\Osvaldo/4.1/Machine Learning/ML-Facticia/dataset/training",
            name="cf_erl",
        )

    def run(self, export_path: str, data_path: str, images: list[str]|None = None, load_mode=False):

        if images is None:
            images = os.listdir(data_path)

        if not load_mode:
            for image in images:
                self.crop_image(export_path, data_path, image)

        self.cropped_images = {}

        for filename in images:

            json_file = f"{export_path}/{filename}/{filename}.json"
            crops = from_json(json_file)

            self.cropped_images[filename] = crops

    def crop_image(self, export_path: str, data_path: str, image: str):

        image_path = f"{export_path}/{image}"
        filename = image

        result = self.yolo_model(image_path)

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

    def get_proximity(self, export_path, images:list[str]|None=None):

        proximity = {}
        
        if images is None:
            images = os.listdir(export_path)

        for image in images:
            
            texts = [x.text for x in self.cropped_images[image] if x.type == 0 or x.type == 3]
            image_crops = [x for x in self.cropped_images[image] if x.type == 2]
            
            
            for crop in image_crops:
                
                image_path = f"{export_path}/{image}/{crop.filename}.jpg"
                proximity[crop] = self.clip_model.get_relation(image_path, texts)
                
        return proximity



    def calculate_corners(self, x, y, w, h, r):
        x, y, w, h, r = float(x), float(y), float(w), float(h), float(r)
        r = np.deg2rad(r)

        corners = np.array([
            [-w / 2, -h / 2], 
            [ w / 2, -h / 2],
            [ w / 2,  h / 2],
            [-w / 2,  h / 2]   
        ])

        rotation_matrix = np.array([
            [np.cos(r), -np.sin(r)],
            [np.sin(r),  np.cos(r)]
        ])

        rotated_corners = np.dot(corners, rotation_matrix.T) + [x, y]
        return rotated_corners


    def calculate_midpoint(self, p1, p2):
        return (p1 + p2) / 2

    def calculate_midpoints(self,corners):
        midpoints = [
            self.calculate_midpoint(corners[0], corners[1]),  # (x1, y1) y (x2, y2)
            self.calculate_midpoint(corners[0], corners[3]),  # (x1, y1) y (x4, y4)
            self.calculate_midpoint(corners[1], corners[2]),  # (x2, y2) y (x3, y3)
            self.calculate_midpoint(corners[3], corners[2])   # (x4, y4) y (x3, y3)
        ]
        return midpoints

    def calculate_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)


    def associate_bounding_boxes(self):
        associations = {}
        max_distance = 465
        images = []
        captions = []

        for crop in self.cropped_images:
            crop_xywhr = crop.xywhr

            if crop.type == 2:
                images.append({
                    "filename": crop.filename,
                    "x": crop_xywhr[0],
                    "y": crop_xywhr[1],
                    "w": crop_xywhr[2],
                    "h": crop_xywhr[3],
                    "r": crop_xywhr[4]
                })
            elif crop.type == 0:
                captions.append({
                    "filename": crop.filename,
                    "x": crop_xywhr[0],
                    "y": crop_xywhr[1],
                    "w": crop_xywhr[2],
                    "h": crop_xywhr[3],
                    "r": crop_xywhr[4]
                })


        for img in images:
            img_corners = self.calculate_corners(img['x'], img['y'], img['w'], img['h'], img['r'])
            img_midpoints = self.calculate_midpoints(img_corners)

            possible_captions = []

            for cap in captions:
                cap_corners = self.calculate_corners(cap['x'], cap['y'], cap['w'], cap['h'], cap['r'])
                cap_midpoints = self.calculate_midpoints(cap_corners)

                distances = [
                    self.calculate_distance(img_midpoints[3], cap_midpoints[0]),  
                    self.calculate_distance(img_midpoints[0], cap_midpoints[3]),  
                    self.calculate_distance(img_midpoints[2], cap_midpoints[1]),  
                    self.calculate_distance(img_midpoints[1], cap_midpoints[2])  
                ]

                min_cap_distance = min(distances)

                if min_cap_distance <= max_distance:
                    possible_captions.append({
                        'caption': cap['filename'],
                        'text_caption': cap['text'],
                        'distance': min_cap_distance
                    })

            associations[img['filename']] = possible_captions

        return associations