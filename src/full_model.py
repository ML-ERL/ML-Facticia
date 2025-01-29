from ultralytics import YOLO
import numpy as np

from src.utils import *
from src.clip import CLIPInstance
from src.data_format import DataFormat, from_json
from src.dataset_loader import get_dataset_from_file, clear
from src.image_processor import crop_image

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
            images = [x.split('.')[-2] for x in os.listdir(data_path) if x.split('.')[-1] == 'jpg']
            

        if not load_mode:
            for image in images:
                print(f"Cropping image {image}")
                crop_image(self.yolo_model, export_path, data_path, image)

        self.cropped_images = {}

        for filename in images:

            json_file = f"{export_path}/{filename}/{filename}.json"
            crops = from_json(json_file)

            self.cropped_images[filename] = crops
       

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
            img_corners = calculate_corners(img['x'], img['y'], img['w'], img['h'], img['r'])
            img_midpoints = calculate_midpoints(img_corners)

            possible_captions = []

            for cap in captions:
                cap_corners = calculate_corners(cap['x'], cap['y'], cap['w'], cap['h'], cap['r'])
                cap_midpoints = calculate_midpoints(cap_corners)

                distances = [
                    calculate_distance(img_midpoints[3], cap_midpoints[0]),  
                    calculate_distance(img_midpoints[0], cap_midpoints[3]),  
                    calculate_distance(img_midpoints[2], cap_midpoints[1]),  
                    calculate_distance(img_midpoints[1], cap_midpoints[2])  
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