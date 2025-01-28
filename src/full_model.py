from ultralytics import YOLO
from clip import CLIPInstance

from src.data_format import DataFormat, from_json
from src.dataset_loader import get_dataset_from_file, clear
from src.image_processor import extract_image
from src.text_processor import extract_text

import json, os

class FullModel:
    
    def __init__(self, model=None):
        
        # Train model when pretrained model is not given
        if model == None:
            
            model = YOLO("yolo11n-obb.pt")
            
            data_path = "./dataset/facticia"
            dataset = "./dataset/data.yaml"
            
            clear()
            get_dataset_from_file(data_path)
            
            model.train(
                data=dataset,
                project="C:\Osvaldo/4.1/Machine Learning/ML-Facticia/dataset/training",
                name="cf_erl",
            )
            
        self.yolo_model = model
        self.clip_model = CLIPInstance()
            
    def run(self, export_path:str, data_path:str, images:list[str], load_mode=False):
        
        if not load_mode:
            for image in images:
                self.crop_image(export_path, data_path, image)
                
        self.cropped_images = {}
        
        for filename in images:
            
            json_file = f"{export_path}/{filename}/{filename}.json"
            crops = from_json(json_file)
            
            self.cropped_images[filename] = crops
                        
    def crop_image(self, export_path:str, data_path:str, image:str):
        
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