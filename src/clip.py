from PIL import Image
import torch
import os
from transformers import CLIPProcessor, CLIPModel

class CLIPInstance:
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", local_dir="local_clip"):
        
        self.local_dir = local_dir
        self.model_name = model_name
        
        os.makedirs(local_dir, exist_ok=True)
        
        if not os.path.isfile(os.path.join(local_dir, 'config.json')):
            
            print("Downloading model and processor")
            
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)
            
            self.processor.save_pretrained(local_dir)
            self.model.save_pretrained(local_dir)
 
        else:
            
            print("Loading model and processor form local storage")
            
            self.processor = CLIPProcessor.from_pretrained(local_dir)
            self.model = CLIPModel.from_pretrained(local_dir)
        
        
    def get_relation(self, image_path, texts):
        image = Image.open(image_path)
        
        inputs = self.processor(text=texts, images=image, return_tensors='pt', padding=True, truncation=True, max_length=77)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logis_per_image = outputs.logits_per_image # getting image-text similarity score
        probs = logis_per_image.softmax(dim=1).cpu().numpy() # converting to probabilities
        
        return probs
    
    
            
    