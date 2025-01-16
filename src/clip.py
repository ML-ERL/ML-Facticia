from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

class CLIPInstance:
    
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
    def get_relation(self, image_path, texts):
        image = Image.open(image_path)
        
        # inputs = self.processor(text=text, images=image, return_tensors=True)
        inputs = self.processor(text=texts, images=image, return_tensors='pt', padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logis_per_image = outputs.logits_per_image # getting image-text similarity score
        probs = logis_per_image.softmax(dim=1).cpu().numpy() # converting to probabilities
        
        return probs
            
        