from src.sam import SAM
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import os, cv2

# Change directory
os.getcwd().split('/')[-1] == "experimentation" and os.chdir('../')
matplotlib.use('Agg')

sam_model = SAM()

path = "./sam/images/"

images = os.listdir(path)

for image_name in images:
    
    # image = Image.open(image_path)
    image = cv2.imread(f"{path}{image_name}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    masks, scores, _ = sam_model.predict_image(image)
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5, cmap="viridis")
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.savefig(f"./sam/output/O{i}_{image_name}")
        plt.close()
    