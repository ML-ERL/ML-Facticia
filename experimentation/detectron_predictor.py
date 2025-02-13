from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import cv2, os

os.getcwd().split('/')[-1] == "experimentation" and os.chdir('../')

# Load configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/PubLayNet/faster_rcnn_R_50_FPN_3x/137848164/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # PubLayNet has 5 classes: text, title, list, table, figure

# Set threshold for detection
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# Create predictor
predictor = DefaultPredictor(cfg)

path = "./sam/images/"
images = os.listdir(path)

# Load image
image = cv2.imread(f"{path}{images[0]}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

outputs = predictor(image)

# Get metadata for PubLayNet
metadata = MetadataCatalog.get("pub lay net_train")

# Visualize the results
v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Layout Detection", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()