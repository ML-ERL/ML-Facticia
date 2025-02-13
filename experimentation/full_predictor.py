import cv2, os
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt

os.getcwd().split('/')[-1] == "experimentation" and os.chdir('../')

# ----------------------------
# Step 1: Load Layout Model (PubLayNet)
# ----------------------------
def load_layout_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/PubLayNet/faster_rcnn_R_50_FPN_3x/137848164/model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # PubLayNet classes: text, title, list, table, figure
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Detection threshold
    return DefaultPredictor(cfg)

# ----------------------------
# Step 2: Load SAM Model
# ----------------------------
def load_sam_model(sam_checkpoint_path):
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path)
    return SamPredictor(sam)

# ----------------------------
# Step 3: Detect Regions and Segment with SAM
# ----------------------------
def segment_regions(image_path, layout_predictor, sam_predictor):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()

    # Detect layout regions with PubLayNet
    outputs = layout_predictor(image)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()

    # Prepare visualization
    metadata = MetadataCatalog.get("pub_lay_net_train")
    v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
    out_layout = v.draw_instance_predictions(instances)
    layout_vis = out_layout.get_image()[:, :, ::-1]

    # Segment regions with SAM
    segmented_regions = []
    for i, (box, cls) in enumerate(zip(boxes, classes)):
        x1, y1, x2, y2 = map(int, box)
        region = image[y1:y2, x1:x2]

        # Skip small regions
        if region.size == 0:
            continue

        # Segment with SAM
        sam_predictor.set_image(region)
        masks, _, _ = sam_predictor.predict()

        # Save segmented regions and masks
        for j, mask in enumerate(masks):
            masked_region = cv2.bitwise_and(region, region, mask=mask.astype(np.uint8))
            segmented_regions.append({
                "class": "text" if cls in [0, 1] else "image",
                "region": masked_region,
                "mask": mask,
                "bbox": (x1, y1, x2, y2)
            })

            # Save segmented region
            cv2.imwrite(f"region_{i}_mask_{j}.png", cv2.cvtColor(masked_region, cv2.COLOR_RGB2BGR))

    # Visualize SAM masks on original image
    for region in segmented_regions:
        x1, y1, x2, y2 = region["bbox"]
        mask_full = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_full[y1:y2, x1:x2] = region["mask"]
        image[mask_full == 1] = [255, 0, 0]  # Highlight masks in red

    return layout_vis, image

# ----------------------------
# Step 4: Run the Pipeline
# ----------------------------
if __name__ == "main":
    
    
    # Paths (update these!)
    path = "./sam/images/"
    images = os.listdir(path)
    image_path = f"{path}{images[0]}"
    sam_checkpoint_path = "./sam/sam_vit_b_01ec64.pth"

    # Load models
    layout_predictor = load_layout_model()
    sam_predictor = load_sam_model(sam_checkpoint_path)

    # Process image
    layout_vis, segmented_vis = segment_regions(image_path, layout_predictor, sam_predictor)

    # Save and display results
    cv2.imwrite("layout_detection.jpg", cv2.cvtColor(layout_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite("segmented_output.jpg", cv2.cvtColor(segmented_vis, cv2.COLOR_RGB2BGR))

    # Show results
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title("Layout Detection")
    plt.imshow(layout_vis)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Segmented Regions (SAM)")
    plt.imshow(segmented_vis)
    plt.axis("off")
    plt.show()