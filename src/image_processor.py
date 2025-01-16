import cv2
import numpy as np

def extract_image(image_path, yolo_coordinates):
    
    img = cv2.imread(image_path)
    
    if img is None:
        return None
    
    height, width = img.shape[:2]
    
    box_points = np.array([
        [int(yolo_coordinates[0] * width), int(yolo_coordinates[1] * height)], 
        [int(yolo_coordinates[2] * width), int(yolo_coordinates[3] * height)], 
        [int(yolo_coordinates[4] * width), int(yolo_coordinates[5] * height)], 
        [int(yolo_coordinates[6] * width), int(yolo_coordinates[7] * height)]
    ])
    
    # Get bounding box of rotated points
    x_min = int(min(box_points[:, 0]))
    x_max = int(max(box_points[:, 0]))
    y_min = int(min(box_points[:, 1]))
    y_max = int(max(box_points[:, 1]))

    # Ensure coordinates are within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width - 1, x_max)
    y_max = min(height - 1, y_max)
    
    cropped_image = img[y_min:y_max + 1, x_min:x_max + 1]
    
    cropped_img = None
    
    return cropped_img