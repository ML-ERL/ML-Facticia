import numpy as np
import json

def calculate_corners(x, y, w, h, r):
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


def calculate_midpoint(p1, p2):
    return (p1 + p2) / 2

def calculate_midpoints(corners):
    midpoints = [
        calculate_midpoint(corners[0], corners[1]),  # (x1, y1) y (x2, y2)
        calculate_midpoint(corners[0], corners[3]),  # (x1, y1) y (x4, y4)
        calculate_midpoint(corners[1], corners[2]),  # (x2, y2) y (x3, y3)
        calculate_midpoint(corners[3], corners[2])   # (x4, y4) y (x3, y3)
    ]
    return midpoints

def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def associate_bounding_boxes(images, captions):
    associations = []

    for img in images:
        img_corners = calculate_corners(img['x'], img['y'], img['w'], img['h'], img['r'])
        img_midpoints = calculate_midpoints(img_corners)

        closest_caption = None
        min_distance = float('inf')

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

            if min_cap_distance < min_distance:
                closest_caption = cap
                min_distance = min_cap_distance

        associations.append({
            'image': img['filename'],
            'caption': closest_caption['filename'] if closest_caption else None,
            'distance': min_distance if closest_caption else None
        })

    return associations



file_path1 = '/Users/claudia/Downloads/6b3534ea-ohcbh_cf_erl8_217/6b3534ea-ohcbh_cf_erl8_217/6b3534ea-ohcbh_cf_erl8_217.json'  # Reemplaza con la ruta a tu archivo
file_path2 = '/Users/claudia/Downloads/6b3534ea-ohcbh_cf_erl8_217/07c212d0-ohcbh_cf_erl5_045/07c212d0-ohcbh_cf_erl5_045.json'  # Reemplaza con la ruta a tu archivo
file_path3 = '/Users/claudia/Downloads/6b3534ea-ohcbh_cf_erl8_217/8199f721-ohcbh_cf_erl5_063/8199f721-ohcbh_cf_erl5_063.json'  # Reemplaza con la ruta a tu archivo

with open(file_path3, "r") as archivo:
    data = json.load(archivo)

imagenes = []
captions = []

for key, value in data.items():
    if isinstance(value, dict):  
        type = value.get("type", None)
        filename = value.get("filename", None)
        x = value.get("x", None)
        y = value.get("y", None)
        w = value.get("w", None)
        h = value.get("h", None)
        r = value.get("r", None)
        
        if type == 2:  
            imagenes.append({
                "filename": filename,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "r": r
            })
        elif type == 0:  
            captions.append({
                "filename": filename,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "r": r
            })

max_distance = 250

associations = associate_bounding_boxes(imagenes, captions)

for assoc in associations:
    if (assoc['caption'] != None):
        print(f"Imagen: {assoc['image']} -> Caption: {assoc['caption']} (Distancia: {assoc['distance']:.2f})")

