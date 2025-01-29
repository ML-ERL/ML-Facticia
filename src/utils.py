import numpy as np

class BoundingBox:
    def __init__(self, x, y, w, h, label):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label


class Node:
    def __init__(self, bounding_box):
        self.bounding_box = bounding_box
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def pre_order(self):
        return sum([child.pre_order() for child in self.children], [self.bounding_box])

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