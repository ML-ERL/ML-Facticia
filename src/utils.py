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
        
