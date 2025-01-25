class DataFormat:
    
    def __init__(self, filename, xywhr, type, text=""):
        
        self.filename = filename
        self.xywhr = xywhr
        self.type = type
        self.text = text
        
    def to_dict(self):
        
        x, y, w, h, r = self.xywhr.cpu().numpy()
        
        return {
            "filename": self.filename,
            "x": str(x),
            "y": str(y),
            "w": str(w),
            "h": str(h),
            "r": str(r),
            "type": int(self.type.item()),
            "text": self.text
        }