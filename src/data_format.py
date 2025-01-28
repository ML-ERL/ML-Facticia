import json

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
            "text": self.text,
        }


def from_json(json_file:str) -> list[DataFormat]:

    data = {}
    crops = []
    
    with open(json_file, "r") as f:
            data = json.load(f)

    for value in data.values():
        crop_name = value["filename"]
        crop_type = value["type"]
        crop_text = value["text"]
        crop_xywhr = [
            float(value["x"]),
            float(value["y"]),
            float(value["w"]),
            float(value["h"]),
            float(value["r"]),
        ]
        crops.append(DataFormat(crop_name, crop_xywhr, crop_type, crop_text))

    return crops