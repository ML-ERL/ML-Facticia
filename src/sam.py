from segment_anything import sam_model_registry, SamPredictor

class SAM:
    def __init__(self):
        sam = sam_model_registry['vit_b'](checkpoint='./sam/sam_vit_b_01ec64.pth')
        self.predictor = SamPredictor(sam)

    def predict_image(self, image):
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            # point_coords=point_coords,
            # point_labels=point_labels,
            multimask_output=True,
        )
        
        return masks, scores, logits