# from tensorflow.keras.preprocessing import image
# import tensorflow as tf

# class Image2VecKeras:
    
#     def __init__(self):
#         self.model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, pooling='avg')
        
#     def extract_features(self, image_path, target_size=(420, 420)):
#         img = image.load_img(image_path, target_size=target_size)
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
        
#         features = self.model.predict(img_array)
#         return features.flatten()
    
import numpy as np
import cv2

class Image2VecOpenCV:
    
    def __init__(self):
        self.sift = cv2.SIFT_create()
    
    def extract_features(self, image_path):
        img = cv2.imread(image_path)
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.sift.detectAndCompute(grayscale, None)
        
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)
        
        cv2.imshow('SIFT Keypoints', img_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return descriptors