import numpy as np
import cv2 as cv

def extract_sift_feature(image, eps=1e-6):
    
    sift = cv.SIFT_create(nfeatures=0, 
                          nOctaveLayers=3, 
                          contrastThreshold=0.04, 
                          edgeThreshold=10, 
                          sigma=1.6)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    keypoints, descriptors = sift.detectAndCompute(image_gray, mask=None)
    
    if len(keypoints) == 0:
        return ([], None)
    
    descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
    descriptors = np.sqrt(descriptors)
    
    return keypoints, descriptors


def visualize_sift_feature(image, keypoints, name):
    output_image = cv.drawKeypoints(image, keypoints, None)
    
    cv.imshow('image', output_image)
    cv.imwrite(f"./data/detected_shift_feature_{name}.jpg", output_image)