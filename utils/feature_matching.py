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


def match_features(
        keypoints1, descriptors1, keypoints2, descriptors2, 
        distance_threshold=0.25, ratio_threshold=0.7):
    """ Match local features
    Args:
        keypoints1 (opencv KeyPoint list): (N, ) keypoints from image1
        descriptors1 (numpy array): (D, ) descriptors for keypoints1
        keypoints2 (opencv KeyPoint list): (M, ) keypoints from image2
        descriptors2 (numpy array): (D, ) descriptors for keypoints2
        distance_threshold(float): threshold to determine matched points
        ratio_threshold(float): threshold to validate matched points
    Returns:
        matches (list): list of matched keypoints. 
            Each element of the list is ((point1[0], point1[1]), (point2[0], point2[1])). 
            Here, point1 and point2 are matched local feature. point1[0] and point1[1] 
            are x and y coordinates of point1. point2[0] and point2[1] are x and y 
            coordinates of point2.
    """
    distances = compute_distance(descriptors1, descriptors2)
    indices = -1 * np.ones((distances.shape[0],), dtype=np.int64)

    indices = distance_test(distances, indices, distance_threshold)
    indices = ratio_test(distances, indices, ratio_threshold)
    indices = consistency_test(distances, indices)

    matches = []
    for i, j in enumerate(indices):
        if j == -1:
            continue
        matches.append([keypoints1[i].pt, keypoints2[j].pt])

    return matches