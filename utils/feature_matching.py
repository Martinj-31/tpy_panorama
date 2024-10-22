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


def compute_distance(descriptors1, descriptors2):
    """ Compute squared Euclidean the distances between descriptors
    Args:
        descriptors1 (numpy array): (N, D)
        descriptors2 (numpy array): (M, D)
    Returns:
        distances (numpy array): (N, M)
    """
    d1_sum_of_squares = np.sum(np.square(descriptors1), axis=1).reshape(-1, 1)
    d2_sum_of_squares = np.sum(np.square(descriptors2), axis=1).reshape(1, -1)

    dot_products = np.dot(descriptors1, descriptors2.T)

    distances = d1_sum_of_squares + d2_sum_of_squares - 2 * dot_products
    
    return distances


def distance_test(distances, indices, threshold):
    """ Check the nearest distance to find matched local features
    Args:
        distances (numpy array): (N, M) contains the distances betweend descriptors
        indices (numpy array): (N, ) contains matched indices, -1 = not matched
        threshold (float): threshold to find matched local features
    Returns:
        indices (numpy array): (N, ) contains matched indices, -1 = not matched
    """

    min_indices = np.argmin(distances, axis=1)
    min_distances = distances[np.arange(distances.shape[0]), min_indices]

    for i in range(len(min_distances)):
        if min_distances[i] < threshold:
            indices[i] = min_indices[i]

    return indices


def ratio_test(distances, indices, threshold):
    """ Check the ratio between the nearest distance and the second nearest distance
    Args:
        distances (numpy array): (N, M) contains the distances betweend descriptors
        indices (numpy array): (N, ) contains matched indices, -1 = not matched
        threshold (float): threshold to validate matched local features
    Returns:
        indices (numpy array): (N, ) contains matched indices, -1 = not matched
    """
    
    for i in range(distances.shape[0]):
        if indices[i] == -1:
            continue

        sorted_distances = np.sort(distances[i, :])

        nearest = sorted_distances[0]
        second_nearest = sorted_distances[1]

        if second_nearest == 0:
            ratio = 1
        else:
            ratio = nearest / second_nearest

        if ratio > threshold:
            indices[i] = -1
    
    return indices


def consistency_test(distances, indices):
    """ Check the consistency between matched local features
    Args:
        distances (numpy array): (N, M) contains the distances betweend descriptors
        indices (numpy array): (N, ) contains matched indices, -1 = not matched
    Returns:
        indices (numpy array): (N, ) contains matched indices, -1 = not matched
    """
    reverse_indices = np.argmin(distances, axis=0)

    for i in range(len(indices)):
        if indices[i] == -1:
            continue
        
        if reverse_indices[indices[i]] != i:
            indices[i] = -1
            
    return indices