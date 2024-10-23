import numpy as np
import random
import math


def find_homography(matches, num_samples=4, max_iterations=500, inlier_threshold=0.5, max_inliers_ratio=0.8):
    """ Fine homography
    Args:
        matches (list): list of matched keypoints
        num_samples (int): the minimum number of matching to determine homography
        max_iterations (int): the maximum iteration of RANSAC
        inlier_threshold (float): threshold to determine inline in RANSAC
        max_inlier_ratio (float): threshold to stop RANSAC iterations
    Returns:
        H (numpy array): (3, 3) matrix to perform geomtric transformation
    """
    # Prepare Ax=b systems to find the homography paramters x
    inputs, targets = prepare_data(matches)

    # Find the homography using a RANSAC
    H = np.ones((9,))
    H[:8] = ransac(inputs, targets, num_samples, max_iterations, inlier_threshold, max_inliers_ratio)
    H = H.reshape((3, 3))
    return H


def prepare_data(matches):
    """ Prepare Ax=b systems to find the homography paramters x
    Args:
        matches (list): list of matched keypoints
    Returns:
        inputs (numpy array): (2K, 8) matrix A
        outputs (numpy array): (2K, ) vector b
    """
    num_matches = len(matches)
    inputs = np.zeros((2 * num_matches, 8))
    targets = np.zeros((2 * num_matches,))
    
    for i, match in enumerate(matches):
        x1, y1 = match[0]
        x2, y2 = match[1]

        inputs[2*i] = [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1]
        targets[2*i] = x2

        inputs[2*i+1] = [0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1]
        targets[2*i+1] = y2
        
    return inputs, targets


def ransac(inputs, targets, num_samples, max_iterations=100, inlier_threshold=0.2, max_inliers_ratio=0.9):
    """ RANdom SAmple Consensus (RANSAC)
        Perform RANSAC to find the best model x for Ax=b
        Then, perform least squares on the inliers of best model x to refine it 
    Args:
        inputs (numpy array): (2K, 8) matrix A
        outputs (numpy array): (2K, ) vector b
        num_samples (int): the minimum number of matching to determine homography
        max_iterations (int): the maximum iterations
        inlier_threshold (float): threshold to determine inlines
        max_inlier_ratio (float): threshold to stop iterations
    Returns:
        H (numpy array): (3, 3) matrix to perform geomtric transformation
    """
    best_model = None
    best_inliers = -math.inf
    max_inliers = int(targets.shape[0] * max_inliers_ratio)

    # RANSAC
    for _ in range(max_iterations):
        sample_inputs, sample_targets = random_sample(inputs, targets, num_samples)

        sample_model, _, _, _ = np.linalg.lstsq(sample_inputs, sample_targets, rcond=None)

        num_inliers = evaluate_model(inputs, targets, sample_model, inlier_threshold)

        if num_inliers > best_inliers:
            best_model = sample_model
            best_inliers = num_inliers

            if num_inliers > max_inliers:
                break
    
    # Least square refinement
    model = refine_model(inputs, targets, best_model, inlier_threshold)
    num_inliers = evaluate_model(inputs, targets, model, inlier_threshold)
    if num_inliers > best_inliers:
        best_model = model
        best_inliers = num_inliers

    return best_model


def random_sample(inputs, targets, num_samples):
    """ Randomly sample inputs and targets to determine model 
    Args:
        inputs (numpy array): (2K, 8) matrix
        targets (numpy array): (2K, ) vector
        num_samples (int): the number of random samples
    Returns:
        sampled_inputs (numpy array): (8, 8) matrix
        sampled_targets (numpy array): (8, ) vector
    """
    num_matches = len(targets) // 2
    indices = []
    for i in random.sample(range(num_matches), num_samples):
        indices.append(2 * i)
        indices.append(2 * i + 1)
    sampled_inputs = inputs[indices, :]
    sampled_targets = targets[indices]
    
    return sampled_inputs, sampled_targets


def fit_model(A, b, eps=1e-6):
    """ Find the model x for Ax=b, x=A^{-1}b 
    Args:
        A (numpy array): (8, 8) matrix
        b (numpy array): (8, ) vector
    Returns:
        x (numpy array): (8, ) vector, the solution for Ax=b
    """
    E = eps * np.eye(8)           # To avoid singular matrix
    x = np.linalg.inv(A + E) @ b
    return x