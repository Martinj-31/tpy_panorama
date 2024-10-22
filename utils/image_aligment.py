import numpy as np
import random
import math


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