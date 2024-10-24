import numpy as np
import cv2 as cv


def find_parnorama_size(image1, image2, H):
    """ Find the size of panorama image
    Args:
        image1 (numpy array): (h1, w1, 3) target image
        image2 (numpy array): (h2, w2, 3) source image
        H (numpy array): (3, 3) matrix to perform geomtric transformation (image2 -> image1)
    Returns:
        warped_image1 (numpy array): (h, w, 3) warped target image
        warped_mask1 (numpy array): (h, w) mask for warped target image
        warped_dt1 (numpy array): (h, w) distance transformed mask for warped target image
        warped_image2 (numpy array): (h, w, 3) warped source image
        warped_mask2 (numpy array): (h, w) mask for warped source image
        warped_dt2 (numpy array): (h, w) distance transformed mask for warped source image
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv.perspectiveTransform(corners2, H)
    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
    return xmax, xmin, ymax, ymin


