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


def warp_images(image1, image2, H):
    """ Warp image1 and image2
    Args:
        image1 (numpy array): (h1, w1, 3) target image
        image2 (numpy array): (h2, w2, 3) source image
        H (numpy array): (3, 3) matrix to perform geomtric transformation (image2 -> image1)
    Returns:
        xmax, xmin, ymax, ymin: the possible coordinates
    """
    # Find panorama size
    xmax, xmin, ymax, ymin = find_parnorama_size(image1, image2, H)    

    # Warp image1
    h1, w1 = image1.shape[:2]
    warped_image1 = np.zeros((ymax - ymin, xmax - xmin, 3))
    warped_image1[-ymin:-ymin+h1, -xmin:-xmin+w1, :] = image1
    warped_mask1 = np.zeros((ymax - ymin, xmax - xmin, 1))
    warped_mask1[-ymin:-ymin+h1, -xmin:-xmin+w1, :] = 1.0

    # Warp image2
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    H = T @ H
    warped_image2 = np.zeros((ymax - ymin, xmax - xmin, 3))
    warped_mask2 = np.zeros((ymax - ymin, xmax - xmin, 1))
    warped_image2, warped_mask2 = backward_warp(image2, warped_image2, warped_mask2, H)

    # Distance transform
    warped_dt1 = distance_transform(warped_mask1)
    warped_dt2 = distance_transform(warped_mask2)

    return warped_image1, warped_mask1, warped_dt1, warped_image2, warped_mask2, warped_dt2


def backward_warp(image, warped_image, warped_mask, H):
    """ Backward warp for images and masks
    Args:
        image (numpy array): (h1, w1, 3) source image
        warped_image (numpy array): (h2, w2, 3) target image (to be filled)
        warped mask (numpy array): (h2, w2) mask for target image (to be filled)
        H (numpy array): (3, 3) matrix to perform geomtric transformation (image->warped_iamge)
    Returns:
        warped_image (numpy array): (h2, w2, 3) target image
        warped_mask (numpy array): (h, w) mask for target image
    """
    h, w = warped_image.shape[:2]
    
    for y in range(h):
        for x in range(w):
            target_coords = np.array([x, y, 1])
            source_coords = np.linalg.inv(H) @ target_coords
            source_coords /= source_coords[2]
            
            sx, sy = int(source_coords[0]), int(source_coords[1])
            
            if 0 <= sx < image.shape[1] and 0 <= sy < image.shape[0]:

                warped_image[y, x] = image[sy, sx]
                warped_mask[y, x] = 1.0

    return warped_image, warped_mask


def distance_transform(mask):
    dt = cv.distanceTransform(mask.astype(np.uint8), distanceType=cv.DIST_L2, maskSize=5)
    return dt

