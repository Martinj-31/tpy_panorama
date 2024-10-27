import numpy as np
import cv2 as cv

def create_binary_blended_mask(warped_mask1, warped_dt1, warped_mask2, warped_dt2):
    h, w = warped_mask1.shape[:2]
    blend_mask = np.zeros_like(warped_mask1)
    
    for i in range(h):
        for j in range(w):
            if warped_mask1[i, j] > 0 and warped_mask2[i, j] > 0:
                blend_mask[i, j] = warped_dt2[i][j] / (warped_dt1[i][j] + warped_dt2[i][j] + 1e-5)  # 두 이미지의 겹치는 부분을 블렌딩
            elif warped_mask1[i, j] > 0:
                blend_mask[i, j] = 1
            elif warped_mask2[i, j] > 0:
                blend_mask[i, j] = 0
    
    return blend_mask


def binary_blending(warped_image1, warped_mask1, warped_dt1, warped_image2, warped_mask2, warped_dt2):
    blend_mask = create_binary_blended_mask(warped_mask1, warped_dt1, warped_mask2, warped_dt2)
    blended_image = blend_mask * warped_image1 + (1. - blend_mask) * warped_image2
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image


def create_alpha_blended_mask(warped_mask1, warped_dt1, warped_mask2, warped_dt2):
    h, w = warped_mask1.shape[:2]
    blend_mask = np.zeros_like(warped_mask1)
    feather_radius = 30

    max_dt = max(np.max(warped_dt1), np.max(warped_dt2))
    
    norm_dt1 = warped_dt1 / (max_dt + 1e-5)
    norm_dt2 = warped_dt2 / (max_dt + 1e-5)

    for i in range(h):
        for j in range(w):
            if warped_mask1[i, j] > 0 and warped_mask2[i, j] > 0:
                dt1 = norm_dt1[i, j]
                dt2 = norm_dt2[i, j]
                
                alpha = dt2 / (dt1 + dt2 + 1e-5)
                
                distance = abs(dt1 - dt2)
                if distance < feather_radius:
                    feather_weight = distance / feather_radius
                    alpha = alpha * feather_weight + (1 - feather_weight) * (1 - alpha)
                
                blend_mask[i, j] = alpha
            elif warped_mask1[i, j] > 0:
                blend_mask[i, j] = 1
            elif warped_mask2[i, j] > 0:
                blend_mask[i, j] = 0

    return blend_mask


def alpha_blending(warped_image1, warped_mask1, warped_dt1, warped_image2, warped_mask2, warped_dt2):
    blend_mask = create_alpha_blended_mask(warped_mask1, warped_dt1, warped_mask2, warped_dt2)
    blended_image = blend_mask * warped_image1 + (1. - blend_mask) * warped_image2
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image