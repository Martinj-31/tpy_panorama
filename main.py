import argparse
import cv2 as cv
import numpy as np

from utils.feature_matching import extract_sift_feature, visualize_sift_feature, match_features
from utils.image_aligment import find_homography
from utils.image_warping import warp_images
from utils.image_blending import binary_blending, alpha_blending, laplacian_pyramid_blending


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1_path', type=str, default='data/uttower1.jpg')
    parser.add_argument('--image2_path', type=str, default='data/uttower2.jpg')
    
    args = parser.parse_args()
    return args



args = get_config()

# 이미지 불러오기
image1 = cv.imread(args.image1_path)
image2 = cv.imread(args.image2_path)

# 이미지의 keypoints와 descriptors 추출하기
keypoints1, descriptors1 = extract_sift_feature(image1)
keypoints2, descriptors2 = extract_sift_feature(image2)

visualize_sift_feature(image1, keypoints1, 'image1')
visualize_sift_feature(image2, keypoints2, 'image2')

# Match local features
matches = match_features(keypoints1, descriptors1, keypoints2, descriptors2, args.distance_threshold, args.ratio_threshold)

# Find homography
H = find_homography(matches, args.num_samples, args.max_iterations, args.inlier_threshold, args.max_inliers_ratio)
# 구한 homograpy를 warp_images에서 계산하기 편하도록 역행렬로 변환
H = np.linalg.inv(H)

# Warp images
image1, mask1, dt1, image2, mask2, dt2 = warp_images(image1, image2, H)

# Blend images
binary_blended_image = binary_blending(image1, mask1, dt1, image2, mask2, dt2)
cv.imwrite(args.binary_blending_path, binary_blended_image)

alpha_blended_image = alpha_blending(image1, mask1, dt1, image2, mask2, dt2)
cv.imwrite(args.alpha_blending_path, alpha_blended_image)

laplacian_pyramid_blended_image = laplacian_pyramid_blending(image1, mask1, dt1, image2, mask2, dt2, args.num_levels)
cv.imwrite(args.laplacian_pyramid_blending_path, laplacian_pyramid_blended_image)
