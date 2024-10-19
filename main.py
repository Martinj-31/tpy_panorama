import argparse
import cv2 as cv

from utils.feature_matching import extract_sift_feature, visualize_sift_feature


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