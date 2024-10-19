import argparse
import cv2 as cv


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

