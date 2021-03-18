#이는 욜로와 통합할때 사용하기 위한 유틸리티 파일입니다. 기본적으로 개인 유틸리티는 욜로의 기본루트에 있는 것으로 가정합니다.

import cv2
import os

def take_picture(x, img, save_path):
    x1, y1, x2, y2 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
    roi = img[x1:x2,y1:y2].copy()
    os.makedirs(save_path + '/img', exist_ok=True)
    cv2.imwrite(save_path + '/img', roi)
    return roi
