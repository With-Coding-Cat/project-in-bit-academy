#이는 욜로와 통합할때 사용하기 위한 유틸리티 파일입니다. 기본적으로 개인 유틸리티는 욜로의 기본루트에 있는 것으로 가정합니다.

import cv2
import os

img_name_numbering = 0
def take_picture(x, img, save_path):
    global img_name_numbering
    img_name_numbering += 1
    x1, y1, x2, y2 = int(x[0]), int(x[1]), int(x[2]), int(x[3])
    roi = img[y1:y2,x1:x2].copy()
    os.makedirs(str(save_path / 'img'), exist_ok=True)
    #cv2.imwrite(str(save_path / 'img/{}.jpg'.format(time.localtime(time.time()))), roi)
    cv2.imwrite(str(save_path / 'img/{}.jpg'.format(img_name_numbering)), roi)
    return roi
