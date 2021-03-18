from PIL import Image
import numpy as np 

def taxi_img(img):
    gray = img.convert('L')
    bw = gray.point(lambda x: 0 if x<128 else 255, '1')
    img = bw.convert('RGBA')
    pixdata = img.load()
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pixdata[x, y] == (255, 255, 255, 255): # 흰색을
                pixdata[x, y] = (255,187,0,255)       # 노랑색으로
    return img

def old_img(img):
    gray = img.convert('L')
    bw = gray.point(lambda x: 0 if x<128 else 255, '1')
    img = bw.convert('RGBA')
    pixdata = img.load()
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pixdata[x, y] == (255, 255, 255, 255):  # 흰색을
                pixdata[x, y] = (0,105,0,255)          # 초록으로
                
            if pixdata[x, y] == (0, 0, 0, 255):       # 검정을
                pixdata[x, y] = (255, 255, 255, 255)  # 흰색으로
    return img
    
    
    
# 사용 방법  
im = Image.open('C09sa1223.jpg')   # 사진 이름

bg_fps = taxi_img(im)
bg_fps.show()
bg_fps.save("test1.png",'PNG')

bg_fps = old_img(im)
bg_fps.show()
bg_fps.save("test2.png",'PNG')