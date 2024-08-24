import json

import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

import requests

#-------文件读写--------

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
#-------文本检测与识别前的图片预处理-------

class PreprocessImage():
    def __init__(self, init_img_path):
        self.img_path = init_img_path
        self.img = cv2.imread(self.img_path)
        if self.img is None:
            raise FileNotFoundError(f"图像文件'{self.img_path}'不存在")
            
    #行分割
    def segment_lines(self):
        image = self.img
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 垂直投影
        vertical_projection = np.sum(binary, axis=1)

        # 找到行的上下边界
        rows = np.where(vertical_projection > 0)[0]
        line_segments = []
        start = rows[0]
        for end in np.hstack((np.array(np.where(np.diff(rows) > 1)).flatten(), rows[-1])):
            line_segments.append((start, end))
            start = end + 1

        # 分割出每行文本
        lines = []
        for i, (start, end) in enumerate(line_segments):
            line = image[start:end, :]
            lines.append(line)
            out_path = f'C:\\Users\\chait\\Desktop\\Anki-auto\\template_img\\line_{i}.jpg'
            cv2.imwrite(out_path, line)

        return lines

    #将原图转为只有绿色的图
    def get_green_img(self):
        #转换图片通道从BGR到HSV
        img = Image.open(self.img_path)
        img = np.array(img.convert('RGB'))
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #定义HSV中绿色的范围
        lower_green = np.array([40,50,40])
        upper_green = np.array([80, 255, 255])  
        #lower_green = np.array([60,255,255])
        #upper_green = np.array([60, 255, 255])  
        # 颜色过滤，得到只包含绿色的区域的掩模
        mask = cv2.inRange(hsv, lower_green, upper_green)


        # 通过掩模获取原图中的绿色区域
        green_img = cv2.bitwise_and(img, img, mask=mask)  #bitwise_and函数是对二进制数据进行“与”操作，如255 的二进制表示： 11111111 210 的二进制表示： 11010010 按位与操作结果：   11010010 (即十进制 210)

        cv2.imwrite('./template_img/word.jpg', green_img)

        return green_img, mask
    
    #把图片裁成多个只有绿色的部分的小图片
    def get_word_img(self):

        green_img, mask = self.get_green_img()

        low_green = np.array([60,255,255])
        high_green = np.array([60, 255, 255])  
        
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contours in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contours)

            crop_img = green_img[y:y+h, x:x+w]  

            crop_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

            mask_green = cv2.inRange(crop_hsv, low_green, high_green)
            mask_non_green = cv2.bitwise_not(mask_green)
            white_background = np.ones_like(crop_img, dtype=np.uint8) * 255
            result_img = cv2.bitwise_and(crop_img, crop_img, mask=mask_non_green)
            crop_img = cv2.add(result_img, cv2.bitwise_and(white_background, white_background, mask=mask_green))
                 
            gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            cv2.imwrite(f'./template_img/word_{i}_binary.jpg', binary_img)
            cv2.imwrite(f'./template_img/word_{i}_crop.jpg', crop_img)

