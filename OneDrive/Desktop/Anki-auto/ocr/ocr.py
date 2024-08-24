import cv2
from PIL import Image, ImageEnhance
from paddleocr import PaddleOCR, draw_ocr

class OCR(PaddleOCR):
    def __init__(self, lang='japan'):

        super().__init__(lang=lang)

    def txt(self, img_path):
        
        result = self.ocr(img_path, cls=True)
        result = result[0]
        txts = [line[1][0] for line in result]
        return txts