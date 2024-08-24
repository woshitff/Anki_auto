from Data.utils import PreprocessImage
import os

from ocr.ocr import OCR



img_prepare = PreprocessImage('C:\\Users\\chait\\Desktop\\Anki-auto\\origin_img\\1.png')
###这是为什么呢，只有写绝对路径才能读取图片吗？

img_prepare.get_word_img()

#print(OCR().txt(img_path='C:\\Users\\chait\\Desktop\\Anki-auto\\template_img\\word_1_binary.jpg'))
#print(OCR().txt(img_path='C:\\Users\\chait\\Desktop\\Anki-auto\\image\\3.png'))