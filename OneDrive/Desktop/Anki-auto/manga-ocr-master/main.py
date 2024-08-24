from manga_ocr import MangaOcr

#加入这个解决windows下SSL证书问题
import os
os.environ['CURL_CA_BUNDLE'] = ''

mocr = MangaOcr()
text=[]
for i in range(0, 15):
    text.append(mocr(f'C:\\Users\\chait\\Desktop\\Anki-auto\\template_img\\word_{i}_crop.jpg'))
print(text)