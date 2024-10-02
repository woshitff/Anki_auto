from manga_ocr import MangaOcr
import time

#加入这个解决windows下SSL证书问题
import os
os.environ['CURL_CA_BUNDLE'] = ''

start_time = time.time()
mocr = MangaOcr(force_cpu=False)
text=[]
for i in range(0, 14):
#     text.append(mocr(f'C:\\Users\\chait\\Desktop\\Anki-auto\\Dataset\\template_img\\word_{i}_binary.jpg'))
        # text.append(mocr(fr'C:\Users\chait\Desktop\Anki-auto\Dataset\Image\template_img_sr\week_1\word_binary_sr_x8\word_{i}_binary_out.jpg'))
    text.append(mocr(fr'C:\Users\chait\Desktop\Anki-auto\Dataset\Image\template_img_sr\week_3\word_binary_sr_x8\word_{i}_binary_out.jpg'))

print(text)
print(str((time.time() - start_time)/60) + ' minutes')