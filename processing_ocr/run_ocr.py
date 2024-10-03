import time
import glob
import re
import os
import warnings
import logging
import sys

os.environ['CURL_CA_BUNDLE'] = ''
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="VisionEncoderDecoderModel has generative capabilities")
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    handlers=[
        logging.FileHandler('./log/run_ocr.log', mode='w', encoding='utf-8'),  # 日志输出到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

import yaml

import numpy as np

from image_processing import ImagePreprocessing
from manga_ocr_master.manga_ocr.ocr import MangaOcr as MangaOcr

# ----------- tools for logging -----------
class StreamToLogger(object):
    def __init__(self, log_level):
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.log_level(line.rstrip())  # 输出到日志

    def flush(self):
        pass  # 由于 logging 自动处理缓冲，因此不需要显式 flush

# 将 stdout 和 stderr 都重定向到 logging
sys.stdout = StreamToLogger(logging.info)
sys.stderr = StreamToLogger(logging.error)

# ----------- tools for processing OCR -----------
def extract_number(file_path,sr_method):
    # 提取文件名，假设文件名格式为 word_数字_binary_out.jpg
    file_name = os.path.basename(file_path)
    if sr_method == 'None':
        number = re.search(r'word_(\d+)_binary', file_name)
    else:
        number = re.search(r'word_(\d+)_binary_out', file_name)
    return int(number.group(1)) if number else 0

# def get_binarization_params(config):
#     binarization_config = config['Binarization_Params']['Thresholding_Binarization']
    
#     if 'manual' in binarization_config:
#         binary_method = 'manual'
#         params = {'threshold': binarization_config['manual']['threshold']}
    
#     elif 'otsu' in binarization_config:
#         binary_method = 'otsu'
#         params = {}  
    
#     elif 'adaptive' in binarization_config:
#         binary_method = 'adaptive'
#         params = {
#             'block_size': binarization_config['adaptive']['block_size'],
#             'c': binarization_config['adaptive']['c']
#         }
    
#     else:
#         raise ValueError("No valid binary method found in the configuration")

#     return binary_method, params

# ----------- main function -----------
def main(config):
    logging.info("Starting image preprocessing")

    # # # start preprocessing Image
    start_time = time.time()
    preprocessor = ImagePreprocessing(config['Data_Params']['image_path'])
    
    # start crop image
    logging.info("Starting image cropping")
    crop_imgs = preprocessor.process_segmentation()

    # # # start augmentate image
    # logging.info("Starting image enhancement")
    # preprocessor.process_enhancement(
    #     crop_imgs=crop_imgs,
    #     low_green=np.array(config['ImageAugmentation_Params']['Make_white_bg']['low_green']),
    #     high_green=np.array(config['ImageAugmentation_Params']['Make_white_bg']['high_green']),  
    #     gamma=config['ImageAugmentation_Params']['Gamma_correction']['gamma'],
    #     hist_method=config['ImageAugmentation_Params']['Histogram_equalization']['method'],
    #     hist_params=config['ImageAugmentation_Params']['Histogram_equalization'].get(
    #         config['ImageAugmentation_Params']['Histogram_equalization']['method'], {}),
    #     binary_method=config['ImageAugmentation_Params']['Thresholding_Binarization']['method'],
    #     binary_params=config['ImageAugmentation_Params']['Thresholding_Binarization'].get(
    #         config['ImageAugmentation_Params']['Thresholding_Binarization']['method'], {}),
        # block_size=block_size if binary_method == 'adaptive' else None,  
        # c=c if binary_method == 'adaptive' else None  
    # )
    # _, sr_img_path = preprocessor.process_sr(modal=config['ImageUpscaling_Parms']['modal'],
    #                                             scale=config['ImageUpscaling_Parms']['scale'],
    #                                             sr_method=config['ImageUpscaling_Parms']['sr_method'])

    # # # start OCR
    # logging.info("Starting OCR processing")
    # start_time = time.time()
    # mocr = MangaOcr(force_cpu=False)
    # text=[]
    # img_paths = glob.glob(fr'{sr_img_path}\*.jpg')
    # # print(img_paths)
    # img_paths = sorted(img_paths, key=lambda file_path: extract_number(file_path, config['SR_Params']['sr_method']))
    # for img_path in img_paths:
    #     text.append(mocr(img_path))
    # print(text)
    # print(f'OCR 处理耗时: {(time.time() - start_time) / 60:.4f} minutes')
    # logging.info("Image preprocessing completed")


if __name__ == '__main__':
    with open('./configs/image_process.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)