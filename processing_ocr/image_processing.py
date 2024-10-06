import os 
import subprocess
from typing import Sequence
from contextlib import contextmanager
import functools   
import time
import logging

import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from itertools import pairwise
import cv2

# from realesrgan import RealESRGAN

#-------OCR前的图片预处理-------
def get_img_size(img_path):
    """
    获取图片的尺寸

    Args:
        img_path (str): 图片路径

    Returns:
        img_size (tuple): 图片的尺寸，格式为(width, height)
    """
    img = cv2.imread(img_path)
    print(img.shape)
    img_size = img.shape[:2]
    print('img_size:', img_size)
    return img_size

def run_RealESRGAN(input_dir, output_dir, scale=8):
    """
    调用Real-ESRGAN模型进行超分辨率
    
    Parameters:
      input_dir: 输入图片文件夹
      output_dir: 输出图片文件夹
    
    Returns:
      None
    """
    try:
        command = [
            'python', 'Real-ESRGAN/inference_realesrgan.py',
            '-i', input_dir,
            '-o', output_dir,
            '-n', 'RealESRGAN_x4plus',
            '-s', str(scale)
        ]
        subprocess.run(command, shell=True, check=True) 
        print('completed Real-ESRGAN processing')
    except subprocess.CalledProcessError as e:
        print(f'Error: {e.output}')

# --------basic tools for image processing--------
def log_function_call(func):
    """
    装饰器，用于打印方法执行的开始和结束标志
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n============ Start {func.__name__} ============")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"============ End {func.__name__} ============")
        print(f"Executed in: {end_time - start_time:.4f} seconds\n")
        return result
    return wrapper

@contextmanager
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    yield  

### for contour generation
class ShapeMatrixCreator: # todo
    """
    用于创建形状矩阵的类
    """
    def __init__(self):
        pass
    def optimal_k(self, points, method='elbow'):# 可以先写个按y坐标获得最佳聚类数量的方法 todo
        """
        使用肘部法或轮廓系数计算最佳聚类数量。

        :param points: 输入点的坐标，形状为(N, 2)的numpy数组。
        :param method: 选择的计算方法，'elbow'或'silhouette'。
        :return: 最佳聚类数量。
        """
        if method == 'elbow':
            inertia = []
            K = range(2, 10)  # 检查聚类数量从2到9
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=0)
                kmeans.fit(points)
                inertia.append(kmeans.inertia_)  # 收集每个k的惯性

            # 找到肘部位置
            best_k = K[np.diff(inertia, 2).argmin() + 2]  # 简单估算肘部位置
            return best_k

        elif method == 'silhouette':
            best_k = 2  
            best_score = -1  

            for k in range(2, 10):  
                kmeans = KMeans(n_clusters=k, random_state=0)
                labels = kmeans.fit_predict(points)
                score = silhouette_score(points, labels)  

                if score > best_score:
                    best_score = score
                    best_k = k  

            return best_k

        elif method == 'manual_group_by_height':
            pass
        else:
            raise ValueError("Method must be 'elbow' or 'silhouette'.")
    
    def initial_cluster(self, points, cluster_method='Hierarchical Clustering', cluster_params=None):# todo
        """
        对points进行初步聚类。主要目的是获得按y坐标分类的个数, 以及按照y坐标分类, 注意这里按y坐标分类个数不一定是通过聚类算法得到的, 可能是通过其他方法得到的

        parms:
            points: 输入点的坐标，形状为(N, 2)的numpy数组。
            cluster_method: 选择的聚类方法，'KMeans'或'Hierarchical Clustering'。
            cluster_params: 聚类方法的参数。
        return:
            labeled_points: 聚类后的点。(N, 3)的numpy数组。包含坐标及聚类标签。
        """
        if cluster_params is None:
            cluster_params = {}
        if cluster_method == 'Hierarchical Clustering':
            z = linkage(points, method='ward', metric='euclidean')
            clusters = fcluster(z, t=2, criterion='distance')
        elif cluster_method == 'KMeans':
            pass
        else:
            raise ValueError("Method is not supported.")

        labeled_points = np.hstack((points, clusters.reshape(-1, 1)))

        return labeled_points

    def refine_clusters(self, labeled_points):# todo
        """
        对初步聚类的结果进行细化, 对每个子类进行聚类
        parms:
            labeled_points: 聚类后的点。(N, 3)的numpy数组。包含坐标及聚类标签。
        return:
            refined_points: 细化后的点。(N, 3)的numpy数组。包含坐标及聚类标签。
        """
        pass
    
    def categorize_clusters(self, labeled_points):# todo
        """
        对前两步聚类的结果进行分类, 最终返回二维分类结果, 主要是为了统一不同纵坐标的横坐标聚类结果

        Parameters
        ----------
        labeled_points : numpy.ndarray
            聚类后的点。形状为 (N, 4) 的数组，包含坐标及横纵向聚类标签。

        Returns
        -------
        categories : numpy.ndarray
            分类结果。形状为 (N, 4) 的数组，包含坐标及二维分类标签。

        Examples
        --------
        >>> labeled_points = np.array([[1, 1, 0, 0], [2, 2, 0, 0], [3, 3, 0, 1], [4, 4, 1, 1]])
        >>> categorize_clusters(labeled_points)
        array([[1, 1, 0, 0],
               [2, 2, 0, 0],
               [3, 3, 1, 1],
               [4, 4, 2, 1]])
        """
        pass

    def create_shape_matrix(self, points):# todo
        """
        根据给定的点生成形状矩阵。
        
        :param points: 轮廓点的坐标，形状为(N, 2)的numpy数组。
        :return: 形状矩阵, 1表示分类的点, 0表示无分类的点。
        """
        n_classes_row = self.optimal_k(points[:, 1].reshape(-1, 1))
        n_classes_col = self.optimal_k(points[:, 0].reshape(-1, 1))
        # print(f"n_classes_row: {n_classes_row}, n_classes_col: {n_classes_col}")
    
        shape_matrix = np.zeros((n_classes_row, n_classes_col), dtype=int)

        row_labels = KMeans(n_clusters=n_classes_row).fit_predict(points[:, 1].reshape(-1, 1))
        col_labels = KMeans(n_clusters=n_classes_col).fit_predict(points[:, 0].reshape(-1, 1))

        labeled_points = np.hstack((points, row_labels.reshape(-1, 1), col_labels.reshape(-1, 1)))

        sorted_y_indices = np.argsort(-points[:, 1])  
        sorted_y_points = labeled_points[sorted_y_indices] 
        unique_row_labels = sorted(set(sorted_y_points[:, 2])) 
        row_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_row_labels)}
        sorted_y_points[:, 2] = np.array([row_mapping[i] for i in sorted_y_points[:, 2]])

        sorted_x_indices = np.argsort(-sorted_y_points[:, 0])  
        sorted_x_points = sorted_y_points[sorted_x_indices]
        unique_col_labels = sorted(set(sorted_x_points[:, 3])) 
        col_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_col_labels)}
        sorted_x_points[:, 3] = np.array([col_mapping[i] for i in sorted_x_points[:, 3]])

        new_labeled_points = sorted_x_points
        for i in range(len(new_labeled_points)):
            row = int(sorted_x_points[i][2])  
            col = int(sorted_x_points[i][3])  
            shape_matrix[row, col] = 1  

        return shape_matrix, n_classes_col, n_classes_row

# --------OCR前的图片预处理--------

class ImagePreprocessing:
    def __init__(self, origin_img_path):
        self.origin_img_path = origin_img_path
        self.img_id = os.path.basename(origin_img_path).split('.')[0]
        template_img_dir = 'C:\\Users\\chait\\Desktop\\Anki-auto\\Dataset\\Image\\template_img\\'
        template_img_dir = template_img_dir + self.img_id
        if not os.path.exists(template_img_dir):
            os.makedirs(template_img_dir)
        self.template_img_dir = template_img_dir
        self.img = cv2.imread(self.origin_img_path)
        if self.img is None:
            raise FileNotFoundError(f"图像文件'{self.origin_img_path}'不存在")
        
        self.shape_matrix_creator = ShapeMatrixCreator()
        self.segmentation = self.ImageSegmentation(self)
        self.augmentation = self.ImageAugmentation(self)
        self.upscaling = self.ImageUpscaling(self)
        self.visual = self.Visualization(self)
            
    @log_function_call
    def process_segmentation(self):
        cv2.imwrite(f'{self.template_img_dir}/img_clip/origin_img.jpg', self.img)

        logging.info("*********************************************")
        logging.info("* Step 1: extract text regions")
        logging.info("*********************************************")

        green_img, mask = self.segmentation.extract_text_regions(self.img)
        with ensure_directory_exists(f'{self.template_img_dir}/img_clip'):
            cv2.imwrite(f'{self.template_img_dir}/img_clip/word.jpg', green_img)

        contours = self.segmentation.get_contours(mask)
        with ensure_directory_exists(f'{self.template_img_dir}/img_clip/contour/origin_contour'):
            for i, contour in enumerate(contours):
                self.visual.visualize_and_save_contours(self.img, contour, f'{self.template_img_dir}/img_clip/contour/origin_contour/contour_{i}.jpg')
                
        logging.info("")
        logging.info("*********************************************")
        logging.info("* Step 2: ensure only one word in each image")
        logging.info("*********************************************")

        filtered_contours, unfiltered_contours, sorted_rows= self.segmentation.split_to_single_words(contours)
        with ensure_directory_exists(f'{self.template_img_dir}/img_clip/contour/initial_contour'):
            for i, contour in enumerate(filtered_contours):
                self.visual.visualize_and_save_contours(self.img, contour, f'{self.template_img_dir}/img_clip/contour/initial_contour/filtered_contour_{i}.jpg')
            for i, contour in enumerate(unfiltered_contours):
                self.visual.visualize_and_save_contours(self.img, contour, f'{self.template_img_dir}/img_clip/contour/initial_contour/unfiltered_contour_{i}.jpg')
            for i, row in enumerate(sorted_rows):
                row = row.reshape(-1, 1, 2)
                self.visual.visualize_and_save_contours(self.img, row, f'{self.template_img_dir}/img_clip/contour/initial_contour/sorted_row_{i}.jpg')
        crop_imgs = self.segmentation.initial_crop_img(filtered_contours, green_img)

        logging.info(f'Step 2 finished. Number of cropped images: {len(crop_imgs)}')
        logging.info("")

        logging.info("")
        logging.info("*********************************************")
        logging.info("* Step 3: refine cropped images")
        logging.info("*********************************************")

        refined_crop_imgs = self.segmentation.refine_crop_img(crop_imgs)
        # refined_crop_imgs = [self.segmentation._refine_crop_img(crop_imgs[5])]
        with ensure_directory_exists(f'{self.template_img_dir}/img_clip/cropped_img/refined_img'):
            for i, refined_crop_img in enumerate(refined_crop_imgs):
                cv2.imwrite(f'{self.template_img_dir}/img_clip/cropped_img/refined_img/word_{i}_crop_refined.jpg', refined_crop_img)

        print(f'Image segmentation finished. Total number of words: {len(refined_crop_imgs)}')
        logging.info("")
        
        return refined_crop_imgs

    @log_function_call
    def process_enhancement(self,
                            crop_imgs: list = [],
                            low_green: Sequence[int] = (60, 255, 255),
                            high_green: Sequence[int] = (60, 255, 255),
                            gamma: float = 1,
                            hist_method: str = "clahe",
                            hist_params: dict = {"clip_limit": 2.0, "tile_grid_size": (8, 8)},
                            binary_method: str = "manual",
                            binary_params: dict = {"thresh_val": 200, "max_val": 255}):
        """
        裁剪图片中包含的绿色区域并转换为白底、灰度化、直方图均衡化、二值化

        Args:
            None: method takes no arguments

        Returns:
            None: method returns nothing, but saves cropped images to the specified directory
        """
        save_index = 0
        for i, crop_img in enumerate(crop_imgs):
            # Convert to white background
            crop_img = self.augmentation._make_white_bg(crop_img, low_green, high_green)
            # Apply graylization to remove color
            gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            # Apply gamma correction to improve contrast
            gamma_corrected_img = self.augmentation._apply_adjust_gamma(gray_img, gamma=gamma)
            # Apply Histogram Equalization to improve contrast
            equalized_img = self.augmentation._apply_hist_equalization(gamma_corrected_img, method=hist_method, **hist_params) 
            # Apply thresholding to binarize the image
            binary_img = self.augmentation._apply_threshold_binary(equalized_img, binary_method=binary_method, **binary_params) 

            if not os.path.exists(f'{self.template_img_dir}/img_aurgmented/word_green'):
                os.makedirs(f'{self.template_img_dir}/img_aurgmented/word_green')
            if not os.path.exists(f'{self.template_img_dir}/img_aurgmented/word_gray'):
                os.makedirs(f'{self.template_img_dir}/img_aurgmented/word_gray')
            if not os.path.exists(f'{self.template_img_dir}/img_aurgmented/word_gray_gamma'):
                os.makedirs(f'{self.template_img_dir}/img_aurgmented/word_gray_gamma')
            if not os.path.exists(f'{self.template_img_dir}/img_aurgmented/word_gray_equalized'):
                os.makedirs(f'{self.template_img_dir}/img_aurgmented/word_gray_equalized')
            if not os.path.exists(f'{self.template_img_dir}/img_aurgmented/word_binary'):
                os.makedirs(f'{self.template_img_dir}/img_aurgmented/word_binary')

            cv2.imwrite(f'{self.template_img_dir}/img_aurgmented/word_green/word_{save_index}_green.jpg', crop_img)
            cv2.imwrite(f'{self.template_img_dir}/img_aurgmented/word_gray/word_{save_index}_gray.jpg', gray_img)
            cv2.imwrite(f'{self.template_img_dir}/img_aurgmented/word_gray_gamma/word_{save_index}_gray_gamma.jpg', gamma_corrected_img)
            cv2.imwrite(f'{self.template_img_dir}/img_aurgmented/word_gray_equalized/word_{save_index}_gray_equalized.jpg', equalized_img)
            cv2.imwrite(f'{self.template_img_dir}/img_aurgmented/word_binary/word_{save_index}_binary.jpg', binary_img)

            save_index += 1

        print(f'Image enhancement finished. Total number of words: {len(crop_imgs)}')

    @log_function_call
    def process_sr(self,
                    modal: str = "binary",
                    scale: int = 8, 
                    sr_method: str = "CUBIC"):
        """
        对图像进行超分辨率放大。
        
        Args:
            modal (str): 图像来源，可选值有'origin'和'binary', 'gray'和'gray_gamma'等
            scale (int): 放大倍数
            method (str): 超分辨率方法，可选值有'CUBIC', 'RealESRGAN'和'None'
        Returns:
            None: method returns nothing, but saves upscaled images to the specified directory
        """
        
        father_dir = self.template_img_dir
        modal_dir = f"word_{modal}"
        input_dir = os.path.join(father_dir, 'img_aurgmented', modal_dir)
        output_dir = os.path.join(father_dir, 'img_aurgmented', modal_dir + f"_{sr_method}" + f"_x{scale}")
    
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"无法读取图像: {img_name}")
                continue

            if sr_method == "CUBIC":
                height, width = img.shape[:2]
                new_size = (int(width * scale), int(height * scale))
                upscaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
                output_img_path = os.path.join(output_dir, os.path.splitext(img_name)[0]+"_out.jpg")
                cv2.imwrite(output_img_path, upscaled_img)
            elif sr_method == "RealESRGAN":
                run_RealESRGAN(input_dir, output_dir, scale)
            elif sr_method == "None":
                output_img_path = os.path.join(output_dir, os.path.splitext(img_name)[0]+"_out.jpg")
                cv2.imwrite(output_img_path, img)
            else:
                raise ValueError(f"Invalid method: {sr_method}")
        print("Image SR finished.")
        return input_dir, output_dir

    class ImageSegmentation:
        def __init__(self, parent_distance):
            self.parent = parent_distance
        # *********************************************
        # * Step 0: 辅助函数
        # *********************************************
        def _contour_to_mask(self, contour, img_shape):
            """
            将轮廓转换为掩模

            Args:
                contour (list): 轮廓
                img_shape (tuple): 图像的形状

            Returns:
                mask (numpy.ndarray): 掩模
            """
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            return mask
        
        def _mask_to_contour(self, mask):
            """
            将掩模转换为轮廓

            Args:
                mask (numpy.ndarray): 掩模

            Returns:
                contour (list): 轮廓
            """
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours[0]

        def _calculate_mask_intersection_and_ratio(self, mask1, mask2):
            """
            计算两个mask的交集, 并返回交集区域在mask1中的比例
            Args:
                mask1 (numpy array): 第一个mask (例如 contour_mask)
                mask2 (numpy array): 第二个mask (例如 _get_green_regions 返回的 mask)
            Returns:
                intersection_mask (numpy array): 两个mask的交集
                ratio (float): 交集区域在mask1中所占的比例
            """
            intersection_mask = cv2.bitwise_and(mask1, mask2)
            intersection_pixels = np.count_nonzero(intersection_mask)

            mask1_pixels = np.count_nonzero(mask1)
            ratio = intersection_pixels / mask1_pixels if mask1_pixels > 0 else 0

            return intersection_mask, ratio
        
        def _segment_image_to_lines(self):
            """
            将图片分割成行，并保存每行图片

            Args:
                None:
            Returns:
                lines_img: list of image, each image is a line of the original image
            """
            image = self.parent.img
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 计算每行的投影
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
                out_path = f'{self.parent.template_img_dir}\\line_{i}.jpg'
                cv2.imwrite(out_path, line)

            return lines
        # *********************************************
        # * Step 1: 提取文本区域
        # *********************************************
        def extract_text_regions(self, img):
            """
            获取图片中包含的绿色区域

            Args:
                None: method takes no arguments

            Returns:
                green_img (numpy.ndarray): 包含绿色区域的图片。
                mask (numpy.ndarray): 掩模，用于提取绿色区域。
            """
            #转换图片通道从BGR到HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            #定义HSV中绿色的范围
            lower_green = np.array([40,50,40])
            upper_green = np.array([80, 255, 255])    
            # 颜色过滤，得到只包含绿色的区域的掩模
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # 通过掩模获取原图中的绿色区域
            green_img = cv2.bitwise_and(img, img, mask=mask)  
            # green_img[cv2.bitwise_not(mask)==255] = [255,255,255]
            # if img is self.parent.img:
            #     if not os.path.exists(f'{self.parent.template_img_dir}/img_clip'):
            #         os.makedirs(f'{self.parent.template_img_dir}/img_clip')
            #     cv2.imwrite(f'{self.parent.template_img_dir}/img_clip/word.jpg', green_img)

            return green_img, mask
       
        def get_contours(self, 
                      mask: np.ndarray, 
                      ):
            """
            获取图片中轮廓

            Args:
                mask (numpy.ndarray): 掩模，用于提取轮廓
            Returns:
                contours (list): 轮廓列表
            """
            H, W = mask.shape[:2]
            kernel_size = max(3, 2 * ((min(H, W) // 50) // 2) + 1)
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((kernel_size, kernel_size), np.uint8),iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        # *********************************************
        # * Step 2: 确保每个图像中只有一个词语，非语义手段判断
        # *********************************************
        def _sort_contours(self, contours, line_threshold):
            """
            对轮廓按照从上到下、从左到右的顺序进行排序

            Args:
                contours (list): 轮廓列表 coutours.shape = (N, 1, 2)
                line_threshold (int): 行的高度阈值
            """
            contours.sort(key=lambda c: cv2.boundingRect(c)[1])

            grouped_contours = []
            current_group = [contours[0]]  # 初始化第一个分组
            last_y = cv2.boundingRect(contours[0])[1]

            for contour in contours[1:]:
                x, y, w, h = cv2.boundingRect(contour)
                if abs(y - last_y) < line_threshold:
                    current_group.append(contour)
                else:
                    grouped_contours.append(current_group)  
                    current_group = [contour] 
                last_y = y

            if current_group:
                grouped_contours.append(current_group)

            for group in grouped_contours:
                group.sort(key=lambda c: cv2.boundingRect(c)[0])

            sorted_contours = [contour for group in grouped_contours for contour in group]
            return sorted_contours
        
        def _points_enhancer(self, points): 
            """
            增强点的数量，使得每行的点数量均衡。
            Args:
                points (list): 点列表
            Returns:
                new_points (list): 增强后的点列表
            """
            # 先考虑对一行文字的点进行增强
            points = np.array(points) if isinstance(points, list) else points
            
            x, y, w, h = cv2.boundingRect(points)
            rect_points = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            
            points = np.vstack([points, rect_points])
            
            return np.array(points)

        def _split_and_merge_rows(self, points):
            """
            将一个轮廓的点按行分组并邻行合并

            Parameters
            ----------
            points : numpy.ndarray
                轮廓的点。形状为 (N, 2) 的数组。
            
            Returns
            -------
            line_segments : list[(np.ndarray, np.ndarray)]
                每行的点组成的list。每个元素是一个元组 (points, label)，其中 points 形状为 (M, 2) 的数组, label 是点的行标签(M,1)
            """
            boundary_rows = []
            current_row = [points[0]]
            
            for i in range(1, len(points)):
                if abs(points[i][1] - points[i-1][1]) < 15:
                    current_row.append(points[i])
                else:
                    boundary_rows.append(current_row)
                    current_row = [points[i]]
            boundary_rows.append(current_row)

            line_segments = []
            for idx, (row1, row2) in enumerate(zip(boundary_rows[:-1], boundary_rows[1:])):
                merged_row = row1 + row2  
                labels_merged = [idx] * len(row1) + [idx + 1] * len(row2)

                merged_row = self._points_enhancer(merged_row)

                addition_points_count = len(merged_row) - len(labels_merged)
                if addition_points_count > 0:
                    row1_mean_y = np.mean([p[1] for p in row1])
                    row2_mean_y = np.mean([p[1] for p in row2])
                    for point in merged_row[-addition_points_count:]:
                        closet_label = idx if abs(point[1] - row1_mean_y) < abs(point[1] - row2_mean_y) else (idx + 1)
                        labels_merged.append(closet_label)

                line_segments.append((merged_row, np.array(labels_merged)))

            return line_segments

        def _group_points_by_col(self, points, threshold=5):
            """
            将点按列分组

            Parameters
            ----------
            points : numpy.ndarray
                点的坐标。形状为 (N, 2) 的数组。
            threshold : int
                分组的阈值

            Returns
            -------
            grouped_points : list
                按列分组的点。每个元素是一列的点。形状为 (M, 2) 的数组。
            """
            grouped_points = []
            current_group = [points[0]]

            for point in points[1:]:
                if abs(point[0] - current_group[-1][0]) < threshold:  
                    current_group.append(point)
                else:
                    grouped_points.append(current_group)
                    current_group = [point]
            if current_group:
                grouped_points.append(current_group)
            
            return grouped_points

        def _generate_line_contours(self, points, line_labels): 
            """
            根据给定的点生成文本行的矩形轮廓，进行 x 方向上的分组并根据 y 坐标差异进行过滤。
            
            Parameters
            ------------
            points : np.ndarray
                点的坐标，形状为 (N, 2) 的数组。
            line_labels : np.ndarray
                点的行标签，形状为 (N, 1) 的数组
                
            Returns:
            ------------
            filtered_contours_list (list[np.ndarray]):
                过滤后的轮廓列表，其中每个轮廓的形状为 (N, 1, 2) 的数组。
            """
            points = np.array(points) if isinstance(points, list) else points
            line_labels = np.array(line_labels) if isinstance(line_labels, list) else line_labels

            sorted_indices = np.argsort(points[:, 0])
            points = points[sorted_indices]
            line_labels = line_labels[sorted_indices]

            # 按 x 坐标分组
            grouped_points = self._group_points_by_col(points, threshold=5)
            points_to_labels = {tuple(p): label for p, label in zip(points, line_labels)}
            grouped_labels = [[points_to_labels.get(tuple(g)) for g in group] for group in grouped_points]
        
            # 按 y 坐标差异进行过滤, 这里过滤是为了去除每一行非边界的点
            filtered_points = [
                group for group, labels in zip(grouped_points, grouped_labels) if len(np.unique(labels)) == 2
            ]

            filtered_points_list = [
                np.array(group).reshape(-1, 1, 2).astype(np.int32) for group in filtered_points
            ]
            filtered_contours_list = [
                np.concatenate((points1, points2), axis=0) for points1, points2 in pairwise(filtered_points_list)
            ]
            
            return filtered_contours_list

        def _filter_line_contour(self, contour): 
            """
            """
            _, mask = self.extract_text_regions(self.parent.img)
            contour_mask = self._contour_to_mask(contour, mask.shape)

            if self._calculate_mask_intersection_and_ratio(contour_mask, mask)[1] < 0.1:
                return None
            else:
                return contour

        def _split_per_contour(self, contour): 
            """
            将轮廓分割成多个点并分组,组合成新的轮廓

            Parameters
            ------------
            contour : np.ndarray 
                轮廓, 形状为 (N, 1, 2)

            Returns
            ------------
            new_contour_list (list[np.ndarray]): 
                新的轮廓列表, 其中每个轮廓形状为 (N, 1, 2)
            """
            # print('contour', contour)
            points = contour[:, 0, :] # points.type = np.ndarray
            points = sorted(points, key=lambda p: p[1])

            line_segments = self._split_and_merge_rows(points)

            unfiltered_contours = [] 
            filtered_contours = []
            for line_points, line_labels in line_segments:
                contours = self._generate_line_contours(line_points, line_labels) 
                unfiltered_contours.extend(contours)
                filtered_contours.extend([c for c in contours if self._filter_line_contour(c) is not None])

            return filtered_contours, unfiltered_contours, line_segments
        
        def split_to_single_words(self, \
                         contours: list \
                        ):
            """
            将多边形轮廓按行进行分割，并根据行排序（从上到下、从左到右), 最终获得一个个单词, 或者是单词里的字的轮廓
            Args:
                contours (list): 轮廓列表
                green_img (numpy.ndarray): 包含绿色区域的图片 
                left_margin_threshold (int): 左边距阈值
                right_margin_threshold (int): 右边距阈值
            Returns:
                new_contours (list): 过滤后的轮廓列表
            Example:
                >>> contours = [np.array([[[10, 10]], [[100, 10]], [[100, 30]], [[10, 30]]]),
                            np.array([[[20, 20], [100, 20], [100, 100], [20, 100]]])]
                >>> new_contours = filter_contours_by_height(contours)
                >>> print(new_contours)
                output: [np.array([[[10, 10], [100, 10], [100,20], [10,20]]]),
                        np.array([[[10, 20], [100, 20], [100, 30], [20, 30]]]),
                        np.array([[[20, 20], [100, 20], [100, 100], [20, 100]]])]
            """
            filtered_contours = []
            unfiltered_contours = []
            merged_rows_points = []
            
            for i, contour in enumerate(contours):
                new_contour, old_contour, line_segments = self._split_per_contour(contour)
                merged_rows = [line for line, _ in line_segments]
                filtered_contours.append(new_contour)
                unfiltered_contours.append(old_contour)
                merged_rows_points.append(merged_rows)
            # eg: merged_rows_points = [[np.ndarray([[0, 1], [0, 2], [1, 1]]), np.ndarray([[1, 1], [2, 1], [2, 2]])], [np.ndarray([[0, 1], [0, 2], [1, 1]]), np.ndarray([[1, 1], [2, 1], [2, 2]])]]

            heights = [cv2.boundingRect(c)[3] for c in contours]
            median_height = np.median(heights)
            filtered_contours = [c for sublist in filtered_contours for c in sublist]
            sorted_filtered_contours = self._sort_contours(filtered_contours, median_height*0.5)

            unfiltered_contours = [c for sublist in unfiltered_contours for c in sublist]
            sorted_unfiltered_contours = self._sort_contours(unfiltered_contours, median_height*0.5)

            filtered_rows_points = [c for sublist in merged_rows_points for c in sublist]
            sorted_rows_points = self._sort_contours(filtered_rows_points, median_height*0.5)
            # eg: filtered_rows_points = [np.ndarray([[0, 1], [0, 2], [1, 1]]), np.ndarray([[1, 1], [2, 1], [2, 2]]), np.ndarray([[0, 1], [0, 2], [1, 1]]), np.ndarray([[1, 1], [2, 1], [2, 2]])]
            
            return sorted_filtered_contours, sorted_unfiltered_contours, sorted_rows_points
                
        def initial_crop_img(self, 
                               contours: list,
                               green_img: np.ndarray):
            """
            根据轮廓获取图片中的文字区域,需要在这里处理换行问题

            Args:
                contours (list[np.ndarray, (N, 1, 2)]): 轮廓列表
                green_img (numpy.ndarray): 包含绿色区域的图片
            Returns:
                crop_imgs (list): 文字区域列表
            """
            # 裁剪并合并换行的文字区域        
            crop_imgs = []        
            if not os.path.exists(f'{self.parent.template_img_dir}/img_clip/cropped_img/initial_img'):
                os.makedirs(f'{self.parent.template_img_dir}/img_clip/cropped_img/initial_img')
            # print('contour', contours)

            img_width = self.parent.img.shape[1]
            save_index = 0
            skip = False
            # black_threshold = 0.2

            for i, contour in enumerate(contours): 
                if skip:
                    skip = False
                    print(f'Skipping contour {i}')
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                crop_img = green_img[y:y+h, x:x+w]  
                # # 过滤掉黑色区域
                # non_zero_pixels = np.count_nonzero(crop_img)
                # total_pixels = crop_img.size
                # non_zero_ratio = non_zero_pixels / total_pixels
                # if non_zero_ratio < black_threshold:
                #     print(f'Skipping contour {i}, non_zero_ratio: {non_zero_ratio}')
                #     continue

                right_margin_threshold =  img_width * 0.04  # 使用图像宽度的 4% 作为右边距阈值
                left_margin_threshold =  img_width * 0.025
                if i < len(contours) - 1:
                    next_contour = contours[i+1]
                    next_x, next_y, next_w, next_h = cv2.boundingRect(next_contour)
                    # print(f'crop_img: {save_index}, {x}, {y}, {abs(x + w - self.img.shape[1])}, {abs(next_x)}, {right_margin_threshold}')
                    if abs(x + w - self.parent.img.shape[1]) < right_margin_threshold and abs(next_x) < left_margin_threshold:
                        logging.info(f'Merging: word_{i}_crop.jpg and word_{i+1}_crop.jpg need to merge')

                        next_y_max = min(next_y + h, green_img.shape[0])
                        crop_img_next = green_img[next_y:next_y_max, next_x:next_x + next_w]

                        height_diff = crop_img.shape[0] - crop_img_next.shape[0]
                        if height_diff > 0:
                            padding = ((0, height_diff), (0, 0), (0, 0))
                            crop_img_next = np.pad(crop_img_next, padding, mode='constant', constant_values=0)
                        elif height_diff < 0:
                            padding = ((abs(height_diff), 0), (0, 0), (0, 0))
                            crop_img = np.pad(crop_img, padding, mode='constant', constant_values=0)

                        crop_img = np.concatenate((crop_img, crop_img_next), axis=1)
                        crop_imgs.append(crop_img)
                        cv2.imwrite(f'{self.parent.template_img_dir}/img_clip/cropped_img/initial_img/word_{save_index}_crop.jpg', crop_img)
                        save_index += 1
                        skip = True
                    else:
                        crop_imgs.append(crop_img)
                        cv2.imwrite(f'{self.parent.template_img_dir}/img_clip/cropped_img/initial_img/word_{save_index}_crop.jpg', crop_img)
                        save_index += 1
                else:
                    crop_imgs.append(crop_img)
                    cv2.imwrite(f'{self.parent.template_img_dir}/img_clip/cropped_img/initial_img/word_{save_index}_crop.jpg', crop_img)
                    save_index += 1
            return crop_imgs
        # *********************************************
        # * Step 3: 对粗处理后的图片进行细化处理
        # *********************************************
        def _refine_big_contour(self, contour):
            """
            将大轮廓分割, 保留主要文字的轮廓, 去掉attach的噪声
            Args:
                contour (numpy.ndarray): 轮廓
                img (numpy.ndarray): 图片
            Returns:
                contours (list of np.ndarray): 矩形轮廓列表
            """        
            points = contour[:, 0, :]
            points = points[np.argsort(points[:, 0])]

            grouped_points = []
            current_group = [points[0]]

            x_threshold = 3 
            for point in points[1:]:
                if abs(point[0] - current_group[-1][0]) < x_threshold:
                    current_group.append(point)
                else:
                    grouped_points.append(current_group)
                    current_group = [point]
            if current_group:
                grouped_points.append(current_group)
            
            x, y, w, h = cv2.boundingRect(contour)
            y_threshold = h*0.2
            filtered_points = []
            for group in grouped_points:
                y_coords = [p[1] for p in group]
                y_diff = max(y_coords) - min(y_coords)
                if (h-y_diff) < y_threshold:
                    filtered_points.append(group)
            if len(grouped_points) > 1:
                filtered_points = [filtered_points[0], filtered_points[-1]]
            assert len(filtered_points) == 2, print(f'Contour has more than one rectangle')

            filtered_points = np.vstack(filtered_points)
            filtered_contours = np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))
            return filtered_contours

        def refine_crop_img(self, crop_imgs):
            """
            更精细化的裁剪图片,同时可以去掉下划线等噪声, 避免使用形态学方法导致文字受到干扰
            主要有两种问题：
                1. 噪声在主轮廓内
                2. 噪声在主轮廓外
            Args:
                crop_img (numpy.ndarray): 待裁剪图片
            Returns:
                refined_crop_img (numpy.ndarray): 裁剪后的图片
            """
            refined_crop_imgs = []
            for i, crop_img in enumerate(crop_imgs):
                img_area = float(crop_img.shape[0] * crop_img.shape[1])
                green_img, mask = self.extract_text_regions(crop_img)

                with ensure_directory_exists(f'{self.parent.template_img_dir}/img_clip/mask/refined_mask'):
                    self.parent.visual.visualize_and_save_mask(crop_img, mask, f'{self.parent.template_img_dir}/img_clip/mask/refined_mask/mask_{i}.jpg')
                    self.parent.visual.visualize_and_save_mask(np.ones_like(crop_img, dtype=np.uint8)*255, mask, f'{self.parent.template_img_dir}/img_clip/mask/refined_mask/mask_white_{i}.jpg')
        
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
                # print(contours)
                with ensure_directory_exists(f'{self.parent.template_img_dir}/img_clip/contour/refined_contour'):
                    self.parent.visual.visualize_and_save_contours(crop_img, contours, f'{self.parent.template_img_dir}/img_clip/contour/refined_contour/contours_{i}.jpg')
                    self.parent.visual.visualize_and_save_contours(np.ones_like(crop_img, dtype=np.uint8)*255, contours, f'{self.parent.template_img_dir}/img_clip/contour/refined_contour/contours_white_{i}.jpg')
                
                cropped_img = []
                for i, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    contour_wh_ratio = float(w) / float(h)
                    contour_area = cv2.contourArea(contour)
                    # condition to filter out detached noises
                    if h > 0 and contour_wh_ratio > 10 or contour_area/img_area < 0.07:
                        continue
                    elif h == 0:
                        print(f'Contour {i} has zero height, skipping')
                    else:
                        pass
                    # condition to filter out attached noises
                    refined_contour = self._refine_big_contour(contour)
                    x, y, w, h = cv2.boundingRect(refined_contour)
                    cropped_img.append(green_img[y:y+h, x:x+w])
        
                max_height = max(img.shape[0] for img in cropped_img)
                resized_images = []
                for img in cropped_img:
                    height, width = img.shape[:2]
                    resized_img = cv2.resize(img, (width, max_height))
                    resized_images.append(resized_img)
                # print(f'Number of resized images: {len(resized_images)}')
                # print(f'Shape of resized_images: {resized_images[0].shape}, {resized_images[-1].shape}')
                refined_crop_img = cv2.hconcat(resized_images)
                # print(f'Shape of refined_crop_img: {refined_crop_img.shape}')
                refined_crop_imgs.append(refined_crop_img)
            return refined_crop_imgs
                
    class ImageAugmentation:
        def __init__(self, parent_distance):
            self.parent = parent_distance
                    
        def _make_white_bg(self, img, low_val, high_val):
            """
            将图片转换为白底

            Args:
                img (numpy.ndarray): 输入图片
                low_val (np.ndarray): 低阈值
                high_val (np.ndarray): 高阈值
            Returns:
                img (numpy.ndarray): 输出图片
            """
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, low_val, high_val)
            white_background = np.ones_like(img, dtype=np.uint8) * 255
            mask_inv = cv2.bitwise_not(mask)
            result_img = cv2.bitwise_and(img, img, mask=mask_inv)
            img = cv2.add(result_img, cv2.bitwise_and(white_background, white_background, mask=mask))

            return img
        
        def _apply_adjust_gamma(self, img, gamma=1.5):
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(img, table)
        
        def _apply_hist_equalization(self, img, method="clahe", **hist_params):
            """
            根据参数method对图片进行直方图均衡化

            Args:
                img_gray (numpy.ndarray): 输入灰度图像
                method (str): 直方图均衡化方法，可选值有'clahe'和'hist'
                clip_limit (float): 裁剪限值,仅在method为'clahe'时有效
                tile_grid_size (tuple): 块大小,仅在method为'clahe'时有效
            Returns:
                img_equalized (numpy.ndarray): 输出均衡化后的图像
            """
            if method == "clahe":
                clip_limit = hist_params.get("clip_limit", 2.0)
                tile_grid_size = hist_params.get("tile_grid_size", (8, 8))
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
                equalized_img = clahe.apply(img)
            elif method == "hist":
                equalized_img = cv2.equalizeHist(img)
            else:
                raise ValueError(f"Invalid method: {method}")
            return equalized_img
        
        def _apply_threshold_binary(self, img, binary_method="otsu", **binary_params):
            """
            根据参数method对图片进行二值化

            Args:
                img (numpy.ndarray): 输入图像
                binary_method (str): 二值化方法，可选值有'otsu','manual'和'adaptive'
                binary_params (dict): 二值化参数
            Returns:
                binary_img (numpy.ndarray): 输出二值化后的图像
            """
            if binary_method == "otsu":
                _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif binary_method == "manual":
                thresh_val = binary_params.get("thresh_val", 200)
                max_val = binary_params.get("max_val", 255)
                _, binary_img = cv2.threshold(img, thresh_val, max_val, cv2.THRESH_BINARY)
            elif binary_method == "adaptive":
                max_val = binary_params.get("max_val", 255)
                adaptive_method = binary_params.get("adaptive_method", cv2.ADAPTIVE_THRESH_MEAN_C)
                block_size = binary_params.get("block_size", 11)
                C = binary_params.get("C", 2)
                binary_img = cv2.adaptiveThreshold(img, max_val, adaptive_method, cv2.THRESH_BINARY, block_size, C)
            else:
                raise ValueError(f"Invalid method: {binary_method}")
            return binary_img
        
    class ImageUpscaling:
        def __init__(self, parent_distance):
            self.parent = parent_distance
        
    class Visualization:
        def __init__(self, parent_distance):
            self.parent = parent_distance

        def visualize_and_save_mask(self, img, mask, output_path):        
            visualization = img.copy()
            visualization[mask == 255] = [0, 0, 255]  
            cv2.imwrite(output_path, visualization)
        
        def visualize_and_save_contours(self, img, contours, output_path):
            visualization = img.copy()
            cv2.drawContours(visualization, contours, -1, (0, 0, 255), 2)
            for i, contour in enumerate(contours):
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                cv2.putText(visualization, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 
                            img.shape[0]/1000, (255, 0, 0), 2)  
            cv2.imwrite(output_path, visualization)
        
        def visualize_and_save_shapematrix(self, img, contours, output_path):
            """
            将轮廓绘制在图片上,并绘制其形状矩阵以及行列类别数量
            """
            visualization = img.copy()
            cv2.drawContours(visualization, contours, -1, (0, 0, 255), 2)
            for i, contour in enumerate(contours):
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                cv2.putText(visualization, str(i+1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 
                            img.shape[0]/1000, (255, 0, 0), 2)  
                
            points = contours[:, 0, :]
            shape_matrix, n_class_x, n_class_y = self.parent.shape_matrix_creator.create_shape_matrix(points)
            scale_factor = 0.2  
            
            new_height = int(img.shape[0] * scale_factor)
            new_width = int(img.shape[1] * scale_factor)

            shape_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            for i in range(shape_matrix.shape[0]):
                for j in range(shape_matrix.shape[1]):
                    y_start = int((shape_matrix.shape[0] - 1 - i) * (new_height / shape_matrix.shape[0]))
                    y_end = int((shape_matrix.shape[0] - i) * (new_height / shape_matrix.shape[0]))
                    x_start = int(j * (new_width / shape_matrix.shape[1]))
                    x_end = int((j + 1) * (new_width / shape_matrix.shape[1]))
                    if shape_matrix[i, j] == 1:  
                        shape_image[y_start:y_end, x_start:x_end] = [0, 255, 255] 
                    else:
                        shape_image[y_start:y_end, x_start:x_end] = [255, 255, 255] 
            grid_color = (0, 0, 255)  
            for i in range(1, shape_matrix.shape[0]):
                y_pos = int(i * (new_height / shape_matrix.shape[0]))
                cv2.line(shape_image, (0, y_pos), (new_width, y_pos), grid_color, 1)  

            for j in range(1, shape_matrix.shape[1]):
                x_pos = int(j * (new_width / shape_matrix.shape[1]))
                cv2.line(shape_image, (x_pos, 0), (x_pos, new_height), grid_color, 1)   
            
            text_image_height = int(scale_factor*img.shape[0])
            text_image = np.ones((text_image_height, new_width, 3), dtype=np.uint8) * 200 
            text = f"({n_class_x}, {n_class_y})"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_color = (0, 0, 0)  
            font_thickness = 2

            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            text_x = (new_width - text_width) // 2
            text_y = (text_image_height + text_height) // 2  
            cv2.putText(text_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

            combined_image = np.vstack((shape_image, text_image))
            empty_column = np.zeros((img.shape[0], new_width, 3), dtype=np.uint8) 
            empty_column[:new_height+text_image_height, :new_width] = combined_image  
            black_line = np.zeros((empty_column.shape[0], 5, 3), dtype=np.uint8) 

            visualization = np.hstack((visualization, black_line, empty_column))  
            cv2.imwrite(output_path, visualization)

if __name__ == '__main__':
    origin_img_path = 'C:\\Users\\chait\\Desktop\\Anki-auto\\Dataset\\Image\\origin_img\\1.png'
    preprocessor = ImagePreprocessing(origin_img_path, week_num=3)
    
