from .image_processing import ImagePreprocessing

# 导入 manga_ocr_master 中的 MangaOcr 类
from .manga_ocr_master.manga_ocr.ocr import MangaOcr

# 如果需要使用 Real-ESRGAN 作为一个模块进行处理，也可以导入它
# 例如 Real-ESRGAN 是某个类的名字或函数
# from .Real-ESRGAN.some_module import SomeClassOrFunction

# 定义哪些模块对外公开
__all__ = ["ImagePreprocessing", "MangaOcr"]