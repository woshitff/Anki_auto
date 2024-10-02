import os
# print(os.getcwd())
from datetime import datetime

from .fields import Core_2000_Fields
from .anki_card import add_note

class IndexGenerator:
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y%m%d%H%M')
        self.counter = 0

    def generate_index(self):
        # 增加计数器并格式化为三位数
        self.counter += 1
        counter_str = f"{self.counter:03d}"
        # 生成唯一索引
        index = f"{self.timestamp}-{counter_str}"
        return index

def main():
    jap_text = "日本語"
    timestamp_generator = IndexGenerator()
    jap_fields = Core_2000_Fields(jap_text, timestamp_generator)
    add_note(
        deck_name="アカネ",
        model_name="Core 2000",
        fields_data=jap_fields()
    )

if __name__ == "__main__":
    main()
    # done
    # 1. 完成了基本的anki卡片生成功能，输入：日语文本，输出：anki卡片
    # 2. 完成了索引生成器
    # todo 
    # 1. 音频效果不好，更换模型  # done 已更换模型 tts -> gtts 
    # 2. 查询单词含义会出现问题  # done 已解决，把单词从self.word 改为 self.text
    # 3. 增加例句功能   # undo 这个应该和ocr功能有关
    # 4. 类似“日本語”这样的词组，会出现问题，主要在单词查询的过程有问题，原因是分词器的问题