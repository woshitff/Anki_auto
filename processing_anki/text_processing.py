import json
from typing import List
import time

import fugashi
import requests
from bs4 import BeautifulSoup
import re
from gtts import gTTS
#-------文件读写--------

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
#-------文本转fields的一些工具函数-------
class JapText2Fields():
    def __init__(self, text: str='飲んで'):
        self.text = text
        self.tagger = fugashi.Tagger()
        self.words = self.tagger(text) # type: list[fugashi.Word] # each word is a fugashi.Word object 
        # eg： [飲ん, で] <class 'fugashi.fugashi.UnidicNode'>

    def katakana_to_hiragana(self,katakana):
        """
        convert katakana to hiragana
        """
        hiragana = ''.join(chr(ord(char) - ord('ァ') + ord('ぁ')) if 'ァ' <= char <= 'ヶ' else char for char in katakana)
        return hiragana

    def kanji_to_hiragana(self):  
        """
        convert kanji to hiragana
        """
        hiragana = ''.join(
            [self.katakana_to_hiragana(word.feature.kana) for word in self.words if word.feature.kana]
            ) 
        return hiragana

    def kanji_to_furigana(self):
        """
        convert kanji to furigana
        """
        furigana = ''.join([
            f'{word.surface}[{self.katakana_to_hiragana(word.feature.kana)}]'
            if re.search(r'[\u4e00-\u9fff]', word.surface) else word.surface
            for word in self.words
        ])
        return furigana

    def get_word_pos(self):
        """
        get the first word's posision
        """
        word = self.words[0]
        pos = word.feature.pos1
        return pos

    def get_word_meaning(self):
        """
        get the word's meaning from tangorin.com

        Parameters:
            text: str, the word to be searched for its meaning
        Returns:
            meanings: list, a list of meanings of the word
        Example:
            >>> get_word_meaning('日本語')
            output: ['Japanese language']
        """
        lemma = self.text
        print(f"查询单词：{lemma}")
        url = f"https://tangorin.com/words?search={lemma}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"网络请求失败，状态码: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        meanings = []
        results_wrapper = soup.find('div', class_='ResultsWrapper')
        if results_wrapper:
            for li in results_wrapper.find('li', lang='en'):
                meaning = li.get_text(strip=True)
                meanings.append(meaning)
        if not meanings:
            print("未找到任何含义")
        meanings = ''.join(meanings)
        print(f"含义：{meanings}")
        return meanings

    def get_text_audio(self, audio_path):
        """
        get the text's audio using TTS model of tacotron2-DDC using dataset of kokoro
        """
        tts = gTTS(text=self.text, lang='ja')
        tts.save(audio_path)
        return audio_path

if __name__ == '__main__':
    A = JapText2Fields('日本語')
    # for char in A.words:
    #     print(char.feature)
    # print(A.words)
    # # print(type(A.words[0]))
    # print(A.kanji_to_hiragana())
    # print(type(A.kanji_to_hiragana()))
    # print(A.kanji_to_furigana())

    # print(A.get_word_pos())
    print(A.get_word_meaning())
    # A.get_text_audio('audio.wav')
    

