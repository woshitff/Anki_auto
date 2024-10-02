#*- coding: utf-8 -*-
from .text_processing import JapText2Fields

#-------core_2000_fields-------#
class Core_2000_Fields:
    def __init__(self, word, timestamp_generator):
        self.word = word
        self.fields_dict = {
            "Optimized-Voc-Index": '',
            "Vocabulary-Kanji": '',
            "Vocabulary-Furigana": '',
            "Vocabulary-Kana": '',
            "Vocabulary-English": '',
            "Vocabulary-Audio": '',
            "Vocabulary-Pos": '',
            "Caution": '',
            "Expression": '',
            "Reading": '',
            "Sentence-Kana": '',
            "Sentence-English": '',
            "Sentence-Clozed": '',
            "Sentence-Audio": '',
            "Notes": '',
            "Core-Index": '',
            "Optimized-Sent-Index": '',
            "Frequency": ''
        }

        self.word_fields = JapText2Fields(word)
        self.sentence_fields = JapText2Fields(word)
        self.generator = timestamp_generator
        self.vocab_audio_path = f"C:\\Users\\chait\\Desktop\\Anki-auto\\Dataset\\Audio\\{self.word}.mp3"

    def assign_all_fields(self):
        self.fields_dict['Optimized-Voc-Index'] = self.generator.generate_index()
        self.fields_dict['Vocabulary-Kanji'] = self.word
        self.fields_dict['Vocabulary-Furigana'] = self.word_fields.kanji_to_furigana()
        self.fields_dict['Vocabulary-Kana'] = self.word_fields.kanji_to_hiragana()
        self.fields_dict['Vocabulary-English'] = self.word_fields.get_word_meaning()
        self.fields_dict['Vocabulary-Audio'] = self.word_fields.get_text_audio(self.vocab_audio_path)
        self.fields_dict['Vocabulary-Pos'] = self.word_fields.get_word_pos()
        self.fields_dict['Caution'] = ''
        self.fields_dict['Expression'] = ''
        self.fields_dict['Reading'] = ''
        self.fields_dict['Sentence-Kana'] = ''
        self.fields_dict['Sentence-English'] = ''
        self.fields_dict['Sentence-Clozed'] = ''
        self.fields_dict['Sentence-Audio'] = ''
        self.fields_dict['Notes'] = ''
        self.fields_dict['Core-Index'] = ''
        self.fields_dict['Optimized-Sent-Index'] = ''
        self.fields_dict['Frequency'] = ''

    def __call__(self):
        self.assign_all_fields()
        self.fields_dict['Vocabulary-Audio'] = f"[sound:{self.vocab_audio_path}]"
        return self.fields_dict

    
        