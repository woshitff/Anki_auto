import requests

import argparse

def arg_parse():
    arg_parser = argparse.ArgumentParser()

    arg_parser.description = "Generate Anki cards from JSON file"

    arg_parser.add_argument('--deck_name', type=str, 
                            default='アカネ', help='Deck name')
    arg_parser.add_argument('--model_name', type=str, 
                            default='Core 2000', help='Model name')

    return arg_parser.parse_args()
# -------生成anki卡片-------

def invoke(action, params={}):
    request_json = {
        'action': action,
        'version': 6,
        'params': params
    }
    response = requests.post('http://localhost:8765', json=request_json).json()
    if len(response) != 2:
        raise Exception('Invalid response from AnkiConnect')
    if response.get('error') is not None:
        raise Exception(response['error'])
    return response['result']

def get_all_model_names():
    return invoke('modelNames')

def get_model_templates(model_name):
    return invoke('modelTemplates', {'modelName': model_name})

def get_model_field_names(model_name):
    return invoke('modelFieldNames', {'modelName': model_name})

def language_fields_generator(word, model_name):
    field_list = get_model_field_names(model_name)
    field_dict = {key: '' for key in field_list}
    pass        # todo
    #return a dictionary of fields

def add_note(deck_name, model_name, fields_data, tags=[]):
    note = {
        "deckName": deck_name,
        "modelName": model_name,
        "fields": fields_data,
        "tags": tags
    }
    return invoke('addNote', {'note': note})

if __name__ == '__main__':
    # add_note('アカネ', 'Core 2000', {
    #                     "Optimized-Voc-Index": 'optimized_voc_index',
    #                     "Vocabulary-Kanji": 'kanji',
    #                     "Vocabulary-Furigana": 'meaning',
    #                     "Vocabulary-Kana": 'kana',
    #                     "Vocabulary-English": 'english',
    #                     "Vocabulary-Audio": 'audio',
    #                     "Vocabulary-Pos": 'example',
    #                     "Caution": 'caution',
    #                     "Expression": 'expression',
    #                     "Reading": 'reading',
    #                     "Sentence-Kana": 'sentence_kana',
    #                     "Sentence-English": 'sentence_english',
    #                     "Sentence-Clozed": 'sentence_clozed',
    #                     "Sentence-Audio": 'sentence_audio',
    #                     "Notes": 'notes',
    #                     "Core-Index": 'core_index',
    #                     "Optimized-Sent-Index": 'optimized_sent_index',
    #                     "Frequency": 'frequency'
    #              })
    field_names = get_model_field_names('Core 2000')
    print(field_names)
