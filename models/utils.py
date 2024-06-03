#!/usr/bin/env python

import os
import re
from transformers import LlamaTokenizerFast

tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "tokenizer")
tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
def get_date_from_query(query, year = True):
    dates = []
    if(not year):
        pattern1 = r'(\d{4}-\d{1,2}-\d{1,2})'
        pattern1 = re.compile(pattern1)
        date1 = pattern1.findall(query)
        dates.extend(date1)
        return dates
    pattern2 = r'(\d{4})'
    pattern2 = re.compile(pattern2)
    date2 = pattern2.findall(query)
    dates.extend(date2)
    return dates
def get_sentences_from_json(entity, info, max_len = 512):
    try: 
        res_info = []
        if(isinstance(info, dict)):
            # jdata = json.dumps(info)
            jdata = str(info)
        else:
            jdata = str(info)

        if(len(jdata) < max_len):
            res_info.append(entity + " " + jdata)
        else:
            sz = len(jdata)
            block = int(sz/max_len)
            for i in range(0, block + 1):
                substr = jdata[i*max_len:(i+1)*max_len]
                res_info.append(entity + " info: " + substr)
    except:
        return []
    return res_info

def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens"""
    max_token_length = 300
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length+1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction
