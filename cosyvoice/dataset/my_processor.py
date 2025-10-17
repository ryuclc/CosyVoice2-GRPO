# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random

import pyarrow.parquet as pq
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pyworld as pw


AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}


def is_mixed_language(s):
    # 定义各语言的Unicode范围
    lang_ranges = {
        'chinese': (0x4e00, 0x9fff),        # 中文
        'english': (0x0041, 0x005a),        # 英文大写
        'english_lower': (0x0061, 0x007a),  # 英文小写
        'japanese': (0x3040, 0x30ff),       # 日文
        'korean': (0xac00, 0xd7af),         # 韩文
        'numbers': (0x0030, 0x0039)         # 数字（作为中性字符）
    }
    
    # 记录字符串中出现的语言
    present_langs = set()
    
    for char in s:
        code = ord(char)
        lang_found = None
        
        # 检查字符属于哪种语言
        for lang, (start, end) in lang_ranges.items():
            if start <= code <= end:
                # 英文大小写合并为同一类别
                if lang == 'english_lower':
                    lang_found = 'english'
                elif lang == 'chinese' or lang == 'japanese':
                    lang_found = 'chinese+japanese'
                else:
                    lang_found = lang
                break
        
        # if lang_found and lang_found != 'numbers':  # 忽略数字的影响
        if lang_found:
            present_langs.add(lang_found)
            
        # 一旦发现两种及以上语言，可提前返回
        if len(present_langs) >= 2:
            return True
    
    return len(present_langs) >= 2


def filter_mix_lang(data, mode='train'):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        if is_mixed_language(sample['text']):
            continue

        yield sample

