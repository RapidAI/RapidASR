# -*- coding: UTF-8 -*-
'''
Project -> File   ：RapidASR-2.0.0 -> rapid_punc.py
Author ：standy
Date   ：2023/5/3 11:45
'''

from pathlib import Path
from typing import  Union, Tuple
import numpy as np

from .utils import (OrtInferSession, read_yaml)
from .utils import (TokenIDConverter, split_to_mini_sentence,code_mix_split_words)
import warnings
from typeguard import check_argument_types
cur_dir = Path(__file__).resolve().parent

class PuncParaformer():

    def __init__(self, config_path: str = None) -> None:
        config = read_yaml(cur_dir / 'punc_model/punc.yaml')

        if config_path:
            config = read_yaml(config_path)

        self.converter = TokenIDConverter(**config['TokenIDConverter']) #转换器
        self.ort_infer = OrtInferSession(config['Model']) #推理器
        self.batch_size = 1
        self.punc_list = config['punc_list']
        self.period = 0
        for i in range(len(self.punc_list)):
            if self.punc_list[i] == ",":
                self.punc_list[i] = "，"
            elif self.punc_list[i] == "?":
                self.punc_list[i] = "？"
            elif self.punc_list[i] == "。":
                self.period = i

    def __call__(self, text: Union[list, str], split_size=20):
        check_argument_types()
        split_text = code_mix_split_words(text)
        split_text_id = self.converter.tokens2ids(split_text)
        mini_sentences = split_to_mini_sentence(split_text, split_size)
        mini_sentences_id = split_to_mini_sentence(split_text_id, split_size)
        assert len(mini_sentences) == len(mini_sentences_id)
        cache_sent = []
        cache_sent_id = []
        new_mini_sentence = ""
        new_mini_sentence_punc = []
        cache_pop_trigger_limit = 200
        for mini_sentence_i in range(len(mini_sentences)):
            mini_sentence = mini_sentences[mini_sentence_i]
            mini_sentence_id = mini_sentences_id[mini_sentence_i]
            mini_sentence = cache_sent + mini_sentence
            mini_sentence_id = np.array(cache_sent_id + mini_sentence_id, dtype='int64')
            data = {
                "text": mini_sentence_id[None,:],
                "text_lengths": np.array([len(mini_sentence_id)], dtype='int32'),
            }
            try:
                outputs = self.infer(data['text'], data['text_lengths']) # 推理
                y = outputs[0] # (1, seq_len, num_class)
                punctuations = np.argmax(y,axis=-1) #返回最大值的索引
                assert punctuations.size == len(mini_sentence) # 如果取得索引的长度和句子长度不一致，报错
            except Exception as e:
                warnings.warn(f'Error occurs when processing {mini_sentence}. Error message: {e}')
                # punctuations = np.zeros(len(mini_sentence), dtype='int64')
            # 搜索最后一个句号/问号作为缓存
            if mini_sentence_i < len(mini_sentences) - 1:
                sentenceEnd = -1
                last_comma_index = -1
                for i in range(len(punctuations) - 2, 1, -1):
                    if self.punc_list[punctuations[i]] == "。" or self.punc_list[punctuations[i]] == "？":
                        sentenceEnd = i
                        break
                    if last_comma_index < 0 and self.punc_list[punctuations[i]] == "，":
                        last_comma_index = i

                if sentenceEnd < 0 and len(mini_sentence) > cache_pop_trigger_limit and last_comma_index >= 0:
                    # 这句话太长了，用逗号隔开了。直接把逗号前面的部分作为一句话
                    sentenceEnd = last_comma_index
                    punctuations[sentenceEnd] = self.period
                cache_sent = mini_sentence[sentenceEnd + 1:]
                cache_sent_id = mini_sentence_id[sentenceEnd + 1:].tolist()
                mini_sentence = mini_sentence[0:sentenceEnd + 1]
                punctuations = punctuations[0:sentenceEnd + 1]
            # print(punctuations)
            new_mini_sentence_punc += [int(x) for x in punctuations]
            words_with_punc = []
            for i in range(len(mini_sentence)):
                if i > 0:
                    if len(mini_sentence[i][0].encode()) == 1 and len(mini_sentence[i - 1][0].encode()) == 1:
                        mini_sentence[i] = " " + mini_sentence[i]
                words_with_punc.append(mini_sentence[i])
                if self.punc_list[punctuations[i]] != "_":
                    words_with_punc.append(self.punc_list[punctuations[i]])
            new_mini_sentence += "".join(words_with_punc)
            # 在句末加上句号
            new_mini_sentence_out = new_mini_sentence
            new_mini_sentence_punc_out = new_mini_sentence_punc
            if mini_sentence_i == len(mini_sentences) - 1:
                if new_mini_sentence[-1] == "，" or new_mini_sentence[-1] == "、":
                    new_mini_sentence_out = new_mini_sentence[:-1] + "。"
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
                elif new_mini_sentence[-1] != "。" and new_mini_sentence[-1] != "？":
                    new_mini_sentence_out = new_mini_sentence + "。"
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
        # print(new_mini_sentence_out, new_mini_sentence_punc_out)
        return new_mini_sentence_out, new_mini_sentence_punc_out

    def infer(self, feats: np.ndarray,
              feats_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer([feats, feats_len])
        return outputs


