# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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
# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import librosa
import numpy as np
import soundfile as sf

from .kaldifeat import compute_fbank_feats


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_symbol_table(symbol_table_file):
    symbol_table = {}
    with open(symbol_table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table


def load_dict(dict_path):
    vocabulary = []
    char_dict = {}
    with open(dict_path, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2

            char_dict[int(arr[1])] = arr[0]
            vocabulary.append(arr[0])
    return vocabulary, char_dict


def parse_raw(sample):
    assert 'src' in sample

    info = sample['src'].split(' ')
    if len(info) > 1:
        wav_file, txt = sample['src'].split(' ')
    else:
        wav_file = info[0]
        txt = ' '
    key = wav_file

    try:
        waveform, sample_rate = sf.read(wav_file)

        example = dict(key=key, txt=txt, wav=waveform, sample_rate=sample_rate)
        return example
    except Exception as ex:
        raise FileNotFoundError(f'The {wav_file} not be found!')


def filter_wav(sample,
               max_length=10240,
               min_length=10,
               token_max_length=200,
               token_min_length=1,
               min_output_input_ratio=0.0005,
               max_output_input_ratio=1):
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

    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'label' in sample

    num_frames = sample['wav'].shape[0] / sample['sample_rate'] * 100

    if num_frames < min_length or num_frames > max_length:
        return None

    label_length = len(sample['label'])
    if label_length < token_min_length \
            or label_length > token_max_length:
        return None

    if num_frames != 0:
        output_input_ratio = label_length / num_frames
        if output_input_ratio < min_output_input_ratio \
                or output_input_ratio > max_output_input_ratio:
            return None
    return sample


def resample(sample, resample_rate=16000):
    """ Resample data.Inplace operation."""

    assert 'sample_rate' in sample
    assert 'wav' in sample

    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    if sample_rate != resample_rate:
        sample['sample_rate'] = resample_rate
        sample['wav'] = librosa.resample(waveform,
                                         orig_sr=sample_rate,
                                         target_sr=resample_rate)
    return sample


def compute_fbank(sample,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank"""
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
    assert 'label' in sample

    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    waveform = waveform * (1 << 15)

    mat = compute_fbank_feats(waveform,
                              num_mel_bins=num_mel_bins,
                              frame_length=frame_length,
                              frame_shift=frame_shift,
                              dither=dither,
                              energy_floor=0.0,
                              sample_frequency=sample_rate)

    return dict(key=sample['key'],
                label=sample['label'],
                feat=mat)


def tokenize(sample, symbol_table):
    non_lang_syms = {}
    non_lang_syms_pattern = None

    assert 'txt' in sample
    txt = sample['txt'].strip()
    if non_lang_syms_pattern is not None:
        parts = non_lang_syms_pattern.split(txt.upper())
        parts = [w for w in parts if len(w.strip()) > 0]
    else:
        parts = [txt]

    label = []
    tokens = []
    for part in parts:
        if part in non_lang_syms:
            tokens.append(part)
        else:
            for ch in part:
                if ch == ' ':
                    ch = "▁"
                tokens.append(ch)

    for ch in tokens:
        if ch in symbol_table:
            label.append(symbol_table[ch])
        elif '<unk>' in symbol_table:
            label.append(symbol_table['<unk>'])

    sample['tokens'] = tokens
    sample['label'] = label
    return sample


def padding(sample):
    assert isinstance(sample, list)
    sample = sample[0]

    key = sample['key']
    feats = sample['feat']
    feat_length = np.array([feats.shape[0]])
    feats = np.array(feats)[np.newaxis, ...]

    return key, feats, feat_length
