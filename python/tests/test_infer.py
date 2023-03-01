# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
from pathlib import Path

import pytest
import librosa

project_dir = Path(__file__).resolve().parent.parent
os.sys.path.append(str(project_dir))

from rapid_paraformer import RapidParaformer


cfg_path = project_dir / 'resources' / 'config.yaml'
paraformer = RapidParaformer(cfg_path)


def test_input_by_path():
    wav_file = 'test_wavs/0478_00017.wav'
    result = paraformer(wav_file)
    assert result[0][:5] == '呃说不配合'


def test_input_by_ndarray():
    wav_file = 'test_wavs/0478_00017.wav'
    waveform, _ = librosa.load(wav_file)
    result = paraformer(waveform[None, ...])
    assert result[0][:5] == '呃说不配合'


def test_input_by_str_list():
    wave_list = [
        'test_wavs/0478_00017.wav',
        'test_wavs/asr_example_zh.wav',
    ]
    result = paraformer(wave_list)
    assert result[0][:5] == '呃说不配合'


def test_empty():
    wav_file = None
    with pytest.raises(TypeError) as exc_info:
        paraformer(wav_file)
        raise TypeError()
    assert exc_info.type is TypeError
