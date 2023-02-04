# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
from pathlib import Path

os.sys.path.append(str(Path(__file__).resolve().parent.parent))

from rapid_paraformer import RapidParaformer

paraformer = RapidParaformer()


def test_normal():
    wav_file = 'test_wavs/0478_00017.wav'
    result = paraformer(wav_file)
    assert result[0][0][:5] == '呃说不配合'
