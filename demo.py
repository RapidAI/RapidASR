# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapid_paraformer import RapidParaformer


paraformer = RapidParaformer()

wav_path = 'test_wavs/0478_00017.wav'
print(wav_path)
result = paraformer(str(wav_path))
print(result)
