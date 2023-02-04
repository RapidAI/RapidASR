# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapid_paraformer import RapidParaformer


paraformer = RapidParaformer()

wav_file = 'test_wavs/0478_00017.wav'
result = paraformer(wav_file)
print(result)
