# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapid_paraformer import RapidParaformer


config_path = 'resources/config.yaml'

paraformer = RapidParaformer(config_path)

wav_path = [
    'test_wavs/0478_00017.wav',
    'test_wavs/asr_example_zh.wav',
    'test_wavs/0478_00017.wav',
    'test_wavs/asr_example_zh.wav',
    'test_wavs/0478_00017.wav',
    'test_wavs/asr_example_zh.wav',
]

print(wav_path)
# wav_path = 'test_wavs/0478_00017.wav'
result = paraformer(wav_path)
print(result)
