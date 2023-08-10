# -*- coding: UTF-8 -*-
'''
Project -> File   ：RapidASR-2.0.0 -> test_vad
Author ：standy
Date   ：2023/5/3 15:57
function ： #测试vad
'''

from rapid_paraformer.rapid_vad import RapidVad
# model_dir = "/Users/laichunping/Documents/ASR/FunASR/export/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
wav_path = "/Users/laichunping/Documents/ASR/RapidASR-2.0.0/test_wavs/0478_00017.wav"
model = RapidVad()

#offline vad
result = model(wav_path)
print(result)