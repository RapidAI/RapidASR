# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: main.py
# @Author: SWHL
# @Contact: liekkaskono@163.com
from deepspeech2 import ASRExecutor

config_path = 'resources/model.yaml'
model_path = 'resources/models/asr0_deepspeech2_online_aishell_ckpt_0.2.0.onnx'
lan_model_path = 'resources/models/language_model/zh_giga.no_cna_cmn.prune01244.klm'
wav_path = 'test_wav/zh.wav'

asr_executor = ASRExecutor(sample_rate=16000,
                           config_path=config_path,
                           onnx_path=model_path,
                           lan_model_path=lan_model_path)

text = asr_executor(audio_file=wav_path)

print('ASR Result: \t{}'.format(text))
