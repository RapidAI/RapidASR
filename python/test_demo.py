# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from wenet import WenetInfer

config_path = "pretrain_model/20211025_conformer_exp/test.yaml"
dict_path = "pretrain_model/20211025_conformer_exp/words.txt"
encoder_onnx_path = "pretrain_model/20211025_conformer_exp/encoder.onnx"
decoder_onnx_path = "pretrain_model/20211025_conformer_exp/decoder.onnx"
mode = 'attention_rescoring'

wenet_infer = WenetInfer(config_path,
                         dict_path, -1,
                         encoder_onnx_path,
                         decoder_onnx_path,
                         mode)

wav_path = 'test_data/test.wav'
key, content, elapse = wenet_infer(wav_path)
print(f'{key}\t{content}\t{elapse}s')
