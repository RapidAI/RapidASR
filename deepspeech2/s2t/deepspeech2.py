# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Deepspeech2 ASR Online Model"""
import numpy as np
import onnxruntime as ort

from .modules.ctc import CTCDecoder


class DeepSpeech2ModelOnline(object):
    def __init__(self, encoder_onnx_path):
        self.encoder_sess = ort.InferenceSession(encoder_onnx_path)
        self.decoder = CTCDecoder()

    def decode(self, audio, audio_len):
        onnx_inputs_name = self.encoder_sess.get_inputs()
        ort_inputs = {
            onnx_inputs_name[0].name: np.array(audio).astype(np.float32),
            onnx_inputs_name[1].name: np.array([audio_len]).astype(np.int64),
            onnx_inputs_name[2].name: np.zeros([5, 1, 1024]).astype(np.float32),
            onnx_inputs_name[3].name: np.zeros([5, 1, 1024]).astype(np.float32)
        }
        ort_outputs = self.encoder_sess.run(None, ort_inputs)
        probs, eouts_len, _, _ = ort_outputs

        batch_size = probs.shape[0]
        self.decoder.reset_decoder(batch_size=batch_size)
        self.decoder.next(probs, eouts_len)
        trans_best, trans_beam = self.decoder.decode()
        return trans_best
