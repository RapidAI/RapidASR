# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
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

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import multiprocessing
import time
from pathlib import Path

import numpy as np
import onnxruntime as rt
import yaml

from swig_decoders import (PathTrie, TrieVector,
                            ctc_beam_search_decoder_batch, map_batch)
from .utils import (compute_fbank, filter_wav, load_dict, padding, parse_raw,
                    read_symbol_table, resample, tokenize)

IGNORE_ID = -1


class WenetInfer(object):
    def __init__(self, config_path, dict_path, gpu,
                 encoder_onnx_path, decoder_onnx_path, mode):
        self.gpu = gpu
        self.mode = mode

        with open(config_path, 'r') as f:
            self.conf = yaml.load(f, Loader=yaml.FullLoader)['dataset_conf']
        self.symbol_table = read_symbol_table(dict_path)

        self.vocabulary, self.char_dict = load_dict(dict_path)
        self.eos = self.sos = len(self.char_dict) - 1

        self.encoder_onnx = encoder_onnx_path
        self.decoder_onnx = decoder_onnx_path

        self.load_model()

    def load_model(self):
        if self.gpu >= 0:
            EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            EP_list = ['CPUExecutionProvider']

        if not Path(self.encoder_onnx).exists():
            raise FileNotFoundError(f'The {self.encoder_onnx} does not exits!')

        if not Path(self.decoder_onnx).exists():
            raise FileNotFoundError(f'The {self.decoder_onnx} dost not exists!')

        self.encoder_ort_session = rt.InferenceSession(self.encoder_onnx,
                                                       providers=EP_list)
        self.decoder_ort_session = None
        if self.mode == "attention_rescoring":
            self.decoder_ort_session = rt.InferenceSession(self.decoder_onnx,
                                                           providers=EP_list)

    def preprocess_data(self, wav_path):
        data = dict(src=wav_path)
        data = parse_raw(data)
        if data is None:
            raise ValueError

        data = tokenize(data, self.symbol_table)

        filter_conf = self.conf.get('filter_conf', {})
        data = filter_wav(data, **filter_conf)
        if data is None:
            return None

        resample_conf = self.conf.get('resample_conf', {})
        data = resample(data, **resample_conf)

        fbank_conf = self.conf.get('fbank_conf', {})
        data = compute_fbank(data, **fbank_conf)

        data = padding([data])
        return data

    def __call__(self, wav_path):
        s = time.time()
        data = self.preprocess_data(wav_path)

        keys, feats, feats_lengths = data
        feats = feats.astype(np.float32)
        feats_lengths = feats_lengths.astype(np.int32)

        ort_inputs = {
            self.encoder_ort_session.get_inputs()[0].name: feats,
            self.encoder_ort_session.get_inputs()[1].name: feats_lengths
        }

        ort_outs = self.encoder_ort_session.run(None, ort_inputs)

        encoder_out, encoder_out_lens, ctc_log_probs, \
            beam_log_probs, beam_log_probs_idx = ort_outs

        beam_size = beam_log_probs.shape[-1]
        batch_size = beam_log_probs.shape[0]

        num_processes = min(multiprocessing.cpu_count(), batch_size)

        if self.mode == 'ctc_greedy_search':
            if beam_size != 1:
                log_probs_idx = beam_log_probs_idx[:, :, 0]

            batch_sents = []
            for idx, seq in enumerate(log_probs_idx):
                batch_sents.append(seq[0:encoder_out_lens[idx]].tolist())

            hyps = map_batch(batch_sents, self.vocabulary,
                             num_processes, True, 0)

        elif self.mode in ('ctc_prefix_beam_search', "attention_rescoring"):
            batch_log_probs_seq_list = beam_log_probs.tolist()
            batch_log_probs_idx_list = beam_log_probs_idx.tolist()
            batch_len_list = encoder_out_lens.tolist()
            batch_log_probs_seq = []
            batch_log_probs_ids = []
            batch_start = []  # only effective in streaming deployment
            batch_root = TrieVector()
            root_dict = {}

            for i in range(len(batch_len_list)):
                num_sent = batch_len_list[i]
                batch_log_probs_seq.append(
                    batch_log_probs_seq_list[i][0:num_sent])
                batch_log_probs_ids.append(
                    batch_log_probs_idx_list[i][0:num_sent])
                root_dict[i] = PathTrie()
                batch_root.append(root_dict[i])
                batch_start.append(True)

            score_hyps = ctc_beam_search_decoder_batch(batch_log_probs_seq,
                                                       batch_log_probs_ids,
                                                       batch_root,
                                                       batch_start,
                                                       beam_size,
                                                       num_processes,
                                                       0, -2, 0.99999)
        if self.mode == 'ctc_prefix_beam_search':
            hyps = []
            for cand_hyps in score_hyps:
                hyps.append(cand_hyps[0][1])
            hyps = map_batch(hyps, self.vocabulary, num_processes, False, 0)

        if self.mode == 'attention_rescoring':
            ctc_score, all_hyps = [], []
            max_len = 0
            for hyps in score_hyps:
                cur_len = len(hyps)
                if len(hyps) < beam_size:
                    hyps += (beam_size - cur_len) * [(-float("INF"), (0,))]

                cur_ctc_score = []
                for hyp in hyps:
                    cur_ctc_score.append(hyp[0])
                    all_hyps.append(list(hyp[1]))
                    if len(hyp[1]) > max_len:
                        max_len = len(hyp[1])
                ctc_score.append(cur_ctc_score)

            ctc_score = np.array(ctc_score, dtype=np.float32)

            hyps_pad_sos_eos = np.ones(
                (batch_size, beam_size, max_len + 2), dtype=np.int64) * IGNORE_ID

            r_hyps_pad_sos_eos = np.ones(
                (batch_size, beam_size, max_len + 2), dtype=np.int64) * IGNORE_ID

            hyps_lens_sos = np.ones(
                (batch_size, beam_size), dtype=np.int32)

            k = 0
            for i in range(batch_size):
                for j in range(beam_size):
                    cand = all_hyps[k]
                    l = len(cand) + 2
                    hyps_pad_sos_eos[i][j][0:l] = [
                        self.sos] + cand + [self.eos]
                    r_hyps_pad_sos_eos[i][j][0:l] = [
                        self.sos] + cand[::-1] + [self.eos]
                    hyps_lens_sos[i][j] = len(cand) + 1
                    k += 1

            decoder_ort_inputs = {
                self.decoder_ort_session.get_inputs()[0].name: encoder_out,
                self.decoder_ort_session.get_inputs()[1].name: encoder_out_lens,
                self.decoder_ort_session.get_inputs()[2].name: hyps_pad_sos_eos,
                self.decoder_ort_session.get_inputs()[3].name: hyps_lens_sos,
                self.decoder_ort_session.get_inputs()[-1].name: ctc_score
            }

            reverse_weight = self.conf.get('reverse_weight', 0.0)
            if reverse_weight > 0:
                r_hyps_pad_sos_eos_name = self.decoder_ort_session.get_inputs()[
                    4].name
                decoder_ort_inputs[r_hyps_pad_sos_eos_name] = r_hyps_pad_sos_eos

            best_index = self.decoder_ort_session.run(
                None, decoder_ort_inputs)[0]
            best_sents = []
            k = 0
            for idx in best_index:
                cur_best_sent = all_hyps[k: k + beam_size][idx]
                best_sents.append(cur_best_sent)
                k += beam_size
            hyps = map_batch(best_sents, self.vocabulary, num_processes)

        elapse = time.time() - s
        return keys, hyps[0], elapse
