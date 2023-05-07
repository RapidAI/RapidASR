import os

import librosa
import numpy as np
import warnings
from .utils import (OrtInferSession,read_yaml)
from .utils import WavFrontend
from .utils import E2EVadModel
from pathlib import Path
from typing import  Union, Tuple, List
cur_dir = Path(__file__).resolve().parent

class RapidVad():
    def __init__(self, config_path: str = None,max_end_sil:int = None) -> None:
        config = read_yaml(cur_dir / 'vad_model/vad.yaml')

        if config_path:
            config = read_yaml(config_path)

        cmvn_file = os.path.join(cur_dir / 'vad_model/vad.mvn')

        self.frontend_vad = WavFrontend(
            cmvn_file=cmvn_file,
            **config['frontend_conf']
        )

        # self.ort_infer = OrtInferSession(model_file, device_id, intra_op_num_threads=intra_op_num_threads)
        self.ort_infer = OrtInferSession(config['Model'])
        self.batch_size = config['batch_size']
        self.vad_scorer = E2EVadModel(config["vad_post_conf"])
        self.max_end_sil = max_end_sil if max_end_sil is not None else config["vad_post_conf"]["max_end_silence_time"]
        self.encoder_conf = config["encoder_conf"]

    def prepare_cache(self, in_cache: list = []):
        if len(in_cache) > 0:
            return in_cache
        fsmn_layers = self.encoder_conf["fsmn_layers"]
        proj_dim = self.encoder_conf["proj_dim"]
        lorder = self.encoder_conf["lorder"]
        for i in range(fsmn_layers):
            cache = np.zeros((1, proj_dim, lorder - 1, 1)).astype(np.float32)
            in_cache.append(cache)
        return in_cache

    def __call__(self, audio_in: Union[str, np.ndarray, List[str]], **kwargs) -> List:
        # waveform = self.load_data(audio_in, self.frontend.fs)
        waveform = librosa.load(audio_in)[0][None, ...] # 读取音频 ，并转换为二维数组

        segments = [[]] * self.batch_size
        speech, _ = self.frontend_vad.forward_fbank(waveform) # 提取特征
        feats, feats_len = self.frontend_vad.forward_lfr_cmvn(speech) # 提取特征
        # print(feats.shape, feats_len.shape)

        is_final = kwargs.get('kwargs', False)
        waveform = np.array(waveform)
        param_dict = kwargs.get('param_dict', dict())
        in_cache = param_dict.get('in_cache', list())
        in_cache = self.prepare_cache(in_cache)
        try:
            t_offset = 0
            step = int(min(feats_len.max(), 6000))
            for t_offset in range(0, int(feats_len), min(step, feats_len - t_offset)):
                if t_offset + step >= feats_len - 1:
                    step = feats_len - t_offset
                    is_final = True
                else:
                    is_final = False
                feats_package = feats[:, t_offset:int(t_offset + step), :]
                waveform_package = waveform[:,
                                   t_offset * 160:min(waveform.shape[-1], (int(t_offset + step) - 1) * 160 + 400)]

                inputs = [feats_package]
                # inputs = [feats]
                inputs.extend(in_cache)
                scores, out_caches = self.infer(inputs)
                in_cache = out_caches
                segments_part = self.vad_scorer(scores, waveform_package, is_final=is_final,
                                                max_end_sil=self.max_end_sil, online=False)
            # segments = self.vad_scorer(scores, waveform[0][None, :], is_final=is_final, max_end_sil=self.max_end_sil)

                if segments_part:
                    for batch_num in range(0, self.batch_size):
                        segments[batch_num] += segments_part[batch_num]

        except Exception as e:
            segments = ''
            warnings.warn("input wav is silence or noise")


        return segments

    def load_data(self,
                  wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(
            f'The type of {wav_content} is not in [str, np.ndarray, list]')

    def extract_feat(self,
                     waveform_list: List[np.ndarray]
                     ) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []

        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len)) # 填充特征
        feats_len = np.array(feats_len).astype(np.int32)[0]
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))

            feat = np.squeeze(feat, axis=0) # 去掉第一维
            print(feat.shape)
            return np.pad(feat, pad_width, 'constant', constant_values=0)


        feat_res = [pad_feat(feat, feat.shape[1]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(self, feats: List) -> Tuple[np.ndarray, np.ndarray]:

        outputs = self.ort_infer(feats)

        # scores, out_caches = outputs[0], outputs[1:]
        return outputs,[]
