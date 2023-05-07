# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import List,Union

import librosa
import numpy as np

from .utils import (CharTokenizer, Hypothesis, OrtInferSession,
                    TokenIDConverter, WavFrontend, read_yaml)

cur_dir = Path(__file__).resolve().parent


class RapidParaformer():
    def __init__(self, config_path: str = None) -> None:
        config = read_yaml(cur_dir / 'config.yaml')
        if config_path:
            config = read_yaml(config_path)

        self.converter = TokenIDConverter(**config['TokenIDConverter'])
        self.tokenizer = CharTokenizer(**config['CharTokenizer'])
        self.frontend_asr = WavFrontend(
            cmvn_file=config['WavFrontend']['cmvn_file'],
            **config['WavFrontend']['frontend_conf']
        )
        self.ort_infer = OrtInferSession(config['Model'])

    def __call__(self, wav_path: str) -> List:
        if isinstance(wav_path, str):

            waveform = librosa.load(wav_path)[0][None, ...] # 读取音频文件，并转换为numpy数组
        elif isinstance(wav_path, np.ndarray):
            waveform = self.load_data(wav_path)[0][None, ...] #兼容numpy数组格式数据
        else:
            raise TypeError('wav_path must be str or numpy.ndarray')

        speech, _ = self.frontend_asr.forward_fbank(waveform)
        feats, feats_len = self.frontend_asr.forward_lfr_cmvn(speech)
        try:
            am_scores = self.ort_infer(input_content=[feats, feats_len])
        except Exception as e:
            # raise RuntimeError(f'ONNXRuntime Error: {e}')
            return [[]]


        results = []
        for am_score in am_scores:
            pred_res = self.infer_one_feat(am_score)
            results.append(pred_res)
        return results

    def infer_one_feat(self, am_score: np.ndarray) -> List[str]:
        yseq = am_score.argmax(axis=-1)
        score = am_score.max(axis=-1)
        score = np.sum(score, axis=-1)

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        # asr_model.sos:1  asr_model.eos:2
        yseq = np.array([1] + yseq.tolist() + [2])
        nbest_hyps = [Hypothesis(yseq=yseq, score=score)]

        infer_res = []
        for hyp in nbest_hyps:
            # remove sos/eos and get results
            last_pos = -1
            token_int = hyp.yseq[1:last_pos].tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x not in (0, 2), token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)

            text = self.tokenizer.tokens2text(token)
            infer_res.append(text)
        # print(infer_res)
        return infer_res

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
if __name__ == '__main__':
    paraformer = RapidParaformer()

    wav_file = '/Users/laichunping/Documents/ASR/RapidASR-2.0.0/test_wavs/0478_00017.wav'
    for i in range(1000):
        result = paraformer(wav_file)
        print(result)
