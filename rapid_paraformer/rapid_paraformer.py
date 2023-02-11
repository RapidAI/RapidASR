# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import traceback
from pathlib import Path
from typing import List

import librosa
import numpy as np

from .utils import (CharTokenizer, Hypothesis, ONNXRuntimeError, OrtInferSession,
                    TokenIDConverter, WavFrontend, read_yaml, get_logger,
                    OpenVINOInferSession)

cur_dir = Path(__file__).resolve().parent
logging = get_logger()


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
        self.vino_infer = OpenVINOInferSession(config['Model'])

    def __call__(self, wav_path: str) -> List:
        waveform = librosa.load(wav_path)[0][None, ...]

        speech, _ = self.frontend_asr.forward_fbank(waveform)
        feats, feats_len = self.frontend_asr.forward_lfr_cmvn(speech)
        try:
            # am_scores = self.ort_infer(input_content=[feats, feats_len])
            am_scores = self.vino_infer(input_content=[feats, feats_len])
        except ONNXRuntimeError:
            logging.error(traceback.format_exc())
            return []

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
        return infer_res


if __name__ == '__main__':
    paraformer = RapidParaformer()

    wav_file = '0478_00017.wav'
    for i in range(1000):
        result = paraformer(wav_file)
        print(result)
