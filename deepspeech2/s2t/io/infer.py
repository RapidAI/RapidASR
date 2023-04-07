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
import os
import sys
from collections import OrderedDict
from typing import Optional, Union

import paddle
import soundfile
from ..deepspeech2 import DeepSpeech2ModelOnline
from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.utils.utility import UpdateConfig
from yacs.config import CfgNode


__all__ = ['ASRExecutor']

model_alias = {
    "deepspeech2offline":
    "paddlespeech.s2t.models.ds2:DeepSpeech2Model",
    "deepspeech2online":
    "paddlespeech.s2t.models.ds2_online:DeepSpeech2ModelOnline",
    "conformer":
    "paddlespeech.s2t.models.u2:U2Model",
    "transformer":
    "paddlespeech.s2t.models.u2:U2Model",
    "wenetspeech":
    "paddlespeech.s2t.models.u2:U2Model",
}


class ASRExecutor(object):
    def __init__(self,
                 model: str = 'conformer_wenetspeech',
                 lang: str = 'zh',
                 sample_rate: int = 16000,
                 config: os.PathLike = None,
                 onnx_path: os.PathLike = None,
                 decode_method: str = 'attention_rescoring',
                 language_model_dir=None):
        self.model = model
        self.lang = lang
        self.sample_rate = sample_rate
        self.config = config
        self.onnx_path = onnx_path
        self.decode_method = decode_method

        self._inputs = OrderedDict()
        self._outputs = OrderedDict()

        self._init_from_path(self.model, self.lang, self.sample_rate,
                             self.config, self.decode_method,
                             self.onnx_path)

    def __call__(self, audio_file: os.PathLike, force_yes: bool = False):
        audio_file = os.path.abspath(audio_file)
        if not self._check(audio_file, self.sample_rate, force_yes):
            sys.exit(-1)

        self.preprocess(audio_file)
        self.infer()
        res = self.postprocess()  # Retrieve result of asr.
        return res

    def _init_from_path(self,
                        model_type: str = 'wenetspeech',
                        lang: str = 'zh',
                        sample_rate: int = 16000,
                        cfg_path: Optional[os.PathLike] = None,
                        decode_method: str = 'attention_rescoring',
                        onnx_path: Optional[os.PathLike] = None):
        """
        Init model and other resources from a specific path.
        """
        if hasattr(self, 'model'):
            print('Model had been initialized.')
            return

        self.cfg_path = os.path.abspath(cfg_path)
        self.res_path = os.path.dirname(
            os.path.dirname(os.path.abspath(self.cfg_path)))

        # Init body.
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(self.cfg_path)

        with UpdateConfig(self.config):
            self.vocab = self.config.vocab_filepath

            self.config.decode.lang_model_path = os.path.join(
                MODEL_HOME, 'language_model',
                self.config.decode.lang_model_path)

            self.collate_fn_test = SpeechCollator.from_config(self.config)

            self.text_feature = TextFeaturizer(unit_type=self.config.unit_type,
                                               vocab=self.vocab)

        # model_type: {model_name}_{dataset}
        model_name = model_type[:model_type.rindex('_')]
        model_class = DeepSpeech2ModelOnline()
        model_conf = self.config
        model = model_class.from_config(model_conf)
        self.model = model

    def preprocess(self, input: Union[str, os.PathLike]):
        """
        Input preprocess and return paddle.Tensor stored in self.input.
        Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """

        audio_file = input
        if isinstance(audio_file, (str, os.PathLike)):
            print("Preprocess audio_file:" + audio_file)

        # Get the object for feature extraction
        audio, _ = self.collate_fn_test.process_utterance(
            audio_file=audio_file, transcript=" ")

        audio_len = audio.shape[0]
        audio = paddle.to_tensor(audio, dtype='float32')

        audio_len = paddle.to_tensor(audio_len)
        audio = paddle.unsqueeze(audio, axis=0)

        # vocab_list = collate_fn_test.vocab_list
        self._inputs["audio"] = audio
        self._inputs["audio_len"] = audio_len
        print(f"audio feat shape: {audio.shape}")

    def infer(self):
        """
        Model inference and result stored in self.output.
        """
        cfg = self.config.decode
        audio = self._inputs["audio"]
        audio_len = self._inputs["audio_len"]

        decode_batch_size = audio.shape[0]
        self.model.decoder.init_decoder(
            decode_batch_size, self.text_feature.vocab_list,
            cfg.decoding_method, cfg.lang_model_path, cfg.alpha, cfg.beta,
            cfg.beam_size, cfg.cutoff_prob, cfg.cutoff_top_n,
            cfg.num_proc_bsearch)

        result_transcripts = self.model.decode(audio, audio_len)
        self.model.decoder.del_decoder()
        self._outputs["result"] = result_transcripts[0]

    def postprocess(self) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        return self._outputs["result"]

    def _check(self, audio_file: str, sample_rate: int, force_yes: bool):
        self.sample_rate = sample_rate
        if self.sample_rate != 16000 and self.sample_rate != 8000:
            print(
                "invalid sample rate, please input --sr 8000 or --sr 16000")
            return False

        if isinstance(audio_file, (str, os.PathLike)):
            if not os.path.isfile(audio_file):
                print("Please input the right audio file path")
                return False

        print("checking the audio file format......")
        try:
            audio, audio_sample_rate = soundfile.read(
                audio_file, dtype="int16", always_2d=True)
        except Exception as e:
            print(
                "can not open the audio file, please check the audio file format is 'wav'. \n \
                 you can try to use sox to change the file format.\n \
                 For example: \n \
                 sample rate: 16k \n \
                 sox input_audio.xx --rate 16k --bits 16 --channels 1 output_audio.wav \n \
                 sample rate: 8k \n \
                 sox input_audio.xx --rate 8k --bits 16 --channels 1 output_audio.wav \n \
                 ")
            return False

        print("The sample rate is %d" % audio_sample_rate)
        if audio_sample_rate != self.sample_rate:
            print("The sample rate of the input file is not {}.\n \
                            The program will resample the wav file to {}.\n \
                            If the result does not meet your expectations，\n \
                            Please input the 16k 16 bit 1 channel wav file. \
                        ".format(self.sample_rate, self.sample_rate))
            if force_yes is False:
                while (True):
                    print(
                        "Whether to change the sample rate and the channel. Y: change the sample. N: exit the prgream."
                    )
                    content = input("Input(Y/N):")
                    if content.strip() == "Y" or content.strip(
                    ) == "y" or content.strip() == "yes" or content.strip(
                    ) == "Yes":
                        print(
                            "change the sampele rate, channel to 16k and 1 channel"
                        )
                        break
                    elif content.strip() == "N" or content.strip(
                    ) == "n" or content.strip() == "no" or content.strip(
                    ) == "No":
                        print("Exit the program")
                        exit(1)
                    else:
                        print("Not regular input, please input again")

            self.change_format = True
        else:
            print("The audio file format is right")
            self.change_format = False

        return True
