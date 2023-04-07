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
try:
    from ..decoders.ctcdecoder import CTCBeamSearchDecoder  # noqa: F401
    from ..decoders.ctcdecoder import Scorer  # noqa: F401
    from ..decoders.ctcdecoder import \
        ctc_beam_search_decoding_batch  # noqa: F401
    from ..decoders.ctcdecoder import ctc_greedy_decoding  # noqa: F401
except ImportError:
    try:
        from ..decoders.ctcdecoder import CTCBeamSearchDecoder  # noqa: F401
        from ..decoders.ctcdecoder import Scorer  # noqa: F401
        from ..decoders.ctcdecoder import \
            ctc_beam_search_decoding_batch  # noqa: F401
        from ..decoders.ctcdecoder import ctc_greedy_decoding  # noqa: F401
    except Exception as e:
        print("paddlespeech_ctcdecoders not installed!")


class CTCDecoder(object):
    def __init__(self):
        # CTCDecoder LM Score handle
        self._ext_scorer = None
        self.beam_search_decoder = None
        self.blank_id = 0

    def _decode_batch_greedy_offline(self, probs_split, vocab_list):
        """This function will be deprecated in future.
        Decode by best path for a batch of probs matrix input.
        :param probs_split: List of 2-D probability matrix, and each consists
                            of prob vectors for one speech utterancce.
        :param probs_split: List of matrix
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        :return: List of transcription texts.
        :rtype: List of str
        """
        results = []
        for i, probs in enumerate(probs_split):
            output_transcription = ctc_greedy_decoding(
                probs_seq=probs, vocabulary=vocab_list, blank_id=self.blank_id)
            results.append(output_transcription)
        return results

    def _init_ext_scorer(self, beam_alpha, beam_beta, language_model_path,
                         vocab_list):
        """Initialize the external scorer.
        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param language_model_path: Filepath for language model. If it is
                                    empty, the external scorer will be set to
                                    None, and the decoding method will be pure
                                    beam search without scorer.
        :type language_model_path: str|None
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        """
        # init once
        if self._ext_scorer is not None:
            return

        if language_model_path != '':
            self._ext_scorer = Scorer(beam_alpha, beam_beta,
                                      language_model_path, vocab_list)
        else:
            self._ext_scorer = None

    def _decode_batch_beam_search_offline(
            self, probs_split, beam_alpha, beam_beta, beam_size, cutoff_prob,
            cutoff_top_n, vocab_list, num_processes):
        """
        This function will be deprecated in future.
        Decode by beam search for a batch of probs matrix input.
        :param probs_split: List of 2-D probability matrix, and each consists
                            of prob vectors for one speech utterancce.
        :param probs_split: List of matrix
        :param beam_alpha: Parameter associated with language model.
        :type beam_alpha: float
        :param beam_beta: Parameter associated with word count.
        :type beam_beta: float
        :param beam_size: Width for Beam search.
        :type beam_size: int
        :param cutoff_prob: Cutoff probability in pruning,
                            default 1.0, no pruning.
        :type cutoff_prob: float
        :param cutoff_top_n: Cutoff number in pruning, only top cutoff_top_n
                        characters with highest probs in vocabulary will be
                        used in beam search, default 40.
        :type cutoff_top_n: int
        :param vocab_list: List of tokens in the vocabulary, for decoding.
        :type vocab_list: list
        :param num_processes: Number of processes (CPU) for decoder.
        :type num_processes: int
        :return: List of transcription texts.
        :rtype: List of str
        """
        if self._ext_scorer is not None:
            self._ext_scorer.reset_params(beam_alpha, beam_beta)

        # beam search decode
        num_processes = min(num_processes, len(probs_split))
        beam_search_results = ctc_beam_search_decoding_batch(
            probs_split=probs_split,
            vocabulary=vocab_list,
            beam_size=beam_size,
            num_processes=num_processes,
            ext_scoring_func=self._ext_scorer,
            cutoff_prob=cutoff_prob,
            cutoff_top_n=cutoff_top_n,
            blank_id=self.blank_id)

        results = [result[0][1] for result in beam_search_results]
        return results

    def init_decoder(self, batch_size, vocab_list, decoding_method,
                     lang_model_path, beam_alpha, beam_beta, beam_size,
                     cutoff_prob, cutoff_top_n, num_processes):
        """
        init ctc decoders
        Args:
            batch_size(int): Batch size for input data
            vocab_list (list): List of tokens in the vocabulary, for decoding
            decoding_method (str): ctc_beam_search
            lang_model_path (str): language model path
            beam_alpha (float): beam_alpha
            beam_beta (float): beam_beta
            beam_size (int): beam_size
            cutoff_prob (float): cutoff probability in beam search
            cutoff_top_n (int): cutoff_top_n
            num_processes (int): num_processes

        Raises:
            ValueError: when decoding_method not support.

        Returns:
            CTCBeamSearchDecoder
        """
        self.batch_size = batch_size
        self.vocab_list = vocab_list
        self.decoding_method = decoding_method
        self.beam_size = beam_size
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.num_processes = num_processes
        if decoding_method == "ctc_beam_search":
            self._init_ext_scorer(beam_alpha, beam_beta, lang_model_path,
                                  vocab_list)
            if self.beam_search_decoder is None:
                self.beam_search_decoder = self.get_decoder(
                    vocab_list, batch_size, beam_alpha, beam_beta, beam_size,
                    num_processes, cutoff_prob, cutoff_top_n)
            return self.beam_search_decoder
        elif decoding_method == "ctc_greedy":
            self._init_ext_scorer(beam_alpha, beam_beta, lang_model_path,
                                  vocab_list)
        else:
            raise ValueError(f"Not support: {decoding_method}")

    def decode_probs_offline(self, probs, logits_lens, vocab_list,
                             decoding_method, lang_model_path, beam_alpha,
                             beam_beta, beam_size, cutoff_prob, cutoff_top_n,
                             num_processes):
        """
        This function will be deprecated in future.
        ctc decoding with probs.
        Args:
            probs (Tensor): activation after softmax
            logits_lens (Tensor): audio output lens
            vocab_list (list): List of tokens in the vocabulary, for decoding
            decoding_method (str): ctc_beam_search
            lang_model_path (str): language model path
            beam_alpha (float): beam_alpha
            beam_beta (float): beam_beta
            beam_size (int): beam_size
            cutoff_prob (float): cutoff probability in beam search
            cutoff_top_n (int): cutoff_top_n
            num_processes (int): num_processes

        Raises:
            ValueError: when decoding_method not support.

        Returns:
            List[str]: transcripts.
        """
        logger.warn(
            "This function will be deprecated in future: decode_probs_offline")
        probs_split = [probs[i, :l, :] for i, l in enumerate(logits_lens)]
        if decoding_method == "ctc_greedy":
            result_transcripts = self._decode_batch_greedy_offline(
                probs_split=probs_split, vocab_list=vocab_list)
        elif decoding_method == "ctc_beam_search":
            result_transcripts = self._decode_batch_beam_search_offline(
                probs_split=probs_split,
                beam_alpha=beam_alpha,
                beam_beta=beam_beta,
                beam_size=beam_size,
                cutoff_prob=cutoff_prob,
                cutoff_top_n=cutoff_top_n,
                vocab_list=vocab_list,
                num_processes=num_processes)
        else:
            raise ValueError(f"Not support: {decoding_method}")
        return result_transcripts

    def get_decoder(self, vocab_list, batch_size, beam_alpha, beam_beta,
                    beam_size, num_processes, cutoff_prob, cutoff_top_n):
        """
        init get ctc decoder
        Args:
            vocab_list (list): List of tokens in the vocabulary, for decoding.
            batch_size(int): Batch size for input data
            beam_alpha (float): beam_alpha
            beam_beta (float): beam_beta
            beam_size (int): beam_size
            num_processes (int): num_processes
            cutoff_prob (float): cutoff probability in beam search
            cutoff_top_n (int): cutoff_top_n

        Raises:
            ValueError: when decoding_method not support.

        Returns:
            CTCBeamSearchDecoder
        """
        num_processes = min(num_processes, batch_size)
        if self._ext_scorer is not None:
            self._ext_scorer.reset_params(beam_alpha, beam_beta)
        if self.decoding_method == "ctc_beam_search":
            beam_search_decoder = CTCBeamSearchDecoder(
                vocab_list, batch_size, beam_size, num_processes, cutoff_prob,
                cutoff_top_n, self._ext_scorer, self.blank_id)
        else:
            raise ValueError(f"Not support: {self.decoding_method}")
        return beam_search_decoder

    def next(self, probs, logits_lens):
        """
        Input probs into ctc decoder
        Args:
            probs (list(list(float))): probs for a batch of data
            logits_lens (list(int)): logits lens for a batch of data
        Raises:
            Exception: when the ctc decoder is not initialized
            ValueError: when decoding_method not support.
        """

        if self.beam_search_decoder is None:
            raise Exception(
                "You need to initialize the beam_search_decoder firstly")
        beam_search_decoder = self.beam_search_decoder

        has_value = (logits_lens > 0).tolist()
        has_value = [
            "true" if has_value[i] is True else "false"
            for i in range(len(has_value))
        ]
        probs_split = [
            probs[i, :l, :].tolist() if has_value[i] else probs[i].tolist()
            for i, l in enumerate(logits_lens)
        ]
        if self.decoding_method == "ctc_beam_search":
            beam_search_decoder.next(probs_split, has_value)
        else:
            raise ValueError(f"Not support: {self.decoding_method}")

        return

    def decode(self):
        """
        Get the decoding result
        Raises:
            Exception: when the ctc decoder is not initialized
            ValueError: when decoding_method not support.
        Returns:
            results_best (list(str)): The best result for a batch of data
            results_beam (list(list(str))): The beam search result for a batch of data
        """
        if self.beam_search_decoder is None:
            raise Exception(
                "You need to initialize the beam_search_decoder firstly")

        beam_search_decoder = self.beam_search_decoder
        if self.decoding_method == "ctc_beam_search":
            batch_beam_results = beam_search_decoder.decode()
            batch_beam_results = [[(res[0], res[1]) for res in beam_results]
                                  for beam_results in batch_beam_results]
            results_best = [result[0][1] for result in batch_beam_results]
            results_beam = [[trans[1] for trans in result]
                            for result in batch_beam_results]

        else:
            raise ValueError(f"Not support: {self.decoding_method}")

        return results_best, results_beam

    def reset_decoder(self,
                      batch_size=-1,
                      beam_size=-1,
                      num_processes=-1,
                      cutoff_prob=-1.0,
                      cutoff_top_n=-1):
        if batch_size > 0:
            self.batch_size = batch_size
        if beam_size > 0:
            self.beam_size = beam_size
        if num_processes > 0:
            self.num_processes = num_processes
        if cutoff_prob > 0:
            self.cutoff_prob = cutoff_prob
        if cutoff_top_n > 0:
            self.cutoff_top_n = cutoff_top_n
        """
        Reset the decoder state
        Args:
            batch_size(int): Batch size for input data
            beam_size (int): beam_size
            num_processes (int): num_processes
            cutoff_prob (float): cutoff probability in beam search
            cutoff_top_n (int): cutoff_top_n
        Raises:
            Exception: when the ctc decoder is not initialized
        """
        if self.beam_search_decoder is None:
            raise Exception(
                "You need to initialize the beam_search_decoder firstly")
        self.beam_search_decoder.reset_state(
            self.batch_size, self.beam_size, self.num_processes,
            self.cutoff_prob, self.cutoff_top_n)

    def del_decoder(self):
        """
        Delete the decoder
        """
        if self.beam_search_decoder is not None:
            del self.beam_search_decoder
            self.beam_search_decoder = None
