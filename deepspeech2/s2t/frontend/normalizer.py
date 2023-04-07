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
"""Contains feature normalizers."""
import numpy as np

from .utility import load_cmvn

__all__ = ["FeatureNormalizer"]


class FeatureNormalizer(object):
    def __init__(self, mean_std_filepath):
        mean_std = mean_std_filepath
        self._read_mean_std_from_file(mean_std)

    def apply(self, features):
        """Normalize features to be of zero mean and unit stddev.

        :param features: Input features to be normalized.
        :type features: ndarray, shape (T, D)
        :param eps:  added to stddev to provide numerical stablibity.
        :type eps: float
        :return: Normalized features.
        :rtype: ndarray
        """
        return (features - self._mean) * self._istd

    def _read_mean_std_from_file(self, mean_std, eps=1e-20):
        """Load mean and std from file."""
        if isinstance(mean_std, list):
            mean = mean_std[0]['cmvn_stats']['mean']
            istd = mean_std[0]['cmvn_stats']['istd']
        else:
            filetype = mean_std.split(".")[-1]
            mean, istd = load_cmvn(mean_std, filetype=filetype)
        self._mean = np.expand_dims(mean, axis=0)
        self._istd = np.expand_dims(istd, axis=0)
