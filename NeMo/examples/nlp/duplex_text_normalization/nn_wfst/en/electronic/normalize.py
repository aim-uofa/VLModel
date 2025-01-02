# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    from nemo_text_processing.text_normalization.normalize import Normalizer
    from nemo_text_processing.text_normalization.token_parser import TokenParser
except (ImportError, ModuleNotFoundError):
    raise ModuleNotFoundError(
        "The package `nemo_text_processing` was not installed in this environment. Please refer to"
        " https://github.com/NVIDIA/NeMo-text-processing and install this package before using "
        "this script"
    )

from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor


class ElectronicNormalizer(Normalizer):
    """
    Normalizer for ELECTRONIC.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        lang: language
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(
        self,
        input_case: str = 'cased',
        lang: str = 'en',
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        max_number_of_permutations_per_split: int = 729,
    ):

        from nn_wfst.en.electronic.tokenize_and_classify import ClassifyFst
        from nn_wfst.en.electronic.verbalize_final import VerbalizeFinalFst

        self.tagger = ClassifyFst(
            input_case=input_case, deterministic=deterministic, cache_dir=cache_dir, overwrite_cache=overwrite_cache
        )
        self.verbalizer = VerbalizeFinalFst(deterministic=deterministic)
        self.post_processor = None
        self.parser = TokenParser()
        self.lang = lang
        self.processor = MosesProcessor(lang_id=lang)
        self.max_number_of_permutations_per_split = max_number_of_permutations_per_split
