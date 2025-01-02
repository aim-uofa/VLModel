# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import argparse
import importlib
import inspect
import os
import traceback

import torch
import wrapt

from nemo.core import Model
from nemo.utils import model_utils

DOMAINS = ['asr', 'tts', 'nlp']


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain', choices=DOMAINS, type=str)

    args = parser.parse_args()
    return args


###############################


def _build_import_path(domain, subdomains: list, imp):
    import_path = ["nemo", "collections", domain]
    import_path.extend(subdomains)
    import_path.append(imp)

    path = ".".join(import_path)
    return path


def _get_class_from_path(domain, subdomains, imp):
    path = _build_import_path(domain, subdomains, imp)

    class_ = None
    result = None

    try:
        class_ = model_utils.import_class_by_path(path)

        if inspect.isclass(class_):
            # Is class wrpped in a wrapt.decorator a the class level? Unwrap for checks.
            if isinstance(class_, wrapt.FunctionWrapper):
                class_ = class_.__wrapped__

            # Subclass tests
            if issubclass(class_, (Model, torch.nn.Module)):
                result = class_
        else:
            class_ = None

        error = None

    except Exception:
        error = traceback.format_exc()

    return class_, result, error


def _test_domain_module_imports(module, domain, subdomains: list):
    module_list = []
    failed_list = []
    error_list = []

    error = None
    if len(subdomains) > 0:
        basepath = module.__path__[0]
        nemo_index = basepath.rfind("nemo")
        basepath = basepath[nemo_index:].replace(os.path.sep, ".")
        new_path = '.'.join([basepath, *subdomains])

        try:
            module = importlib.import_module(new_path)
        except Exception:
            print(f"Could not import `{new_path}` ; Traceback below :")
            error = traceback.format_exc()
            error_list.append(error)

    if error is None:
        for imp in dir(module):
            class_, result, error = _get_class_from_path(domain, subdomains, imp)

            if result is not None:
                module_list.append(class_)

            elif class_ is not None:
                failed_list.append(class_)

            if error is not None:
                error_list.append(error)

    for module in module_list:
        print("Module successfully imported :", module)

    print()
    for module in failed_list:
        print("Module did not match a valid signature of NeMo Model (hence ignored):", module)

    print()
    if len(error_list) > 0:
        print("Imports crashed with following traceback !")

        for error in error_list:
            print("*" * 100)
            print()
            print(error)
            print()
            print("*" * 100)
            print()

    if len(error_list) > 0:
        return False
    else:
        return True


###############################


def test_domain_asr(args):
    import nemo.collections.asr as nemo_asr

    all_passed = _test_domain_module_imports(nemo_asr, domain=args.domain, subdomains=['models'])

    if not all_passed:
        exit(1)


def test_domain_nlp(args):
    # If even this fails, just fail entirely.
    import nemo.collections.nlp as nemo_nlp

    # Basic NLP test
    all_passed = _test_domain_module_imports(nemo_nlp, domain=args.domain, subdomains=['models'])

    # Megatron Test
    all_passed = (
        _test_domain_module_imports(
            nemo_nlp, domain=args.domain, subdomains=['models', 'language_modeling', 'megatron_base_model']
        )
        and all_passed
    )
    all_passed = (
        _test_domain_module_imports(
            nemo_nlp, domain=args.domain, subdomains=['models', 'language_modeling', 'megatron_bert_model']
        )
        and all_passed
    )
    all_passed = (
        _test_domain_module_imports(
            nemo_nlp, domain=args.domain, subdomains=['models', 'language_modeling', 'megatron_glue_model']
        )
        and all_passed
    )
    all_passed = (
        _test_domain_module_imports(
            nemo_nlp, domain=args.domain, subdomains=['models', 'language_modeling', 'megatron_gpt_model']
        )
        and all_passed
    )
    all_passed = (
        _test_domain_module_imports(
            nemo_nlp,
            domain=args.domain,
            subdomains=['models', 'language_modeling', 'megatron_lm_encoder_decoder_model'],
        )
        and all_passed
    )
    all_passed = (
        _test_domain_module_imports(
            nemo_nlp,
            domain=args.domain,
            subdomains=['models', 'language_modeling', 'megatron_gpt_prompt_learning_model'],
        )
        and all_passed
    )
    all_passed = (
        _test_domain_module_imports(
            nemo_nlp, domain=args.domain, subdomains=['models', 'language_modeling', 'megatron_t5_model']
        )
        and all_passed
    )
    all_passed = (
        _test_domain_module_imports(
            nemo_nlp, domain=args.domain, subdomains=['models', 'language_modeling', 'megatron_t5_model']
        )
        and all_passed
    )

    if not all_passed:
        exit(1)


def test_domain_tts(args):
    import nemo.collections.tts as nemo_tts

    all_passed = _test_domain_module_imports(nemo_tts, domain=args.domain, subdomains=['models'])

    if not all_passed:
        exit(1)


###############################


def test_domain(args):
    domain = args.domain

    if domain == 'asr':
        test_domain_asr(args)
    elif domain == 'nlp':
        test_domain_nlp(args)
    elif domain == 'tts':
        test_domain_tts(args)
    else:
        raise RuntimeError(f"Cannot resolve domain : {domain}")


def run_checks():
    args = process_args()
    test_domain(args)


if __name__ == '__main__':
    run_checks()
