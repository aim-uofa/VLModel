#!/usr/bin/env
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""
Convert nemo style (fused) lora checkpoint to canonical (unfused) lora checkpoint.
Currently supports TP=PP=1 only.

Example usage:
python scripts/checkpoint_converters/lora_converters/convert_nemo_to_canonical.py \
    --nemo_lora_path nemo_style_lora_model.nemo \
    --output_path ./canonical_style_lora_model.nemo 

Example usage to also convert into huggingface format (the script expects a adapter_config.json file which is standard in HF):
python scripts/checkpoint_converters/lora_converters/convert_nemo_to_canonical.py \
    --nemo_lora_path nemo_style_lora_model.nemo \
    --output_path ./canonical_style_lora_model.nemo \
    --hf_format --hf_config checkpoints/bin/adapter_config.json
"""
import json
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import OmegaConf, open_dict
from scripts.nlp_language_modeling.merge_lora_weights.merge import replace_number_add_offset

from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector

target_map = {
    "all": ["gate_proj", "o_proj", "up_proj", "down_proj", "k_proj", "q_proj", "v_proj"],
    "attention_qkv": ["k_proj", "q_proj", "v_proj"],
    "attention_dense": ["gate_proj", "o_proj", "up_proj"],
}


def rename_keys(key):
    new_keys = []
    if "lora_kqv_adapter" in key:
        new_keys.append(key.replace(".lora_kqv_adapter.", ".lora_unfused_kqv_adapter.q_adapter."))
        new_keys.append(key.replace(".lora_kqv_adapter.", ".lora_unfused_kqv_adapter.k_adapter."))
        new_keys.append(key.replace(".lora_kqv_adapter.", ".lora_unfused_kqv_adapter.v_adapter."))
    elif "lora_hto4h_adapter" in key:
        new_keys.append(key.replace(".lora_hto4h_adapter.", ".lora_unfused_hto4h_adapter.gate_adapter."))
        new_keys.append(key.replace(".lora_hto4h_adapter.", ".lora_unfused_hto4h_adapter.up_adapter."))
    return new_keys


def rename_qkv_keys(key):
    new_keys = []
    new_keys.append(key.replace(".lora_kqv_adapter.", ".lora_unfused_kqv_adapter.q_adapter."))
    new_keys.append(key.replace(".lora_kqv_adapter.", ".lora_unfused_kqv_adapter.k_adapter."))
    new_keys.append(key.replace(".lora_kqv_adapter.", ".lora_unfused_kqv_adapter.v_adapter."))
    return new_keys


def reformat_module_names_to_hf(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_tensors = dict()
    for module_name, module_weight in tensors.items():
        # map linear_in and linear_out to lora_a/lora_b counterparts
        new_module_name = "base_model." + module_name.replace("linear_in", "lora_A").replace("linear_out", "lora_B")

        # map target modules to their vLLM/HF counterparts
        new_module_name = new_module_name.replace("q_adapter", "q_proj")
        new_module_name = new_module_name.replace("k_adapter", "k_proj")
        new_module_name = new_module_name.replace("v_adapter", "v_proj")
        new_module_name = new_module_name.replace("lora_dense_attention_adapter", "o_proj")
        new_module_name = new_module_name.replace("lora_4htoh_adapter", "down_proj")
        new_module_name = new_module_name.replace("gate_adapter", "gate_proj")
        new_module_name = new_module_name.replace("up_adapter", "up_proj")

        # map other parts of the module names to fit vLLM/huggingface
        new_module_name = new_module_name.replace(".adapter_layer", "")
        new_module_name = new_module_name.replace(".lora_unfused_kqv_proj", "")
        new_module_name = new_module_name.replace(".lora_unfused_hto4h_adapter", "")
        new_module_name = new_module_name.replace("self_attention", "self_attn")
        new_module_name = new_module_name.replace("decoder", "model")

        new_tensors[new_module_name] = module_weight
    return new_tensors


def convert_lora_weights_to_canonical(
    config: Dict[str, Any], lora_weights: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """This function converts nemo style (fused) lora weights to canonical (unfused)
    LoRA weights. Namely, it unfuses the QKV adapter layers and the H-to-4H adapter layers.

    Returns:
        Dict[str, torch.Tensor]: The new LoRA weights with unfused layers.
    """

    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    head_size = hidden_size // num_heads
    num_query_groups = int(config.get("num_query_groups", num_heads))  # num_kv_heads

    heads_per_group = num_heads // num_query_groups
    qkv_total_dim = num_heads + 2 * num_query_groups

    adapter_size = config['peft']['lora_tuning']['adapter_dim']

    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * group_idx, (heads_per_group + 2) * group_idx + heads_per_group)
            for group_idx in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, heads_per_group + 2)
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, heads_per_group + 2)

    qkv_keys_to_update = []
    hto4h_keys_to_update = []
    for key in lora_weights.keys():
        if "lora_kqv_adapter" in key:
            qkv_keys_to_update.append(key)
        if "lora_hto4h_adapter" in key:
            hto4h_keys_to_update.append(key)

    # unfuse QKV layer
    for key in qkv_keys_to_update:
        if "linear_in" in key:
            assert lora_weights[key].size(0) == adapter_size
            for new_key in rename_qkv_keys(key):
                lora_weights[new_key] = lora_weights[key]
                assert len(lora_weights[new_key].size()) == 2
        elif "linear_out" in key:
            assert lora_weights[key].size(1) == adapter_size
            for new_key, size in zip(rename_qkv_keys(key), [q_slice, k_slice, v_slice]):
                lora_weights[new_key] = (
                    lora_weights[key]
                    .reshape((qkv_total_dim, head_size, adapter_size))[size]
                    .reshape((-1, adapter_size))
                )
                assert len(lora_weights[new_key].size()) == 2
        lora_weights.pop(key)

    # This maps to gate_up_proj in HF, but we need to split it up into gate_proj and up_proj
    for key in hto4h_keys_to_update:
        gate_proj_key = key.replace(".lora_hto4h_adapter.", ".lora_unfused_hto4h_adapter.gate_adapter.")
        up_proj_key = key.replace(".lora_hto4h_adapter.", ".lora_unfused_hto4h_adapter.up_adapter.")

        module_weight = lora_weights[key]
        if "linear_in" in key:
            # lora_a gets duplicated
            lora_weights[gate_proj_key] = module_weight
            lora_weights[up_proj_key] = module_weight
        elif "linear_out" in key:
            # lora_b gets split
            split_size = module_weight.shape[0]
            gate_up_split = module_weight.split(split_size // 2)
            lora_weights[gate_proj_key] = gate_up_split[0]
            lora_weights[up_proj_key] = gate_up_split[1]
        lora_weights.pop(key)
    return lora_weights


def convert_lora(lora_nemo, save_path, hf_format=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        NLPSaveRestoreConnector._unpack_nemo_file(lora_nemo, tmpdir)
        config_file = f"{tmpdir}/model_config.yaml"
        lora_config = OmegaConf.load(config_file)
        tp_size = lora_config.tensor_model_parallel_size
        pp_size = lora_config.pipeline_model_parallel_size

        lora_state_dict = [{}] * tp_size

        for pp in range(pp_size):
            for tp in range(tp_size):
                if tp_size == 1:
                    ckpt_file = f"{tmpdir}/model_weights.ckpt"
                elif pp_size == 1:
                    ckpt_file = f"{tmpdir}/mp_rank_{tp:02d}/model_weights.ckpt"
                else:
                    ckpt_file = f"{tmpdir}/tp_rank_{tp:02d}_pp_rank_{pp:03d}/model_weights.ckpt"

                l = torch.load(ckpt_file, map_location=torch.device('cpu'))
                if pp == 0:
                    lora_state_dict[tp] = l
                else:
                    # calculate layer offset
                    layer_offset = lora_config.num_layers // pp_size * pp
                    for key, value in l.items():
                        new_key = replace_number_add_offset(key, layer_offset)
                        lora_state_dict[tp][new_key] = value

        # TODO: currently suport tp=1
        lora_state_dict = lora_state_dict[0]
        if lora_config.peft.lora_tuning.variant == "nemo":
            with open_dict(lora_config):
                lora_config.peft.lora_tuning.variant = "canonical"
            with open(f"{tmpdir}/model_config.yaml", "w") as f:
                OmegaConf.save(lora_config, f)
            lora_state_dict = convert_lora_weights_to_canonical(lora_config, lora_state_dict)
        if hf_format:
            lora_state_dict = reformat_module_names_to_hf(lora_state_dict)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            torch.save(lora_state_dict, f"{save_path}/adapter_model.bin")
            adapter_config = json.load(open(args.hf_config))
            adapter_config['peft_type'] = "LORA"
            adapter_config['r'] = lora_config.peft.lora_tuning.adapter_dim
            adapter_config['lora_alpha'] = lora_config.peft.lora_tuning.alpha
            with open(f"{save_path}/adapter_config.json", "w") as f:
                json.dump(adapter_config, f, indent=4)
        else:
            torch.save(lora_state_dict, f"{tmpdir}/model_weights.ckpt")
            NLPSaveRestoreConnector._make_nemo_file_from_folder(save_path, tmpdir)

    return lora_state_dict, lora_config


def fix_for_O2(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if "model.module." not in k:
            new_state_dict[k.replace('model.', 'model.module.')] = v
    return new_state_dict


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--nemo_lora_path",
        type=str,
        default=None,
        required=True,
        help="Path to NeMo style (fused) lora checkpoint in .nemo file format",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the canonical (unfused) lora .nemo file.",
    )
    parser.add_argument("--hf_format", action='store_true', help="saves tensors in huggingface naming format.")
    parser.add_argument("--hf_config", type=str, help="the adapter config in HF-PEFT format.")
    parser.add_argument("--precision", type=str, default="16", help="Model precision")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    convert_lora(args.nemo_lora_path, args.output_path, args.hf_format)
