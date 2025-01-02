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


import csv
import logging
from typing import Dict, List, Tuple

import numpy as np
import tensorrt_llm
from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.functional import non_gated_version
from tensorrt_llm.layers import MoeConfig
from tensorrt_llm.models.modeling_utils import PretrainedConfig

from nemo.export.trt_llm.converter.model_to_trt_llm_ckpt import convert_model_to_trt_llm_ckpt
from nemo.export.trt_llm.converter.utils import DECODER_MODEL_TYPE, split

LOGGER = logging.getLogger("NeMo")


def prompt_convert(prompt_config, prompt_weights):
    if "task_templates" in prompt_config:
        prompt_templates = prompt_config["task_templates"]
        actual_task_id = 0
        vtokens_embeddings = []
        vtokens_len = []
        for task_name_id, prompt_task in enumerate(prompt_templates):
            prompt_task_name = prompt_task["taskname"]
            LOGGER.info(f"Task {actual_task_id}: {prompt_task['taskname']}")
            prompt_task_weights = prompt_weights["prompt_table"].get(
                f"prompt_table.{prompt_task_name}.prompt_embeddings.weight"
            )
            if prompt_task_weights is None:
                continue
            vtokens_embeddings.append(prompt_task_weights)
            vtokens_len.append(prompt_task_weights.shape[0])
            actual_task_id += 1

        max_vtoken_len = max(vtokens_len)
        embedding_dim = vtokens_embeddings[0].shape[1]

        # pad tasks to longest task embedding table
        for i, vtoken_emb_table in enumerate(vtokens_embeddings):
            padded_table = torch.zeros((max_vtoken_len, embedding_dim))
            padded_table[: vtoken_emb_table.shape[0], :] = vtoken_emb_table
            vtokens_embeddings[i] = padded_table

        vtokens_embeddings = torch.stack(vtokens_embeddings)
    else:
        vtokens_embeddings = prompt_weights["prompt_embeddings_weights"]

    return vtokens_embeddings


def model_to_trtllm_ckpt(
    model,
    nemo_model_config,
    nemo_export_dir,
    decoder_type: str,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    use_parallel_embedding: bool = False,
) -> Tuple[List[Dict], List[PretrainedConfig]]:

    weights_dict = convert_model_to_trt_llm_ckpt(
        model=model,
        nemo_model_config=nemo_model_config,
        nemo_export_dir=nemo_export_dir,
        inference_tp_size=tensor_parallel_size,
        processes=1,
        storage_type=dtype,
        use_parallel_embedding=use_parallel_embedding,
        decoder_type=decoder_type,
    )

    world_size = tensor_parallel_size * pipeline_parallel_size

    lm_head_weight = weights_dict["lm_head.weight"]

    vocab_size = weights_dict["transformer.vocab_embedding.weight"].shape[0]
    vocab_size_padded = pad_vocab_size(vocab_size, tensor_parallel_size)

    if vocab_size_padded != vocab_size:
        pad_width = vocab_size_padded - vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)), "constant", constant_values=0)

    hidden_act = nemo_model_config.get('activation')
    hidden_act = (
        hidden_act.split("-")[-1] if nemo_model_config.get('num_moe_experts', 0) else non_gated_version(hidden_act)
    )

    config = {
        'architecture': DECODER_MODEL_TYPE[decoder_type],
        'dtype': dtype,
        'num_hidden_layers': nemo_model_config.get('num_layers'),
        'num_attention_heads': nemo_model_config.get('num_attention_heads'),
        'num_key_value_heads': nemo_model_config.get('num_query_groups', nemo_model_config['num_attention_heads']),
        'head_size': nemo_model_config.get('kv_channels'),
        'hidden_size': nemo_model_config.get('hidden_size'),
        'intermediate_size': nemo_model_config.get('ffn_hidden_size'),
        'norm_epsilon': nemo_model_config.get('layernorm_epsilon'),
        'vocab_size': vocab_size_padded,
        'position_embedding_type': (
            "rope_gpt_neox" if nemo_model_config.get('position_embedding_type') == "rope" else "learned_absolute"
        ),
        'max_position_embeddings': nemo_model_config.get('max_position_embeddings'),
        'hidden_act': hidden_act,
        'use_parallel_embedding': use_parallel_embedding,
        'embedding_sharding_dim': 0,
        'share_embedding_table': False,
        'quantization': {
            'quant_algo': None,
            'kv_cache_quant_algo': None,
        },
        'bias': nemo_model_config.get('bias'),
        'apply_query_key_layer_scaling': False,
        'rotary_pct': nemo_model_config.get('rotary_percentage', 1.0),
        'rotary_base': nemo_model_config.get('rotary_base', 10000),
        'moe_num_experts': nemo_model_config.get('num_moe_experts', 0),
        'moe_top_k': nemo_model_config.get('moe_router_topk'),
        'moe_normalization_mode': nemo_model_config.get(
            'moe_renorm_mode', MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
        ),
        'moe_tp_mode': nemo_model_config.get('moe_tp_mode', MoeConfig.ParallelismMode.TENSOR_PARALLEL),
        'logits_dtype': 'float32',
        'world_size': world_size,
        'tp_size': tensor_parallel_size,
        'pp_size': pipeline_parallel_size,
    }

    model_configs = []
    weights_dicts = []
    num_layers = nemo_model_config.get('num_layers')
    rotary_scaling = nemo_model_config.get("seq_len_interpolation_factor")

    if decoder_type == "falcon":
        config["new_decoder_architecture"] = False if num_layers == 32 else True
        config["parallel_attention"] = True
    if rotary_scaling is not None:
        config["rotary_scaling"] = {"type": "linear", "factor": float(rotary_scaling)}

    pp_key = {
        "transformer.vocab_embedding.weight",
        "transformer.position_embedding.weight",
        "lm_head.weight",
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
    }

    for i in range(world_size):
        mapping = tensorrt_llm.Mapping(
            world_size=world_size, rank=i, tp_size=tensor_parallel_size, pp_size=pipeline_parallel_size
        )
        layers_range = mapping.pp_layers(num_layers)

        weights_dict_local = {}
        for k, v in weights_dict.items():
            if k in pp_key:
                continue
            new_key = k
            if new_key.endswith(".bin"):  # TP split
                if new_key.endswith(f"{mapping.tp_rank}.bin"):
                    new_key = new_key.replace(f".{mapping.tp_rank}.bin", "")
            if "layers" in new_key:  # PP
                layer_num = int(new_key.split(".")[2])
                if layer_num in layers_range:
                    new_key = new_key.replace(f"layers.{layer_num}", f"layers.{layer_num-layers_range[0]}")
            if config.get("new_decoder_architecture", False) and "post_layernorm" in new_key:
                new_key = new_key.replace("post_layernorm", "mlp_layernorm")
            weights_dict_local[new_key] = v

        if mapping.is_first_pp_rank():
            embedding_weight = (
                np.ascontiguousarray(
                    split(weights_dict["transformer.vocab_embedding.weight"], mapping.tp_size, mapping.tp_rank)
                )
                if use_parallel_embedding
                else weights_dict["transformer.vocab_embedding.weight"]
            )

            weights_dict_local["transformer.vocab_embedding.weight"] = embedding_weight

            pos_embedding_weight = weights_dict.get("transformer.position_embedding.weight")
            if pos_embedding_weight is not None:
                if use_parallel_embedding:
                    pos_embedding_weight = np.ascontiguousarray(
                        split(pos_embedding_weight, mapping.tp_size, mapping.tp_rank)
                    )
                weights_dict_local["transformer.position_embedding.weight"] = pos_embedding_weight

        if mapping.is_last_pp_rank():
            weights_dict_local["lm_head.weight"] = np.ascontiguousarray(
                split(lm_head_weight, mapping.tp_size, mapping.tp_rank)
            )
            weights_dict_local["transformer.ln_f.weight"] = weights_dict["transformer.ln_f.weight"]

            ln_f_bias = weights_dict.get("transformer.ln_f.bias")
            if ln_f_bias is not None:
                weights_dict_local["transformer.ln_f.bias"] = ln_f_bias

        model_config = PretrainedConfig(**config)
        model_config.mapping = mapping
        model_configs.append(model_config)
        weights_dicts.append(weights_dict_local)

    return weights_dicts, model_configs
