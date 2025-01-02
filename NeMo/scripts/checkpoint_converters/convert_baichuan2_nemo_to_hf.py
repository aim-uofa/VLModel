# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from pytorch_lightning import Trainer
from transformers import AutoModelForCausalLM

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging

"""
Script to convert a baichuan2 checkpoint in nemo (mcore path) into a HuggingFace checkpoint.
This script can be used to 1) generate only the HF weights, or 2) generate an entire HF model folder.

1) Generate only HF weights from a nemo file:

    python convert_baichuan2_nemo_to_hf.py \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --output_path /path/to/pytorch_model.bin
    
2) Generate the full HF model folder

    python convert_baichuan2_nemo_to_hf.py \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --output_path /path/to/pytorch_model.bin \
    --hf_input_path /path/to/input_hf_folder \
    --hf_output_path /path/to/output_hf_folder

    Use the --cpu-only flag if the model cannot fit in the GPU. 
    However this option makes the conversion script significantly slower.
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path", type=str, default=None, required=True, help="Path to .nemo file",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to HF .bin file")
    parser.add_argument(
        "--hf_input_path",
        type=str,
        default=None,
        help="A HF model path, "
        "e.g. a folder containing https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/tree/main",
    )
    parser.add_argument(
        "--hf_output_path",
        type=str,
        default=None,
        help="Output HF model path, " "with the same format as above but user's own weights",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision of output weights."
        "Defaults to precision of the input nemo weights (model.cfg.trainer.precision)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Load model in cpu only. Useful if the model cannot fit in GPU memory, "
        "but this option makes the conversion script significantly slower.",
    )
    args = parser.parse_args()
    return args


def convert(input_nemo_file, output_hf_file, precision=None, cpu_only=False) -> None:
    """
    Convert NeMo weights to HF weights
    """

    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    if cpu_only:
        map_location = torch.device('cpu')
        model_config = MegatronGPTModel.restore_from(input_nemo_file, trainer=dummy_trainer, return_config=True)
        model_config.use_cpu_initialization = True
        model_config.tensor_model_parallel_size = 1
    else:
        map_location, model_config = None, None

    if cpu_only:
        logging.info("******** Loading model on CPU. This will take a significant amount of time.")
    model = MegatronGPTModel.restore_from(
        input_nemo_file, trainer=dummy_trainer, override_config_path=model_config, map_location=map_location
    )
    if precision is None:
        precision = model.cfg.precision
    if precision in [32, "32"]:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        logging.warning(f"Precision string {precision} is not recognized, falling back to fp32")
        dtype = torch.float32  # fallback

    param_to_weights = lambda param: param.to(dtype)
    checkpoint = OrderedDict()

    hidden_size = model.cfg.hidden_size
    head_num = model.cfg.num_attention_heads
    num_layers = model.cfg.num_layers
    ffn_hidden_size = model.cfg.ffn_hidden_size
    num_query_groups = model.cfg.get("num_query_groups", head_num)  # different num_query_groups for 70B

    head_size = hidden_size // head_num
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    # Embedding
    embed_weight = model.state_dict()[f'model.embedding.word_embeddings.weight']
    embed_weights_base_name = f'model.embed_tokens.weight'
    checkpoint[embed_weights_base_name] = param_to_weights(embed_weight)

    for l in range(int(num_layers)):
        print(f"converting layer {l}")

        qkv_weights = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.weight']
        qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

        qkv_weights_base_name = f'model.layers.{l}.self_attn.W_pack.weight'
        checkpoint[qkv_weights_base_name] = param_to_weights(
            torch.cat(
                [
                    qkv_weights[q_slice].reshape(-1, hidden_size),
                    qkv_weights[k_slice].reshape(-1, hidden_size),
                    qkv_weights[v_slice].reshape(-1, hidden_size),
                ]
            )
        )

        # attention dense
        o_weight = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_proj.weight']
        o_weight_base_name = f'model.layers.{l}.self_attn.o_proj.weight'
        checkpoint[o_weight_base_name] = param_to_weights(o_weight)

        # mlp
        mlp_weights = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1.weight']
        mlp_down_proj_weight = mlp_weights[:ffn_hidden_size, :]
        mlp_gate_proj_weight = mlp_weights[ffn_hidden_size:, :]

        mlp_down_proj_base_name = f'model.layers.{l}.mlp.gate_proj.weight'
        mlp_gate_proj_base_name = f'model.layers.{l}.mlp.up_proj.weight'

        checkpoint[mlp_down_proj_base_name] = param_to_weights(mlp_down_proj_weight)
        checkpoint[mlp_gate_proj_base_name] = param_to_weights(mlp_gate_proj_weight)

        mlp_up_proj_weight = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc2.weight']
        mlp_up_proj_base_name = f'model.layers.{l}.mlp.down_proj.weight'
        checkpoint[mlp_up_proj_base_name] = param_to_weights(mlp_up_proj_weight)

        # layernorm
        input_ln_weight = model.state_dict()[f'model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight']
        input_ln_base_name = f'model.layers.{l}.input_layernorm.weight'
        checkpoint[input_ln_base_name] = param_to_weights(input_ln_weight)

        post_attn_ln_weight = model.state_dict()[f'model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight']
        post_attn_ln_base_name = f'model.layers.{l}.post_attention_layernorm.weight'
        checkpoint[post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

        print(f"done layer {l}")

    final_ln_weight = model.state_dict()[f'model.decoder.final_layernorm.weight']
    final_ln_base_name = f'model.norm.weight'
    checkpoint[final_ln_base_name] = param_to_weights(final_ln_weight)

    output_layer_weight = model.state_dict()[f'model.output_layer.weight']
    output_layer_base_name = f'lm_head.weight'
    checkpoint[output_layer_base_name] = param_to_weights(output_layer_weight)

    os.makedirs(os.path.dirname(output_hf_file), exist_ok=True)
    torch.save(checkpoint, output_hf_file)
    logging.info(f"Weights saved to {output_hf_file}")


def replace_hf_weights(weights_file, input_hf_path, output_hf_path):
    model = AutoModelForCausalLM.from_pretrained(input_hf_path, local_files_only=True, trust_remote_code=True)
    nemo_exported = torch.load(weights_file)

    model.load_state_dict(nemo_exported)
    model.save_pretrained(output_hf_path)
    logging.info(f"Full HF model saved to {output_hf_path}")


if __name__ == '__main__':
    args = get_args()
    convert(args.input_name_or_path, args.output_path, precision=args.precision, cpu_only=args.cpu_only)
    if args.hf_input_path and args.hf_output_path:
        replace_hf_weights(args.output_path, args.hf_input_path, args.hf_output_path)
    else:
        logging.info("`hf_input_path` and/or `hf_output_path` not provided, not generating full HF model.")
        logging.info(f".bin file is saved to {args.output_path}")
