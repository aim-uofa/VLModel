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
import json
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from omegaconf import open_dict
from pytorch_lightning import Trainer
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import LlavaForConditionalGeneration, LlavaConfig
import sys
sys.path.append('/workspace/NeMo')
from examples.multimodal.multimodal_llm.neva.huggingface_models.our_llava.modeling_llava_next import LlavaNextForConditionalGeneration, LlavaNextConfig
from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging

"""
Script to convert a llama2 checkpoint in nemo (mcore path) into a HuggingFace checkpoint.
This script can be used to 1) generate only the HF weights, or 2) generate an entire HF model folder.

1) Generate only HF weights from a nemo file:

    python convert_llama_nemo_to_hf.py \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --output_path /path/to/pytorch_model.bin
    
2) Generate the full HF model folder

    python convert_llama_nemo_to_hf.py \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --output_path /path/to/pytorch_model.bin \
    --hf_input_path /path/to/input_hf_folder \
    --hf_output_path /path/to/output_hf_folder \
    --input_tokenizer /path/to/tokenizer \
    --hf_output_tokenizer /path/to/output_tokenizer \

    Use the --cpu-only flag if the model cannot fit in the GPU (e.g. Llama2 70b). 
    However this option makes the conversion script significantly slower.
"""

PROCESSOR_CONFIG = {
    "clip": {
        "crop_size": {
            "height": 336,
            "width": 336
        },
        "do_center_crop": True,
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "feature_extractor_type": "CLIPFeatureExtractor",
        "image_mean": [
            0.48145466,
            0.4578275,
            0.40821073
        ],
        "image_processor_type": "CLIPImageProcessor",
        "image_std": [
            0.26862954,
            0.26130258,
            0.27577711
        ],
        "processor_class": "LlavaProcessor",
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {
            "shortest_edge": 336
        }
    },
    "siglip": {
        "crop_size": {
            "height": 384,
            "width": 384
        },
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": False,
        "do_center_crop": False,
        "image_mean": [
            0.5,
            0.5,
            0.5
        ],
        "image_processor_type": "SiglipImageProcessor",
        "image_std": [
            0.5,
            0.5,
            0.5
        ],
        "processor_class": "SiglipProcessor",
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {
            "shortest_edge": 768
        },
        "image_grid_pinpoints":[[768, 768]]
    }
}

vision_tower_config = {
    "siglip": {
        "hidden_size": 1152,
        "image_size": 384,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    },
    "clip": {
        "hidden_size": 1024,
        "image_size": 336,
        "intermediate_size": 4096,
        "model_type": "clip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "patch_size": 14,
    }
}

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path", type=str, default=None, required=True, help="Path to .nemo file or extracted folder",
    )
    parser.add_argument(
        "--processor_type", type=str, default="siglip", help="processor config",
    )
    parser.add_argument(
        "--vision_tower_type", type=str, default="siglip", help="processor config",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to HF .bin file")
    parser.add_argument(
        "--hf_input_path",
        type=str,
        default=None,
        help="A HF model path, " "e.g. a folder containing https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main",
    )
    parser.add_argument(
        "--hf_output_path",
        type=str,
        default=None,
        help="Output HF model path, " "with the same format as above but user's own weights",
    )
    parser.add_argument(
        "--vision_encoder_path",
        type=str,
        default=None,
        help="Output HF model path, " "with the same format as above but user's own weights",
    )
    parser.add_argument(
        "--input_tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer used for the input nemo model. (need to extract the .nemo file first)",
    )
    parser.add_argument(
        "--hf_output_tokenizer",
        type=str,
        default=None,
        help="Path to save the tokenizer used for the output HF model.",
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


def convert(input_nemo_file, input_tokenizer_file, output_hf_file, precision=None, cpu_only=False) -> None:
    """
    Convert NeMo weights to HF weights
    """
    try:
        logging.info("Loading tokenizer from {}".format(input_tokenizer_file))
        tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_file)
    except OSError:
        logging.info("Only huggingface tokenizers are supported for now.")
        exit()
    dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())
    model_config = MegatronNevaModel.restore_from(input_nemo_file, trainer=dummy_trainer, return_config=True)
    """
    Vocab size should be same as finetuning
    """
    vocab_size = len(tokenizer)
    multiple = model_config.make_vocab_size_divisible_by * model_config.tensor_model_parallel_size
    while (vocab_size % multiple) != 0:
        vocab_size += 1
    model_config.override_vocab_size = vocab_size

    model_config.tensor_model_parallel_size = 1
    model_config.pipeline_model_parallel_size = 1

    """
    Clear the pretrained path if module is not frozen. You can load this weights from input_nemo_file
    """
    # model_config.mm_cfg.llm.from_pretrained = None
    model_config.mm_cfg.llm.freeze = False

    if cpu_only:
        map_location = torch.device('cpu')
        model_config.use_cpu_initialization = True
    else:
        map_location = None

    if cpu_only:
        logging.info("******** Loading model on CPU. This will take a significant amount of time.")
    model = MegatronNevaModel.restore_from(
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
    logging.info(f"Using precision {dtype}")

    param_to_weights = lambda param: param.to(dtype)
    checkpoint = OrderedDict()

    hidden_size = model_config.hidden_size
    head_num = model_config.num_attention_heads
    num_layers = model_config.num_layers
    ffn_hidden_size = model_config.ffn_hidden_size
    if model.cfg.get("num_query_groups", head_num):
        num_query_groups = model.cfg.get("num_query_groups", head_num)  # different num_query_groups for 70B
    else:
        num_query_groups = head_num
    head_size = hidden_size // head_num
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    os.makedirs(os.path.dirname(output_hf_file), exist_ok=True)
    # hf config.json
    text_config = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": ffn_hidden_size,
        "max_length": model.cfg.encoder_seq_length,
        "max_position_embeddings": model.cfg.max_position_embeddings,
        "num_key_value_heads": model.cfg.num_query_groups,
        "rms_norm_eps": model.cfg.layernorm_epsilon,
        "rope_theta": getattr(model.cfg, "rotary_base", 10000)
    }
    vision_config = vision_tower_config[args.vision_tower_type]
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    image_token_index = tokenizer.encode("<image>", add_special_tokens=False)[0]
    if args.processor_type == "clip":
        hf_config = LlavaConfig(
        vision_config=vision_config, 
        text_config=text_config,
        vision_feature_select_strategy="full",
        image_token_index=image_token_index,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )
    else:
        hf_config = LlavaNextConfig(
        vision_config=vision_config, 
        text_config=text_config,
        vision_feature_select_strategy="full",
        image_token_index=image_token_index,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )
    hf_config.save_pretrained(output_hf_file)

    # Embedding
    embed_weight = model.state_dict()[f'model.language_model.embedding.word_embeddings.weight']
    embed_weights_base_name = f'language_model.model.embed_tokens.weight'
    checkpoint[embed_weights_base_name] = param_to_weights(embed_weight)

    # mm_projector
    mm_projector_weight = {}
    for name in model.state_dict():
        if "mm_projector" in name:
            if "downsample" in name or "alpha" in name:
                base_name = "multi_modal_projector." + name.split("mm_projector_adapter.")[-1]
            else:
                base_name = "multi_modal_projector.linear_" + name.split("mm_projector_adapter.mm_projector.")[-1]
            # layer index is 0 and 2 in NeMo
            base_name = base_name.replace("0", "1")
            mm_projector_weight[base_name] = model.state_dict()[name]
    checkpoint.update(mm_projector_weight)

    # vision_tower
    # vision_tower_model = AutoModel.from_pretrained(args.vision_encoder_path)
    vision_tower_weight = {}
    if args.vision_encoder_path:
        vision_tower_model = AutoModel.from_pretrained(args.vision_encoder_path).vision_model
        for name in vision_tower_model.state_dict():
            base_name = "vision_tower.vision_model." + name
            vision_tower_weight[base_name] = vision_tower_model.state_dict()[name]
    else:
        for name in model.state_dict():
            if "vision_encoder" in name:
                base_name = "vision_tower.vision_model." + name.split("model.embedding.word_embeddings.vision_encoder.")[1]
                vision_tower_weight[base_name] = model.state_dict()[name]
    checkpoint.update(vision_tower_weight)
    for l in range(int(num_layers)):
        print(f"converting layer {l}")

        qkv_weights = model.state_dict()[f'model.language_model.encoder.layers.{l}.self_attention.query_key_value.weight']
        qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
        ## Example of slices
        ## 7b: num_query_groups = head_num = 32,
        ## q_slice = [0, 3, 6, 9 , ... 90, 93]
        ## k_slice = [1, 4, 7, 10, ... 91, 94]
        ## v_slice = [2, 5, 8, 11, ... 92, 95]
        ## 70b (with GQA): num_query_groups = 8, head_num = 64
        ## q_slice = [0, 1, .. 6, 7, 10, 11, .. 16, 17, 20, 21, .. 67, 70, ... 76, 77]
        ## k_slice = [8, 18, 28, ... 68, 78]
        ## v_slice = [9, 19, 29, ... 69, 79]

        q_weights_base_name = f'language_model.model.layers.{l}.self_attn.q_proj.weight'
        k_weights_base_name = f'language_model.model.layers.{l}.self_attn.k_proj.weight'
        v_weights_base_name = f'language_model.model.layers.{l}.self_attn.v_proj.weight'

        checkpoint[q_weights_base_name] = param_to_weights(qkv_weights[q_slice].reshape(-1, hidden_size))
        checkpoint[k_weights_base_name] = param_to_weights(qkv_weights[k_slice].reshape(-1, hidden_size))
        checkpoint[v_weights_base_name] = param_to_weights(qkv_weights[v_slice].reshape(-1, hidden_size))

        # attention dense
        o_weight = model.state_dict()[f'model.language_model.encoder.layers.{l}.self_attention.dense.weight']
        o_weight_base_name = f'language_model.model.layers.{l}.self_attn.o_proj.weight'
        checkpoint[o_weight_base_name] = param_to_weights(o_weight)

        # mlp
        mlp_weights = model.state_dict()[f'model.language_model.encoder.layers.{l}.mlp.dense_h_to_4h.weight']
        mlp_down_proj_weight = mlp_weights[:ffn_hidden_size, :]
        mlp_gate_proj_weight = mlp_weights[ffn_hidden_size:, :]

        mlp_down_proj_base_name = f'language_model.model.layers.{l}.mlp.gate_proj.weight'
        mlp_gate_proj_base_name = f'language_model.model.layers.{l}.mlp.up_proj.weight'

        checkpoint[mlp_down_proj_base_name] = param_to_weights(mlp_down_proj_weight)
        checkpoint[mlp_gate_proj_base_name] = param_to_weights(mlp_gate_proj_weight)

        mlp_up_proj_weight = model.state_dict()[f'model.language_model.encoder.layers.{l}.mlp.dense_4h_to_h.weight']
        mlp_up_proj_base_name = f'language_model.model.layers.{l}.mlp.down_proj.weight'
        checkpoint[mlp_up_proj_base_name] = param_to_weights(mlp_up_proj_weight)

        # layernorm
        input_ln_weight = model.state_dict()[f'model.language_model.encoder.layers.{l}.input_layernorm.weight']
        input_ln_base_name = f'language_model.model.layers.{l}.input_layernorm.weight'
        checkpoint[input_ln_base_name] = param_to_weights(input_ln_weight)

        post_attn_ln_weight = model.state_dict()[f'model.language_model.encoder.layers.{l}.post_attention_layernorm.weight']
        post_attn_ln_base_name = f'language_model.model.layers.{l}.post_attention_layernorm.weight'
        checkpoint[post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

        print(f"done layer {l}")

    final_ln_weight = model.state_dict()[f'model.language_model.encoder.final_layernorm.weight']
    final_ln_base_name = f'language_model.model.norm.weight'
    checkpoint[final_ln_base_name] = param_to_weights(final_ln_weight)

    output_layer_weight = model.state_dict()[f'model.language_model.output_layer.weight']
    output_layer_base_name = f'language_model.lm_head.weight'
    checkpoint[output_layer_base_name] = param_to_weights(output_layer_weight)

    torch.save(checkpoint, os.path.join(output_hf_file, "pytorch_model.bin"))
    logging.info(f"Weights saved to {output_hf_file}")
    del(checkpoint)

    return dtype, tokenizer


def replace_hf_weights_and_tokenizer(
    weights_file, output_hf_path, input_tokenizer_file
):
    if args.processor_type == "clip":
        config = LlavaConfig.from_pretrained(weights_file)
        model = LlavaForConditionalGeneration(config)
    else:
        config = LlavaNextConfig.from_pretrained(weights_file)
        model = LlavaNextForConditionalGeneration(config)
    nemo_exported = torch.load(os.path.join(weights_file, "pytorch_model.bin"), map_location="cpu")
    model.resize_token_embeddings(model.vocab_size)

    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_file)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    tokenizer.save_pretrained(output_hf_path)
    logging.info(f"Tokenizer saved to {output_hf_path}")

    processor_config = PROCESSOR_CONFIG[args.processor_type]
    processor_file = os.path.join(output_hf_path, "preprocessor_config.json")
    with open(processor_file, 'w') as json_file:
        json.dump(processor_config, json_file, indent=4)

    maping_results = model.load_state_dict(nemo_exported)
    model.save_pretrained(output_hf_path)
    logging.info(f"Full HF model saved to {output_hf_path}")

if __name__ == '__main__':
    args = get_args()
    if not args.hf_output_tokenizer and args.hf_output_path:
        args.hf_output_tokenizer = args.hf_output_path
    dtype, tokenizer = convert(args.input_name_or_path, args.input_tokenizer, args.output_path, precision=args.precision, cpu_only=args.cpu_only)
    if args.hf_output_path:
        replace_hf_weights_and_tokenizer(
            args.output_path,
            args.hf_output_path,
            args.input_tokenizer,
        )
    else:
        logging.info("`hf-in-path` and/or `hf-out-path` not provided, not generating full HF model.")
        logging.info(f".bin file is saved to {args.output_path}")