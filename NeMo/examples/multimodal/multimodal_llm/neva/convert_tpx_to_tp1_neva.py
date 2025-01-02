import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import yaml


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path", type=str, default=None, required=True, help="Path to .nemo file or extracted folder",
    )
    parser.add_argument(
        "--output_name_or_path", type=str, default=None, help="Path to .nemo file or extracted folder",
    )
    parser.add_argument(
        "--tmp_name_or_path", type=str, default=None, help="Path to save extracted folder",
    )
    args = parser.parse_args()
    return args

def convert(input_name_or_path, output_name_or_path, tmp_name_or_path):
    is_nemo_input = input_name_or_path.endswith(".nemo")
    if is_nemo_input:
        if tmp_name_or_path:
            tmp_name_or_path = tmp_name_or_path
        else:
            tmp_name_or_path = input_name_or_path.split(".nemo")[0]
        os.system("tar -xvf {} -C {}".format(input_name_or_path, tmp_name_or_path))

        input_name_or_path = tmp_name_or_path

    os.makedirs(output_name_or_path, exist_ok=True)
    files = os.listdir(input_name_or_path)
    ckpt_files = []
    model_config_file = None
    for file in files:
        if "mp_rank" in file:
            ckpt_files.append(file)
        elif file.endswith(".yaml"):
            model_config_file = file
    if model_config_file is None:
        raise FileNotFoundError("model_config.yaml is needed.")
    with open(os.path.join(input_name_or_path, model_config_file), 'r') as file:
        model_config = yaml.safe_load(file)
    print(model_config)
    tp = model_config['tensor_model_parallel_size']
    ckpt_files = sorted(ckpt_files)
    ckpt_pth = []
    for file in ckpt_files:
        ckpt_pth.append(torch.load(os.path.join(input_name_or_path, file, "model_weights.ckpt"), map_location="cpu"))
    if not is_nemo_input and 'state_dict' in ckpt_pth[0].keys():
        ckpt_pth = [_['state_dict'] for _ in ckpt_pth]
    state_dict = {}
    for name in ckpt_pth[0]:
        shape = ckpt_pth[0][name].shape
        if len(shape) == 1 or len(shape) == 4:
            param = ckpt_pth[0][name]
        elif name == "model.embedding.word_embeddings.weight" or name == "model.output_layer.weight":
            param = torch.concat([ckpt[name] for ckpt in ckpt_pth], dim=0)
        elif "linear_qkv.weight" in name:
            param = torch.concat([ckpt[name] for ckpt in ckpt_pth], dim=0)
        elif "linear_fc1.weight" in name:
            param = torch.concat([ckpt[name] for ckpt in ckpt_pth], dim=0)
            chunk_param = torch.chunk(param, int(2*tp))
            param = torch.cat(chunk_param[::2]+chunk_param[1::2], dim=0)
        elif "mm_projector" not in name and "vision_encoder" not in name:
            param = torch.concat([ckpt[name] for ckpt in ckpt_pth], dim=-1)
        else:
            param = ckpt_pth[0][name]
        state_dict[name] = param
    torch.save(state_dict, os.path.join(output_name_or_path, "model_weights.ckpt"))
    
    model_config['tensor_model_parallel_size'] = 1
    model_config['mm_cfg']['llm']['from_pretrained'] = None
    model_config['mm_cfg']['pretrain_mm_mlp_adapter'] = None
    model_config['make_vocab_size_divisible_by'] = 512
    with open(os.path.join(output_name_or_path, "model_config.yaml"), 'w') as file:
        yaml.safe_dump(model_config, file)


if __name__ == "__main__":
    args = get_args()
    convert(args.input_name_or_path, args.output_name_or_path, args.tmp_name_or_path)