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

import argparse
import logging
import os
import sys
from pathlib import Path

from nemo.deploy import DeployPyTriton
from nemo.export import TensorRTLLM


LOGGER = logging.getLogger("NeMo")


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Deploy nemo models to Triton",
    )
    parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source .nemo file")
    parser.add_argument(
        "-ptnc",
        "--ptuning_nemo_checkpoint",
        nargs='+',
        type=str,
        required=False,
        help="Source .nemo file for prompt embeddings table",
    )
    parser.add_argument(
        '-ti', '--task_ids', nargs='+', type=str, required=False, help='Unique task names for the prompt embedding.'
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        required=False,
        choices=["gptnext", "gpt", "llama", "falcon", "starcoder", "mixtral", "gemma"],
        help="Type of the model. gptnext, gpt, llama, falcon, and starcoder are only supported."
        " gptnext and gpt are the same and keeping it for backward compatibility",
    )
    parser.add_argument("-tmn", "--triton_model_name", required=True, type=str, help="Name for the service")
    parser.add_argument("-tmv", "--triton_model_version", default=1, type=int, help="Version for the service")
    parser.add_argument(
        "-trp", "--triton_port", default=8000, type=int, help="Port for the Triton server to listen for requests"
    )
    parser.add_argument(
        "-tha", "--triton_http_address", default="0.0.0.0", type=str, help="HTTP address for the Triton server"
    )
    parser.add_argument(
        "-tmr", "--triton_model_repository", default=None, type=str, help="Folder for the trt-llm conversion"
    )
    parser.add_argument("-ng", "--num_gpus", default=1, type=int, help="Number of GPUs for the deployment")
    parser.add_argument(
        "-dt",
        "--dtype",
        choices=["bfloat16", "float16", "fp8", "int8"],
        default="bfloat16",
        type=str,
        help="dtype of the model on TensorRT-LLM",
    )
    parser.add_argument("-mil", "--max_input_len", default=256, type=int, help="Max input length of the model")
    parser.add_argument("-mol", "--max_output_len", default=256, type=int, help="Max output length of the model")
    parser.add_argument("-mbs", "--max_batch_size", default=8, type=int, help="Max batch size of the model")
    parser.add_argument("-mnt", "--max_num_tokens", default=None, type=int, help="Max number of tokens")
    parser.add_argument("-ont", "--opt_num_tokens", default=None, type=int, help="Optimum number of tokens")
    parser.add_argument(
        "-mpet", "--max_prompt_embedding_table_size", default=None, type=int, help="Max prompt embedding table size"
    )
    parser.add_argument(
        "-npkc", "--no_paged_kv_cache", default=False, action='store_true', help="Enable paged kv cache."
    )
    parser.add_argument(
        "-drip",
        "--disable_remove_input_padding",
        default=False,
        action='store_true',
        help="Disables the remove input padding option.",
    )
    parser.add_argument(
        "-mbm",
        '--multi_block_mode',
        default=False,
        action='store_true',
        help='Split long kv sequence into multiple blocks (applied to generation MHA kernels). \
                        It is beneifical when batchxnum_heads cannot fully utilize GPU.',
    )
    parser.add_argument(
        "-es", '--enable_streaming', default=False, action='store_true', help="Enables streaming sentences."
    )
    parser.add_argument(
        '--use_lora_plugin',
        nargs='?',
        const=None,
        choices=['float16', 'float32', 'bfloat16'],
        help="Activates the lora plugin which enables embedding sharing.",
    )
    parser.add_argument(
        '--lora_target_modules',
        nargs='+',
        default=None,
        choices=[
            "attn_qkv",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_dense",
            "mlp_h_to_4h",
            "mlp_gate",
            "mlp_4h_to_h",
        ],
        help="Add lora in which modules. Only be activated when use_lora_plugin is enabled.",
    )
    parser.add_argument(
        '--max_lora_rank',
        type=int,
        default=64,
        help='maximum lora rank for different lora modules. '
        'It is used to compute the workspace size of lora plugin.',
    )
    parser.add_argument(
        "-lc", "--lora_ckpt", default=None, type=str, nargs="+", help="The checkpoint list of LoRA weights"
    )
    parser.add_argument(
        "-ucr",
        '--use_cpp_runtime',
        default=False,
        action='store_true',
        help='Use TensorRT LLM C++ runtime',
    )
    parser.add_argument("-dm", "--debug_mode", default=False, action='store_true', help="Enable debug mode")

    args = parser.parse_args(argv)
    return args


def nemo_deploy(argv):
    args = get_args(argv)

    if args.debug_mode:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    LOGGER.setLevel(loglevel)
    LOGGER.info("Logging level set to {}".format(loglevel))
    LOGGER.info(args)

    if args.triton_model_repository is None:
        trt_llm_path = "/tmp/trt_llm_model_dir/"
        LOGGER.info(
            "/tmp/trt_llm_model_dir/ path will be used as the TensorRT LLM folder. "
            "Please set this parameter if you'd like to use a path that has already "
            "included the TensorRT LLM model files."
        )
        Path(trt_llm_path).mkdir(parents=True, exist_ok=True)
    else:
        trt_llm_path = args.triton_model_repository

    if args.nemo_checkpoint is None and args.triton_model_repository is None:
        LOGGER.error(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint."
        )
        return

    if args.nemo_checkpoint is None and not os.path.isdir(args.triton_model_repository):
        LOGGER.error(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint."
        )
        return

    if args.nemo_checkpoint is not None and args.model_type is None:
        LOGGER.error("Model type is required to be defined if a nemo checkpoint is provided.")
        return

    ptuning_tables_files = []
    if not args.ptuning_nemo_checkpoint is None:
        if args.max_prompt_embedding_table_size is None:
            LOGGER.error("max_prompt_embedding_table_size parameter is needed for the prompt tuning table(s).")
            return

        for pt_checkpoint in args.ptuning_nemo_checkpoint:
            ptuning_nemo_checkpoint_path = Path(pt_checkpoint)
            if ptuning_nemo_checkpoint_path.exists():
                if ptuning_nemo_checkpoint_path.is_file():
                    ptuning_tables_files.append(pt_checkpoint)
                else:
                    LOGGER.error("Could not read the prompt tuning tables from {0}".format(pt_checkpoint))
                    return
            else:
                LOGGER.error("File or directory {0} does not exist.".format(pt_checkpoint))
                return

        if args.task_ids is not None:
            if len(ptuning_tables_files) != len(args.task_ids):
                LOGGER.error(
                    "Number of task ids and prompt embedding tables have to match. "
                    "There are {0} tables and {1} task ids.".format(len(ptuning_tables_files), len(args.task_ids))
                )
                return

    trt_llm_exporter = TensorRTLLM(
        model_dir=trt_llm_path,
        lora_ckpt_list=args.lora_ckpt,
        load_model=(args.nemo_checkpoint is None),
        use_python_runtime=(not args.use_cpp_runtime),
    )

    if args.nemo_checkpoint is not None:
        try:
            LOGGER.info("Export operation will be started to export the nemo checkpoint to TensorRT-LLM.")
            trt_llm_exporter.export(
                nemo_checkpoint_path=args.nemo_checkpoint,
                model_type=args.model_type,
                n_gpus=args.num_gpus,
                tensor_parallel_size=args.num_gpus,
                pipeline_parallel_size=1,
                max_input_len=args.max_input_len,
                max_output_len=args.max_output_len,
                max_batch_size=args.max_batch_size,
                max_num_tokens=args.max_num_tokens,
                opt_num_tokens=args.opt_num_tokens,
                max_prompt_embedding_table_size=args.max_prompt_embedding_table_size,
                paged_kv_cache=(not args.no_paged_kv_cache),
                remove_input_padding=(not args.disable_remove_input_padding),
                dtype=args.dtype,
                enable_multi_block_mode=args.multi_block_mode,
                use_lora_plugin=args.use_lora_plugin,
                lora_target_modules=args.lora_target_modules,
                max_lora_rank=args.max_lora_rank,
                save_nemo_model_config=True,
            )
        except Exception as error:
            LOGGER.error("An error has occurred during the model export. Error message: " + str(error))
            return

    try:
        for i, prompt_embeddings_checkpoint_path in enumerate(ptuning_tables_files):
            if args.task_ids is not None:
                task_id = args.task_ids[i]
            else:
                task_id = i

            LOGGER.info(
                "Adding prompt embedding table: {0} with task id: {1}.".format(
                    prompt_embeddings_checkpoint_path, task_id
                )
            )
            trt_llm_exporter.add_prompt_table(
                task_name=str(task_id),
                prompt_embeddings_checkpoint_path=prompt_embeddings_checkpoint_path,
            )
    except Exception as error:
        LOGGER.error("An error has occurred during adding the prompt embedding table(s). Error message: " + str(error))
        return

    try:
        nm = DeployPyTriton(
            model=trt_llm_exporter,
            triton_model_name=args.triton_model_name,
            triton_model_version=args.triton_model_version,
            max_batch_size=args.max_batch_size,
            port=args.triton_port,
            address=args.triton_http_address,
            streaming=args.enable_streaming,
        )

        LOGGER.info("Triton deploy function will be called.")
        nm.deploy()
    except Exception as error:
        LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
        return

    try:
        LOGGER.info("Model serving on Triton is will be started.")
        nm.serve()
    except Exception as error:
        LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
        return

    LOGGER.info("Model serving will be stopped.")
    nm.stop()


if __name__ == '__main__':
    nemo_deploy(sys.argv[1:])
