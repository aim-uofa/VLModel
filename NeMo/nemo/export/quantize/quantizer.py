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

import tarfile
from contextlib import nullcontext
from typing import List, Optional

import torch
import torch.distributed as dist
from megatron.core import mpu, parallel_state
from megatron.core.transformer.module import Float16Module
from omegaconf import OmegaConf
from omegaconf.omegaconf import DictConfig, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from tqdm import tqdm

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging
from nemo.utils.distributed import temporary_directory
from nemo.utils.model_utils import load_config, save_artifacts, unwrap_model

try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_tensorrt_llm_checkpoint
    from modelopt.torch.utils.distributed import set_data_parallel_group, set_tensor_parallel_group

    HAVE_MODELOPT = True

except (ImportError, ModuleNotFoundError) as e:
    HAVE_MODELOPT = False
    HAVE_MODELOPT_ERROR = e


class Quantizer:
    """
    Post-training quantization of Nemo checkpoints.

    PTQ converts selected model layers to low-precision format (e.g., INT4, FP8) for efficient serving.
    The process consist of several steps:

        1. Loading a Nemo model from disk using appropriate parallelism strategy
        2. Calibrating the model to obtain appropriate algorithm-specific scaling factors
        3. Producing output directory or .qnemo tarball with model config (json),
           quantized weights (safetensors) and tokenizer config (yaml).

    The output directory (or .qnemo file) produced is intended to be consumed by TensorRT-LLM toolbox
    for efficient inference. This can be achieved using Nemo inference containers.

    Currently supported and tested model family is Llama2. Model type needs to be specified in
    the quantization command with decoder_type parameter on exporting (see below). Quantizing other
    model families is experimental and might not be fully supported.

    Available quantization methods are listed in QUANT_CFG_CHOICES dictionary below.
    Please consult Model Optimizer documentation https://nvidia.github.io/TensorRT-Model-Optimizer/ for details.
    You can also inspect different choices in examples/nlp/language_modeling/conf/megatron_quantization.yaml
    for quantization algorithms and calibration data as well as recommended settings.

    Quantization algorithm can also be conveniently set to 'null' to perform only weights export step
    for TensorRT-LLM deployment. This is useful to getting baseline results for a full-precision model.
    """

    def __init__(
        self,
        quantization_config: DictConfig,
        inference_config: DictConfig,
        export_config: DictConfig,
        trainer_config: DictConfig,
    ):
        if not HAVE_MODELOPT:
            raise RuntimeError("nvidia-modelopt is needed to use Quantizer") from HAVE_MODELOPT_ERROR
        QUANT_CFG_CHOICES = {
            "int8": mtq.INT8_DEFAULT_CFG,
            "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
            "fp8": mtq.FP8_DEFAULT_CFG,
            "int4_awq": mtq.INT4_AWQ_CFG,
            "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
        }
        SUPPORTED_DTYPE = [16, "16", "bf16"]  # Default precision for non-quantized layers
        assert export_config.dtype in SUPPORTED_DTYPE
        assert quantization_config.algorithm is None or quantization_config.algorithm in QUANT_CFG_CHOICES
        self.quantization_config = quantization_config
        self.inference_config = inference_config
        self.export_config = export_config
        self.trainer_config = trainer_config
        if quantization_config.algorithm is not None:
            quant_cfg = QUANT_CFG_CHOICES[quantization_config.algorithm]

            if "awq" in quantization_config.algorithm:
                weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
                if isinstance(weight_quantizer, list):
                    weight_quantizer = weight_quantizer[0]
                weight_quantizer["block_sizes"][-1] = quantization_config.awq_block_size

            # Always turn on FP8 kv cache to save memory footprint.
            # For int8_sq, we use int8 kv cache.
            # TODO: Investigate why enabling FP8 kv cache will cause accuracy regressions for Nemotron.
            enable_quant_kv_cache = (
                "int8" not in quantization_config.algorithm and export_config.decoder_type != "gptnext"
            )
            print(f'{"Enable" if enable_quant_kv_cache else "Disable"} KV cache quantization')
            quant_cfg["quant_cfg"]["*output_quantizer"] = {
                "num_bits": 8 if quantization_config.algorithm == "int8_sq" else (4, 3),
                "axis": None,
                "enable": enable_quant_kv_cache,
            }

            self.quant_cfg = quant_cfg
        else:
            self.quant_cfg = None

    def _load_model(
        self,
        model_file: str,
        tensor_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_size: Optional[int] = None,
    ):
        """Load model using ModelOpt layer spec for quantization."""
        model_cfg = self._load_and_modify_config(model_file, tensor_model_parallel_size, pipeline_model_parallel_size)

        trainer = Trainer(strategy=NLPDDPStrategy(), **self.trainer_config)
        connector = NLPSaveRestoreConnector()

        model = MegatronGPTModel.restore_from(
            restore_path=model_file,
            trainer=trainer,
            override_config_path=model_cfg,
            save_restore_connector=connector,
        )
        model.freeze()

        try:
            model.model.module.language_model.encoder.activations_checkpoint_method = None
        except AttributeError:
            pass

        self._check_ddp_initialized(model)

        if dist.get_rank() == 0:
            print(model)

        return model

    @staticmethod
    def _check_ddp_initialized(model):
        if not parallel_state.is_initialized():

            def dummy():
                return

            if model.trainer.strategy.launcher is not None:
                model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
            model.trainer.strategy.setup_environment()

        set_data_parallel_group(mpu.get_data_parallel_group())
        set_tensor_parallel_group(mpu.get_tensor_model_parallel_group())

    @staticmethod
    def _load_and_modify_config(
        model_file: str,
        tensor_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_size: Optional[int] = None,
    ):
        model_cfg = load_config(model_file)

        with open_dict(model_cfg):
            model_cfg.activations_checkpoint_method = None
            model_cfg.activations_checkpoint_granularity = None
            model_cfg.sequence_parallel = False
            if tensor_model_parallel_size is not None:
                model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
            if pipeline_model_parallel_size is not None:
                model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
            # Only custom ModelOpt spec is supported for PTQ: this custom spec is largely based on local Megatron-LM
            # layer definitions to avoid Transformer Engine implementations that are currently not supported.
            # This layer spec also requires RoPE fusion to be disabled for tensor view operations in attention
            # layer implementation from megatron/core/transformer/dot_product_attention.py to be functional.
            model_cfg.name = "modelopt"
            model_cfg.apply_rope_fusion = False

        return model_cfg

    @staticmethod
    def _sample_output(model):
        """Generate sample output for a model instance."""
        if torch.distributed.get_rank() == 0:
            print("Generating sample output for a model...")

        response = model.generate(
            inputs=[
                "Born in north-east France, Soyer trained as a",
                "Born in California, Soyer trained as a",
            ],
            length_params={
                "max_length": 100,
                "min_length": 100,
            },
        )

        if torch.distributed.get_rank() == 0:
            print(f'Example NeMo output after PTQ: {response["sentences"]}"')

    def quantize(
        self,
        model_file: str,
        dataloader: Optional[List[List[str]]],
        tensor_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_size: Optional[int] = None,
    ):
        """Quantize model checkpoint using given dataloader and optional custom parallelism settings."""
        model = self._load_model(model_file, tensor_model_parallel_size, pipeline_model_parallel_size)

        if self.quantization_config.algorithm is None:
            return model

        model.set_inference_config(OmegaConf.to_container(self.inference_config))

        def forward_loop(model):
            print("Calibrating the model...")
            for i, batch in enumerate(tqdm(dataloader)):
                model.predict_step(batch, i)

        model = mtq.quantize(model, self.quant_cfg, forward_loop)

        if self.export_config == "gptnext":
            # We found squared_relu may have an under-calibration problem.
            # Clamp the scaling_factor with a min threshold to avoid under-calibration.
            maxbound = 0
            if self.quantization_config.algorithm == "fp8":
                maxbound = 448
            elif self.quantization_config.quantization.algorithm == "int8_sq":
                maxbound = 127
            model = mtq.postprocess_amax(
                model, "*input_quantizer", lambda amax: torch.clamp(amax, min=0.01 * maxbound)
            )

        if dist.get_rank() == 0:
            mtq.print_quant_summary(model)

        return model

    def export(self, model, model_save: str):
        """Export model to '.qnemo' format for TensorRT-LLM engine build."""
        torch_dtype = torch_dtype_from_precision(self.export_config.dtype)

        self._sample_output(model)

        if model.cfg.megatron_amp_O2:
            model.model = unwrap_model(model.model, Float16Module)

        # Setup model export handling: temporary directory for
        # '.qnemo' tarball or directly write to model_save
        save_qnemo = model_save.endswith(".qnemo")
        if save_qnemo:
            export_handler = temporary_directory()
        else:
            export_handler = nullcontext(enter_result=model_save)

        with export_handler as export_dir:
            export_tensorrt_llm_checkpoint(
                model=model,
                decoder_type=self.export_config.decoder_type,
                dtype=torch_dtype,
                export_dir=export_dir,
                inference_tensor_parallel=self.export_config.inference_tensor_parallel,
                inference_pipeline_parallel=self.export_config.inference_pipeline_parallel,
                use_nfs_workspace=self.export_config.inference_pipeline_parallel == 1
                and model.cfg.pipeline_model_parallel_size > 1,
            )
            dist.barrier()  # Wait until all ranks complete export_model_config step
            if dist.get_rank() == 0:
                logging.info(f"Exporting quantized weights, model artifacts, and tokenizer config to {model_save}...")
                save_artifacts(model, export_dir)
                if save_qnemo:
                    with tarfile.open(model_save, "w:gz") as tar:
                        tar.add(export_dir, arcname="./")
