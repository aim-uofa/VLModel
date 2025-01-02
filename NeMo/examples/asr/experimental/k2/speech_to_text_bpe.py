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

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts/tokenizers/ in order to prepare the tokenizer.

```sh
python <NEMO_ROOT>/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest=<path to train manifest files, seperated by commas>
        OR
        --data_file=<path to text data, seperated by commas> \
        --data_root="<output directory>" \
        --vocab_size=<number of tokens in vocabulary> \
        --tokenizer=<"spe" or "wpe"> \
        --no_lower_case \
        --spe_type=<"unigram", "bpe", "char" or "word"> \
        --spe_character_coverage=1.0 \
        --log
```

# [FOR MMI LOSS ONLY] Building a token-level LM for the model training
```sh
python experimental/k2/make_token_lm.py \
        --manifest=<comma separated list of manifest files> \
        --tokenizer_dir=<path to directory of tokenizer (not full path to the vocab file!)> \
        --tokenizer_type=<either `bpe` or `wpe`> \
        --output_file=<path to store the token LM> \
        --lm_builder=<NEMO_ROOT>/scripts/asr_language_modeling/ngram_lm/make_phone_lm.py \
        --ngram_order=2 \
        --do_lowercase
```

# Training the model
```sh
python speech_to_text_ctc_bpe.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.type=<either `bpe` or `wpe`> \
    trainer.devices=-1 \
    trainer.accelerator="ddp" \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>" \
    model.graph_module_cfg.criterion_type=<either `ml` or `map`> \
    model.graph_module_cfg.loss_type=<either `ctc` or `mmi`> \
    model.graph_module_cfg.transcribe_training=False \
    model.graph_module_cfg.split_batch_size=0 \
    model.graph_module_cfg.background_cfg.topo_type=<`default` or `compact` or `shared_blank` or `minimal`> \
    model.graph_module_cfg.background_cfg.topo_with_self_loops=True \
```

# If graph_module_cfg.criterion_type=`map`, you can set the following parameters:
    model.graph_module_cfg.background_cfg.token_lm=<path to the token LM> \
    model.graph_module_cfg.background_cfg.intersect_pruned=False \
    model.graph_module_cfg.background_cfg.boost_coeff=0.0
"""
import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models.configs.k2_sequence_models_config import EncDecK2SeqModelConfig
from nemo.collections.asr.models.k2_sequence_models import EncDecK2SeqModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="experimental/k2/conf/citrinet", config_name="citrinet_mmi_1024.yaml")
def main(cfg: EncDecK2SeqModelConfig):
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecK2SeqModelBPE(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == "__main__":
    main()
