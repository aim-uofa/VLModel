import os
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from nemo.core.config import hydra_runner

import torch
from transformers import  AutoImageProcessor
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.multimodal.data.neva.neva_dataset import NevaDataset, DataCollatorForSupervisedDataset, NevaDataloader
from tqdm import tqdm
from torch.utils.data import DataLoader
from nemo.core.config import hydra_runner
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
try:
    from megatron.core import InferenceParams, dist_checkpointing, parallel_state
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

from nemo.collections.vision.data.megatron.data_samplers import MegatronVisionPretrainingRandomSampler
from nemo.collections.vision.data.megatron.data_samplers import MegatronModalityPretrainingRandomSampler
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning import Trainer

class LitModel(pl.LightningModule):
    def __init__(self, cfg, input_size):
        super().__init__()
        self.layer = torch.nn.Linear(input_size, 10)  # 简单的线性层
        self.cfg = cfg
        self.global_batch_size = 128
        

    def forward(self, x):
        return self.layer(x)

    def generate_dataset(self):
        cfg = self.cfg
        tokenizer = AutoTokenizer('checkpoints/llava3_tokenizer')
        image_processor = AutoImageProcessor.from_pretrained('checkpoints/siglip-so400m-patch14-384')
        image_processor.crop_size = (768, 768)
        image_processor.size = (768, 768)

        dataset = NevaDataset(
            tokenizer=tokenizer,
            # data_path='datasets/LLaVA-finetune/SFT_annotations_0429_for_llava.json',
            # data_path='datasets/LLaVA-finetune/SFT_annotations_0518_for_llava.json',
            data_path='datasets/LLaVA-finetune/llava_v1_5_mix665k.json',
            
            # data_path='debug_nan_samples.json',
            multimodal_cfg=dict(
                is_multimodal=True,
                sep_image_conv_front=False,
                conv_template='llama_2',
                crop_size=(768, 768),
                image_token_len=729,
                image_folder='datasets/combined_images',
                image_aspect_ratio='pad',
                use_im_start_end=False,
                image_processor=image_processor,
                add_extra_token=1,
                context_length=4096,
                datasets_exclude=[],
                from_hf=True
            ),
        )
        return dataset

    def train_dataloader(self):
        cfg = self.cfg
        train_dataset = self.generate_dataset()
        self.cfg['model']['data']['dataloader_type'] = 'modality'
        
        self.multi_modal = [True for i in range(self.trainer.world_size)]
        # 这里你需要定义 consumed_samples
        consumed_samples = 0  # 你需要根据你的具体情况来设置这个值
        tokenizer = AutoTokenizer('checkpoints/llava3_tokenizer')
        cfg.model.micro_batch_size = 2
        self.intervel_step = self.global_batch_size // (self.cfg.model.micro_batch_size * self.trainer.world_size)
        if self.cfg['model']['data']['dataloader_type'] == 'cyclic':
            batch_sampler = MegatronVisionPretrainingRandomSampler(
                dataset=train_dataset,  # 确保这里传入的是 train_dataset
                total_samples=len(train_dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=cfg.model.micro_batch_size,
                data_parallel_rank=self.trainer.global_rank,
                data_parallel_size=self.trainer.world_size,
                drop_last=self.cfg.get('drop_last', True),
                data_sharding=False,
            )
        elif self.cfg['model']['data']['dataloader_type'] == 'modality':
            lengths = train_dataset.modality_lengths
            batch_sampler = MegatronModalityPretrainingRandomSampler(
                dataset=train_dataset,
                total_samples=len(train_dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=self.cfg.model.micro_batch_size,
                data_parallel_rank=self.trainer.global_rank,
                data_parallel_size=self.trainer.world_size,
                drop_last=self.cfg.get('drop_last', True),
                mega_batch_mult=self.cfg.get('mega_batch_mult', 1),
                data_sharding=False,
                lengths=lengths,
                global_batch_size=self.global_batch_size,
            )
        else:
            raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        collate_func = DataCollatorForSupervisedDataset(cfg.model, tokenizer)
        data_loader = NevaDataloader(train_dataset, 
                                    batch_sampler=batch_sampler,
                                    shuffle=False, 
                                    num_workers=1, 
                                    collate_fn=collate_func)
        return data_loader


    def training_step(self, batch, batch_idx):
        B, _, _, _, _, _ = batch['media'].shape

        if batch_idx % self.intervel_step == 0 and batch['media'][0].sum() != 0:
            self.multi_modal[self.trainer.global_rank] = True
        elif batch_idx % self.intervel_step == 0 and batch['media'][0].sum() == 0:
            self.multi_modal[self.trainer.global_rank] = False

        for i in range(B):
            if self.multi_modal[self.trainer.global_rank] and batch['media'][int(i)].sum() == 0:
                print('wrong batch')
            elif not self.multi_modal[self.trainer.global_rank] and batch['media'][int(i)].sum() != 0:
                print('wrong batch')
            
        # print('1')
        # pass
        # output = self.forward(batch)
        # loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
        # return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
        return optimizer


@hydra_runner(config_path="../conf", config_name="llava_llama3_8b_finetune")
def main(cfg) -> None:

    # 创建 Trainer 实例
    trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp",use_distributed_sampler=False)

    # 创建模型实例
    model = LitModel(cfg=cfg, input_size=32)

    # 开始训练
    trainer.fit(model)

if __name__ == '__main__':
    main()
