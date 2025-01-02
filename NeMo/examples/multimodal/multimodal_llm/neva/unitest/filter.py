import json
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
import copy
from tqdm import tqdm
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.multimodal.data.neva.neva_dataset import preprocess_nvgpt, preprocess_multimodal, preprocess_llama_2,preprocess_nv_dpo, preprocess_v1
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


def read_json(input_path):
    for line in open(input_path, "r"):
        record = json.loads(line)

class DatasetFilter():
    def __init__(self, cfg):
        self.cfg = cfg
    
    def generate_dataset(self, data_input_path):
        cfg = self.cfg
        tokenizer = AutoTokenizer('checkpoints/llava3_tokenizer')
        image_processor = AutoImageProcessor.from_pretrained('checkpoints/siglip-so400m-patch14-384')
        image_processor.crop_size = (768, 768)
        image_processor.size = (768, 768)

        dataset = NevaDataset(
            tokenizer=tokenizer,
            data_path='datasets/LLaVA-finetune/SFT_annotations_0511_for_llava.json',
            # data_path='debug_nan_samples.json',
            multimodal_cfg=dict(
                is_multimodal=True,
                sep_image_conv_front=False,
                conv_template='llama_2',
                crop_size=(768, 768),
                image_token_len=729,
                image_folder='datasets/',
                image_aspect_ratio='pad',
                use_im_start_end=False,
                image_processor=image_processor,
                add_extra_token=1,
                context_length=2048,
                datasets_exclude=[],
                from_hf=True
            ),
        )
        return dataset
    
    def filter_dataset(self, data_input_path, data_output_path):
        self.dataset = self.generate_dataset(data_input_path)
        outputs = []
        
        for i in tqdm(range(len(self.dataset.list_data_dict))):
            sources = self.dataset.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            if 'image' in sources[0] and sources[0]['image']:
                sources = preprocess_multimodal(
                    copy.deepcopy(sources),
                    self.dataset.multimodal_cfg,
                    self.dataset.multimodal_cfg['image_token_len'],
                    use_plain=(self.dataset.conv_template == "plain"),
                )
            else:
                sources = copy.deepcopy(sources)
            
            if self.dataset.conv_template in ["nvgpt", "nv_steerlm"]:
                data_dict = preprocess_nvgpt(sources, self.dataset.tokenizer, self.dataset.multimodal_cfg,)
            elif self.dataset.conv_template == "nv_dpo":
                data_dict = preprocess_nv_dpo(sources, self.dataset.tokenizer, self.dataset.multimodal_cfg,)
            elif self.dataset.conv_template == "v1":
                data_dict = preprocess_v1(sources, self.dataset.tokenizer, self.dataset.multimodal_cfg,)
            elif self.dataset.conv_template == "llama_2":
                data_dict = preprocess_llama_2(sources, self.dataset.tokenizer, self.dataset.multimodal_cfg,)
            elif self.dataset.conv_template == "llama_3":
                data_dict = preprocess_llama_3(sources, self.dataset.tokenizer, self.dataset.multimodal_cfg,)
            elif self.dataset.conv_template == "plain":
                data_dict = preprocess_plain(sources, self.dataset.tokenizer, self.dataset.multimodal_cfg,)
            elif self.dataset.conv_template == "custom":
                data_dict = preprocess_custom(sources, self.dataset.tokenizer, self.dataset.multimodal_cfg,)
            else:
                raise ValueError(f"Conversation template `{self.dataset.conv_template}` is not supported in Neva now.")

            if isinstance(i, int):
                data_dict = dict(tokens=data_dict["tokens"][0], labels=data_dict["labels"][0])
            
            max_len = data_dict['tokens'].shape[0]
            max_len = (max_len - 1) // 64 * 64 + 64
            pad_len = max_len - data_dict['tokens'].shape[0]
            # data_dict['tokens'] = F.pad(data_dict['tokens'], (0, pad_len), 'constant', self.dataset.tokenizer.pad_id)
            # data_dict['labels'] = F.pad(data_dict['labels'], (0, pad_len), 'constant', -1)
            
            tokens = data_dict['tokens']
            labels = data_dict['labels']
            labels[labels == -1] = 0
            if labels.sum() == 0:
                continue
            else:
                outputs.append(sources)
        
        with open(data_output_path, 'w') as file:
            json.dump(data, file)


@hydra_runner(config_path="/workspace/NeMo/examples/multimodal/multimodal_llm/neva/conf", config_name="llava_llama3_8b_finetune")
def main(cfg) -> None:

    datasetfilter = DatasetFilter(cfg)
    data_input_path = 'datasets/LLaVA-finetune/SFT_annotations_0511_for_llava.json'
    data_output_path = 'datasets/LLaVA-finetune/SFT_annotations_0511_for_llava_filter.json'
    datasetfilter.filter_dataset(data_input_path, data_output_path)

if __name__ == '__main__':
    main()