import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...utils import DATASET_TYPE
import sys
import HawkLlama.utils.conversation as conversation_lib
from HawkLlama.model import LlavaNextForConditionalGeneration, LlavaNextProcessor

from transformers import StoppingCriteria, StoppingCriteriaList

class LLaVAStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_tokens):
        self.stop_tokens = [tokenizer.encode(token, add_special_tokens=False)[0] for token in stop_tokens]
        self.tokenizer = tokenizer

    def __call__(self, input_ids, *args, **kwargs):
        last_token_id = input_ids[0][-1]
        return last_token_id in self.stop_tokens


class HawkLlama_llama3_vlm(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_pth='AIM-ZJU/HawkLlama_8b', **kwargs):
        self.model_pth = model_pth
        if '34b' in model_pth.lower():
            self.processor = LlavaNextProcessor.from_pretrained(self.model_pth, use_fast=False)
        else:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_pth)
        flash_attn_flag = False
        try:
            import flash_attn
            flash_attn_flag = True
        except ImportError:
            pass

        if flash_attn_flag:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_pth, torch_dtype=torch.bfloat16, use_flash_attention_2=True)
        else:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_pth, torch_dtype=torch.bfloat16)
        model.config.image_token_index = self.processor.tokenizer.encode("<image>", add_special_tokens=False)[0]
        stop_tokens_list = [self.processor.tokenizer.eos_token, '</s>', '<|im_end|>', '<|eot_id|>', '<|end_of_text|>']
        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(LLaVAStoppingCriteria(self.processor.tokenizer, stop_tokens_list))
        model = model.eval()
        self.model = model.cuda()
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, stopping_criteria=stopping_criteria)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def apply_prompt_template(self, prompt):
        template = (
            '<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. '
            'You are able to understand the visual content that the user provides, '
            'and assist the user with a variety of tasks using natural language.<|eot_id|>'
            '<|start_header_id|>user<|end_header_id|>\n\nPLACEHOLDER<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'
        )

        prompt = template.replace('PLACEHOLDER', f'<image>\n{prompt}')
        return prompt

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'
        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        prompt = prompt.replace('<image>', '[ImageHere]')
        if prompt.find('[ImageHere]') != prompt.rfind('[ImageHere]'):
            prompt += '\nThere exists multiple images in the conversation, but only the first one is displayed.'
        image = Image.open(image_path).convert('RGB')
        prompt = self.apply_prompt_template(prompt)
        inputs = self.processor(prompt, image, return_tensors='pt').to('cuda')
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        output = self.model.generate(**inputs, **self.kwargs)
        answer = self.processor.decode(output[0], skip_special_token=True)
        if '<s>' in answer:
            answer = answer.replace('<s>', '').strip()
        if '<|begin_of_text|>' in answer:
            answer = answer.split('<|begin_of_text|>')[1].strip()
        if '[/INST]' in answer:
            answer = answer.split('[/INST]')[1].strip()
        elif 'ASSISTANT:' in answer:
            answer = answer.split('ASSISTANT:')[1].strip()
        elif 'assistant\n' in answer:
            answer = answer.split('assistant\n')[1].strip()
        elif '<|start_header_id|>assistant<|end_header_id|>\n\n' in answer:
            answer = answer.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[1].strip()

        if '</s>' in answer:
            answer = answer.split('</s>')[0].strip()
        if '<|im_end|>' in answer:
            answer = answer.split('<|im_end|>')[0].strip()
        if '<|end_of_text|>' in answer:
            answer = answer.split('<|end_of_text|>')[0].strip()
        if '<|eot_id|>' in answer:
            answer = answer.split('<|eot_id|>')[0].strip()

        return answer