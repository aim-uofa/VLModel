from PIL import Image
import torch
import json

from our_llava import LlavaNextForConditionalGeneration, LlavaNextProcessor
from nemo.collections.multimodal.data.neva.conversation import conv_llava_llama_2
from transformers import AutoImageProcessor

json_path = 'llava_test_examples/input_prompts_test.json'
image_root = 'llava_test_examples/images'
ckpt_root = '/mnt/nas/share/home/lht/code/nemo_workspace/NeMo/our_checkpoints/llava_llama3_8b_siglip_tile_finetune_hf_0429_2ep'
# ckpt_root = 'checkpoints/llava_llama3_8b_siglip_tile_finetune_hf_0429'

with open(json_path, 'r') as f:
    data_list = json.load(f)
    
processor = LlavaNextProcessor.from_pretrained(ckpt_root)
# siglip_processor = AutoImageProcessor.from_pretrained(ckpt_root)
# processor.image_processor = siglip_processor
model = LlavaNextForConditionalGeneration.from_pretrained(ckpt_root, torch_dtype=torch.bfloat16).cuda()

import os
import numpy as np
import random
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

results = []
for i, data in enumerate(data_list):
    ori_rompt = data['prompt']
    prompt = '<image>\n' + ori_rompt

    image_name = data['image']
    image_dir = image_root + '/' + image_name
    # image = Image.open(image_dir)
    conversation = conv_llava_llama_2.copy()

    user_role_ind = 0
    bot_role_ind = 1
    conversation.append_message(conversation.roles[user_role_ind], prompt)
    conversation.append_message(conversation.roles[bot_role_ind], "")
    prompt = conversation.get_prompt()
    inputs = processor(prompt, image_dir, return_tensors="pt", image_aspect_ratio='pad').to("cuda:0")
    # model = model.to(torch.float32)
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    seed_everything(1234)
    output = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=2048,
        temperature=0.2, 
        do_sample=False,
        top_p=0.9,
        use_cache=True
    )
    input_lens = inputs['input_ids'].shape[-1]
    out_answer = processor.decode(output[0,input_lens:], skip_special_tokens=True)
    print('=' * 60)
    print(ori_rompt)
    print(out_answer)
    single_result = {
        'id':i,
        'prompt': ori_rompt,
        'outputs': out_answer,
    }
    results.append(single_result)
with open('llava_test_examples/neva_output.json', 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False)
