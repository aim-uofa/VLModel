from PIL import Image
import torch
import json

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers import LlavaForConditionalGeneration, AutoProcessor
from nemo.collections.multimodal.data.neva.conversation import conv_llava_v1


json_path = 'llava_test_examples/input_prompts.json'
image_root = 'llava_test_examples/images'
ckpt_root = 'checkpoints/llava-v1.6-vicuna-7b-hf'
# ckpt_root = 'checkpoints/llava-1.5-7b-hf'

with open(json_path, 'r') as f:
    data_list = json.load(f)

if 'llava-v1.6-vicuna-7b-hf' in ckpt_root:
    llava_version = '1.6'
else:
    llava_version = '1.5'
    
if llava_version == '1.6':
    processor = LlavaNextProcessor.from_pretrained(ckpt_root)
    model = LlavaNextForConditionalGeneration.from_pretrained(ckpt_root, torch_dtype=torch.bfloat16).cuda()
else:
    processor = AutoProcessor.from_pretrained(ckpt_root)
    model = LlavaForConditionalGeneration.from_pretrained(ckpt_root, torch_dtype=torch.bfloat16).cuda()


results = []
for i, data in enumerate(data_list):
    ori_rompt = data['prompt']
    prompt = '<image>\n' + ori_rompt

    image_name = data['image']
    image_dir = image_root + '/' + image_name
    image = Image.open(image_dir)
    conversation = conv_llava_v1.copy()

    user_role_ind = 0
    bot_role_ind = 1
    conversation.append_message(conversation.roles[user_role_ind], prompt)
    conversation.append_message(conversation.roles[bot_role_ind], "")
    prompt = conversation.get_prompt()

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    output = model.generate(
        **inputs,
        max_new_tokens=2048,
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
with open('llava_test_examples/llava_' + llava_version + '_output.json', 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False)
