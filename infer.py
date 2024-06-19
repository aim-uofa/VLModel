import torch
from PIL import Image
from HawkLlama.model import LlavaNextProcessor, LlavaNextForConditionalGeneration
from HawkLlama.utils.conversation import conv_llava_llama_3, DEFAULT_IMAGE_TOKEN

processor = LlavaNextProcessor.from_pretrained("AIM-ZJU/HawkLlama_8b")

model = LlavaNextForConditionalGeneration.from_pretrained("AIM-ZJU/HawkLlama_8b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True) 
model.to("cuda:0")

image_file = "assets/coin.png"
image = Image.open(image_file).convert('RGB')

prompt = "what coin is that?"
prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

conversation = conv_llava_llama_3.copy()
user_role_ind = 0
bot_role_ind = 1
conversation.append_message(conversation.roles[user_role_ind], prompt)
conversation.append_message(conversation.roles[bot_role_ind], "")
prompt = conversation.get_prompt()
inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
output = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=2048, do_sample=False, use_cache=True)

print(processor.decode(output[0], skip_special_tokens=True))