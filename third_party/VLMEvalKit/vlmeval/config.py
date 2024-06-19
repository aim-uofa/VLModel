from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
LLAVA_V1_7B_MODEL_PTH = 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '


api_models = {
    # GPT-4V series
    'GPT4V': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10),
    'GPT4V_HIGH': partial(GPT4V, model='gpt-4-1106-vision-preview', temperature=0, img_size=-1, img_detail='high', retry=10),
    'GPT4V_20240409': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=512, img_detail='low', retry=10),
    'GPT4V_20240409_HIGH': partial(GPT4V, model='gpt-4-turbo-2024-04-09', temperature=0, img_size=-1, img_detail='high', retry=10),
    # Gemini-V
    'GeminiProVision': partial(GeminiProVision, temperature=0, retry=10),
    # Qwen-VL Series
    'QwenVLPlus': partial(QwenVLAPI, model='qwen-vl-plus', temperature=0, retry=10),
    'QwenVLMax': partial(QwenVLAPI, model='qwen-vl-max', temperature=0, retry=10),
    # Reka Series
    'RekaEdge': partial(Reka, model='reka-edge-20240208'), 
    'RekaFlash': partial(Reka, model='reka-flash-20240226'), 
    'RekaCore': partial(Reka, model='reka-core-20240415'), 
    # Internal Only
    'GPT4V_INT': partial(GPT4V_Internal, model='gpt-4-vision-preview', temperature=0, img_size=512, img_detail='low', retry=10),
    'Step1V': partial(Step1V, temperature=0, retry=10),
    'Claude3V_Opus': partial(Claude3V, model='claude-3-opus-20240229', temperature=0, retry=10),
    'Claude3V_Sonnet': partial(Claude3V, model='claude-3-sonnet-20240229', temperature=0, retry=10),
    'Claude3V_Haiku': partial(Claude3V, model='claude-3-haiku-20240307', temperature=0, retry=10),
}

hawkllama_series = {
    'hawkllama_llama3_vlm': partial(HawkLlama_llama3_vlm, model_pth='AIM-ZJU/HawkLlama_8b'),
}

supported_VLM = {}

model_groups = [
    hawkllama_series, api_models, 
]

for grp in model_groups:
    supported_VLM.update(grp)

transformer_ver = {}

if __name__ == '__main__':
    import sys
    ver = sys.argv[1]
    if ver in transformer_ver:
        print(' '.join(transformer_ver[ver]))