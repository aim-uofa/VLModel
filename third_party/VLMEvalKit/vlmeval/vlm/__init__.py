import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .base import BaseModel
from .HawkLlama import HawkLlama_llama3_vlm
