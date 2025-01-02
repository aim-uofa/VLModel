# Setup

1. 获取docker镜像

```
docker pull nvcr.io/nvidia/pytorch:24.03-py3
```

2. 创建本地磁盘上的工作目录

```
mkdir /mnt/nas/share/home/path/to/your/workspace
```

3. 创建容器并挂载硬盘

```
# 自行修改第5行容器名字和第10行磁盘挂载目录
docker run \
    -it \
    --name CONTAINER_NAME \
    --privileged \
    --ipc=host \
    --net=host \
    --gpus=all \
    -v /mnt/nas/share/home/path/to/your/workspace:/workspace \
    -v /mnt:/mnt \
    nvcr.io/nvidia/pytorch:24.03-py3 \
    bash
```

如果创建容器时报错`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].`需要安装nvidia的包，然后重启docker服务，删除已经创建的容器重新创建。

```
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

4. 在容器内安装NeMo和Megatron-LM

```
cd /workspace
git clone https://github.com/lihengtao/NeMo.git
cd NeMo
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .[nlp]
pip install megatron-core decord
pip install -U Pillow
```

# HF model to NeMo model

```
python scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
    --input_name_or_path checkpoints/vicuna-7b-v1.5 \
    --output_path checkpoints/vicuna-7b-v1.5.nemo
```

除模型的转换外，还需要tokenizer的转换。NeMo在词表额外添加了几个`<extra_xx>`。sentencepiece格式的用脚本转换，huggingface格式的需要手动添加。

# Pretrain

```
export MASTER_PORT=50000
export MASTER_ADDR=10.205.23.131
export WORLD_SIZE=8
export NODE_RANK=0
export GLOO_SOCKET_IFNAME=ens10f0np0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NEMO_CONFIG=llava_7b_pretrain

python examples/multimodal/multimodal_llm/neva/neva_finetune.py \
    model.mm_cfg.llm.from_pretrained=checkpoints/vicuna-7b-v1.5-mcore.nemo \
    model.mm_cfg.vision_encoder.from_pretrained=checkpoints/clip-vit-large-patch14-336 \
    model.data.image_token_len=576 \
    model.data.crop_size=[336,336] \
    model.data.image_size=[336,336] \
    model.mm_cfg.vision_encoder.hidden_size=1024 \
    model.mm_cfg.vision_encoder.class_token_length=1 \
    model.tokenizer.library=huggingface \
    model.tokenizer.model=null \
    model.tokenizer.type=checkpoints/llama2_tokenizer \
    model.data.data_path=datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    model.data.image_folder=datasets/LLaVA-Pretrain/images \
    model.micro_batch_size=32 \
    model.tensor_model_parallel_size=4 \
    model.encoder_seq_length=4096 \
    trainer.num_nodes=1 \
    trainer.max_epochs=-1 \
    trainer.max_steps=2176 \
    model.optim.sched.warmup_steps=65 \
    model.data.conv_template=plain \
    model.mm_cfg.vision_encoder.freeze=True \
    model.mm_cfg.mm_mlp_adapter_type=mlp2x_gelu \
    model.mm_cfg.use_image_tile=False \
    model.data.dataloader_type=cyclic \
    exp_manager.name=llava_vicuna_7b_clip_pretrain_tp4_pp1
```

多节点训练时，`MASTER_ADDR`设为主节点地址，`WORLD_SIZE`设为总GPU数，`NODE_RANK`设置为节点的RANK。例如双节点训练时设置为

```
# 节点0
export MASTER_ADDR=10.205.23.131
export WORLD_SIZE=16
export NODE_RANK=0

# 节点1
export MASTER_ADDR=10.205.23.131
export WORLD_SIZE=16
export NODE_RANK=1
```

# SFT

```
export MASTER_PORT=50000
export MASTER_ADDR=10.205.23.131
export WORLD_SIZE=8
export NODE_RANK=0
export GLOO_SOCKET_IFNAME=ens10f0np0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NEMO_CONFIG=llava_7b_finetune

python examples/multimodal/multimodal_llm/neva/neva_finetune.py \
    model.mm_cfg.llm.from_pretrained=checkpoints/vicuna-7b-v1.5-mcore.nemo \
    model.mm_cfg.vision_encoder.from_pretrained=checkpoints/clip-vit-large-patch14-336 \
    model.data.image_token_len=576 \
    model.data.crop_size=[336,336] \
    model.data.image_size=[336,336] \
    model.mm_cfg.vision_encoder.hidden_size=1024 \
    model.mm_cfg.vision_encoder.class_token_length=1 \
    model.mm_cfg.pretrain_mm_mlp_adapter=nemo_experiments/llava_vicuna_7b_clip_pretrain_tp4_pp1/checkpoints/llava_vicuna_7b_clip_pretrain_tp4_pp1.nemo \
    model.tokenizer.library=huggingface \
    model.tokenizer.model=null \
    model.tokenizer.type=checkpoints/llama2_tokenizer \
    model.data.data_path=datasets/LLaVA-finetune/llava_v1_5_mix665k.json \
    model.data.image_folder=datasets/combined_images \
    model.micro_batch_size=8 \
    model.tensor_model_parallel_size=4 \
    model.encoder_seq_length=4096 \
    trainer.num_nodes=1 \
    trainer.max_epochs=-1 \
    trainer.max_steps=5197 \
    model.optim.sched.warmup_steps=155 \
    model.data.conv_template=v1 \
    model.mm_cfg.vision_encoder.freeze=True \
    model.mm_cfg.mm_mlp_adapter_type=mlp2x_gelu \
    model.data.dataloader_type=modality \
    model.data.sep_image_conv_front=True \
    model.mega_batch_mult=50 \
    model.activations_checkpoint_granularity=full \
    model.activations_checkpoint_method=uniform \
    model.activations_checkpoint_num_layers=1 \
    exp_manager.name=llava_vicuna_7b_clip_finetune_tp4_pp1
```

# FSDP
FSDP通过以下的参数控制，在config.yaml里修改以使用
```
  # FSDP
  fsdp: True # Enable training with torch FSDP.
  fsdp_sharding_strategy: 'full' # Method to shard model states. Available options are 'full', 'hybrid', and 'grad'.
  fsdp_grad_reduce_dtype: '32' # Gradient reduction data type.
  fsdp_sharded_checkpoint: True # Store and load FSDP shared checkpoint.
  fsdp_use_orig_params: True # Set to True to use FSDP for specific peft scheme.
```

# TP和PP

在训练时，可以指定`model.tensor_model_parallel_size`和`model.tensor_model_parallel_size`参数来改变TP和PP的设置。目前PP的策略是vision_encoder和word_embedding放在rank0，Transformer blocks均分到各个rank（`megatron/core/transformer/transformer_block.py`的33行）。

使用TP或PP时，框架会自动将`vicuna-7b-v1.5-mcore.nemo`按需要切分，无需手动转换模型权重。但Pretrain阶段和SFT阶段的TP和PP需要一致，否则可能需要转换权重。

# Inference

下面是用nemo格式inference的示例，但一般会将nemo格式转为huggingface格式后做inference和evaluation。

```
export NEMO_CONFIG=llava_7b_inference

python examples/multimodal/multimodal_llm/neva/neva_inference.py \
    inference.images_base_path=llava_test_examples/images \
    neva_model_file=nemo_experiments/llava_vicuna_7b_clip_finetune_tp4_pp1/checkpoints/llava_vicuna_7b_clip_finetune_tp4_pp1.nemo \
    prompt_file=llava_test_examples/input_prompts.jsonl \
    output_file=llava_test_examples/results.jsonl
```

# Model Convert

## From .nemo to huggingface

```
python scripts/checkpoint_converters/convert_llava_nemo_to_hf.py \
    --input_name_or_path checkpoints/llava_vicuna_7b_clip_finetune_tp4_pp1.nemo \
    --output_path /tmp \
    --input_tokenizer checkpoints/llama2_tokenizer \
    --hf_output_path checkpoints/llava_vicuna_7b_clip_finetune_hf \
    --processor_type clip \
    --vision_tower_type clip \
    --vision_encoder_path checkpoints/clip-vit-large-patch14-336

# 如果微调了vision_encoder，去掉vision_encoder_path参数
```

# Evaluation

使用VLMEvalKit进行评估。

1. 将VLMEvalKit下`vlmeval/config.py`中的`llava_v1.5_7b`的路径指定为`checkpoints/llava_vicuna_7b_clip_finetune_hf`。
   
2. 执行Evaluation。

```
torchrun --nproc-per-node=8 --master-port 29800 VLMEvalKit/run.py --work-dir ./results/nemo_vicuna_clip --verbose --model llava_v1.5_7b --data MMBench_DEV_EN MMBench_DEV_CN MMMU_DEV_VAL ScienceQA_VAL ScienceQA_TEST CCBench SEEDBench_IMG MME HallusionBench MMStar TextVQA_VAL AI2D_TEST ChartQA_TEST DocVQA_VAL RealWorldQA 
```
