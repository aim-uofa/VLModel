Llama 3 LoRA Fine-Tuning and Deployment with NeMo Framework and NVIDIA NIM
==========================================================================

`Llama 3 <https://blogs.nvidia.com/blog/meta-llama3-inference-acceleration/>`_ is an open source large language model by Meta that delivers state-of-the-art performance on popular industry benchmarks. It has been pretrained on over 15 trillion tokens, and supports an 8K token context length. It is available in two sizes, 8B and 70B, and each size has two variants—base pretrained and instruction tuned.

`Low-Rank Adaptation (LoRA) <https://arxiv.org/pdf/2106.09685>`__ has emerged as a popular Parameter Efficient Fine-Tuning (PEFT) technique that tunes a very small number of additional parameters as compared to full fine-tuning, thereby reducing the compute required.

`NVIDIA NeMo
Framework <https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html>`__ provides tools to perform LoRA on Llama 3 to fit your use case, which can then be deployed using `NVIDIA NIM <https://www.nvidia.com/en-us/ai/>`__ for optimized inference on NVIDIA GPUs.

.. figure:: ./img/e2e-lora-train-and-deploy.png
  :width: 1000
  :alt: Diagram showing the steps for LoRA customization using the NVIDIA NeMo Framework and deployment with NVIDIA NIM. The steps include converting the base model to .nemo format, creating LoRA adapters with NeMo, and then depoying the LoRA adapter with NIM for inference.
  :align: center

  Figure 1: Steps for LoRA customization using the NVIDIA NeMo Framework and deployment with NVIDIA NIM


| NIM supports seamless deployment of multiple LoRA adapters (aka “multi-LoRA”) over the same base model by dynamically loading the adapter weights based on incoming requests at runtime. This provides the flexibility to handle inputs from various tasks or use cases without the need for deploying a unique model for each individual use case. More information on NIM for LLMs can be found it its `documentation <https://docs.nvidia.com/nim/large-language-models latest/introduction.html>`__.

Requirements
-------------

In order to proceed, ensure that you have met the following requirements:

* System Configuration
    * Access to at least 1 NVIDIA GPU with a cumulative memory of at least 80GB, for example: 1 x H100-80GB or 1 x A100-80GB.
    * A Docker-enabled environment, with `NVIDIA Container Runtime <https://developer.nvidia.com/container-runtime>`_ installed, which will make the container GPU-aware.
    * `Additional NIM requirements <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#prerequisites>`_.

* Requested the necessary permission from Hugging Face and Meta to download `Meta-Llama-3-8B-Instruct <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>`_. Then, you can use your Hugging Face `access token <https://huggingface.co/docs/hub/en/security-tokens>`_ to download the model, which we will then convert and customize with NeMo Framework.

* `Authenticate with NVIDIA NGC <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#ngc-authentication>`_, and download `NGC CLI Tool <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#ngc-cli-tool>`_.


`Create a LoRA Adapter with NeMo Framework <./llama3-lora-nemofw.ipynb>`__
--------------------------------------------------------------------------

This notebook shows how to perform LoRA PEFT on **Llama 3 8B Instruct** using `PubMedQA <https://pubmedqa.github.io/>`__ with NeMo Framework. PubMedQA is a Question-Answering dataset for biomedical texts. You will use the NeMo Framework which is available as a `docker container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>`__.

To get started
^^^^^^^^^^^^^^

1. Run the container using the following command. It assumes that you have the notebook(s) available in the current working directory. If not, mount the appropriate folder to ``/workspace``.

.. code:: bash

   export FW_VERSION=24.05  # Make sure to choose the latest available tag


.. code:: bash

   docker run \
     --gpus all \
     --shm-size=2g \
     --net=host \
     --ulimit memlock=-1 \
     --rm -it \
     -v ${PWD}:/workspace \
     -w /workspace \
     -v ${PWD}/results:/results \
     nvcr.io/nvidia/nemo:$FW_VERSION bash

2. From within the container, start the Jupyter lab:

.. code:: bash

   jupyter lab --ip 0.0.0.0 --port=8888 --allow-root

3. Then, navigate to `this notebook <./llama3-lora-nemofw.ipynb>`__.


`Deploy Multiple LoRA Inference Adapters with NVIDIA NIM <./llama3-lora-deploy-nim.ipynb>`__
--------------------------------------------------------------------------------------------

This procedure demonstrates how to deploy multiple LoRA adapters with NVIDIA NIM. NIM supports LoRA adapters in ``.nemo`` (from NeMo Framework), and Hugging Face model formats. You will deploy the PubMedQA LoRA adapter from the first notebook, alongside two previously trained LoRA adapters (`GSM8K <https://github.com/openai/grade-school-math>`__, `SQuAD <https://rajpurkar.github.io/SQuAD-explorer/>`__) that are available on NVIDIA NGC as examples.

``NOTE``: Although it’s not mandatory to finish the LoRA training and secure the adapter from the preceding notebook (“Creating a LoRA adapter with NeMo Framework”) to proceed with this one, it is advisable. Regardless, you can continue to learn about LoRA deployment with NIM using other adapters that you’ve downloaded from NVIDIA NGC.


1. Download the example LoRA adapters.

The following steps assume that you have authenticated with NGC and downloaded the CLI tool, as listed in the Requirements section.

.. code:: bash

   # Set path to your LoRA model store
   export LOCAL_PEFT_DIRECTORY="$(pwd)/loras"


.. code:: bash

   mkdir -p $LOCAL_PEFT_DIRECTORY
   pushd $LOCAL_PEFT_DIRECTORY

   # downloading NeMo-format loras
   ngc registry model download-version "nim/meta/llama3-8b-instruct-lora:nemo-math-v1"
   ngc registry model download-version "nim/meta/llama3-8b-instruct-lora:nemo-squad-v1"

   popd
   chmod -R 777 $LOCAL_PEFT_DIRECTORY

2. Prepare the LoRA model store

After training is complete, that LoRA model checkpoint will be
created at
``./results/Meta-Llama-3-8B-Instruct/checkpoints/megatron_gpt_peft_lora_tuning.nemo``,
assuming default paths in the first notebook weren’t modified.

To ensure model store is organized as expected, create a folder named
``llama3-8b-pubmed-qa``, and move your .nemo checkpoint there.

.. code:: bash

   mkdir -p $LOCAL_PEFT_DIRECTORY/llama3-8b-pubmed-qa

   # Ensure the source path is correct
   cp ./results/Meta-Llama-3-8B-Instruct/checkpoints/megatron_gpt_peft_lora_tuning.nemo $LOCAL_PEFT_DIRECTORY/llama3-8b-pubmed-qa



The LoRA model store directory should have a structure like so - with the name of the model as a sub-folder that contains the .nemo file.

::

   <$LOCAL_PEFT_DIRECTORY>
   ├── llama3-8b-instruct-lora_vnemo-math-v1
   │   └── llama3_8b_math.nemo
   ├── llama3-8b-instruct-lora_vnemo-squad-v1
   │   └── llama3_8b_squad.nemo
   └── llama3-8b-pubmed-qa
       └── megatron_gpt_peft_lora_tuning.nemo

The last one was just trained on the PubmedQA dataset in the previous
notebook.


3. Set-up NIM

From your host OS environment, start the NIM docker container while mounting the LoRA model store, as follows:

.. code:: bash

   # Set these configurations
   export NGC_API_KEY=<YOUR_NGC_API_KEY>
   export NIM_PEFT_REFRESH_INTERVAL=3600  # (in seconds) will check NIM_PEFT_SOURCE for newly added models in this interval
   export NIM_CACHE_PATH=</path/to/NIM-model-store-cache>  # Model artifacts (in container) are cached in this directory


.. code:: bash

   mkdir -p $NIM_CACHE_PATH
   chmod -R 777 $NIM_CACHE_PATH

   export NIM_PEFT_SOURCE=/home/nvs/loras # Path to LoRA models internal to the container
   export CONTAINER_NAME=meta-llama3-8b-instruct

   docker run -it --rm --name=$CONTAINER_NAME \
       --runtime=nvidia \
       --gpus all \
       --shm-size=16GB \
       -e NGC_API_KEY \
       -e NIM_PEFT_SOURCE \
       -e NIM_PEFT_REFRESH_INTERVAL \
       -v $NIM_CACHE_PATH:/opt/nim/.cache \
       -v $LOCAL_PEFT_DIRECTORY:$NIM_PEFT_SOURCE \
       -p 8000:8000 \
       nvcr.io/nim/meta/llama3-8b-instruct:1.0.0

The first time you run the command, it will download the model and cache it in ``$NIM_CACHE_PATH`` so subsequent deployments are even faster. There are several options to configure NIM other than the ones listed above. You can find a full list in `NIM configuration <https://docs.nvidia.com/nim/large-language-models/latest/configuration.html>`__ documentation.


4. Start the notebook

From another terminal, follow the same instructions as the previous
notebook to launch Jupyter Lab, and navigate to `this notebook <./llama3-lora-deploy-nim.ipynb>`__.

You can use the same NeMo Framework docker container which already has Jupyter Lab installed.