<h1 align="center">⚡️ Nanotron</h1>

## Installation

To run the code in this project, first create a Conda environment using the `environment.yml` file by installing all dependencies listed there:

```
A list of the original Nanotron installation guide packages:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install datasets transformers datatrove[io] numba wandb
pip install ninja triton "flash-attn>=2.5.0" --no-build-isolation
```

```
Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```



## Quick Start
In `config_resume_training.yaml` replace the `tokenizer_name_or_path` with your original llama 3.2 3B folder path AND replace your `resume_checkpoint_path` with your converted llama model folder using the `examples/llama/convert_hf_to_nanotron.py` script.

The following command will train the llama model on a single node of 2 x A100's:

```shell
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 run_train.py --config-file config_resume_training.yaml
```

The model will be saved in the `checkpoints` directory as specified in the config file.

```
Set the config_resume_training.yaml configurations to the following:

Data parallelism:
-train_steps: 213
-dp: 2, tp: 1, pp: 1

Tensor parallelism:
-train_steps: 426
-dp: 1, tp: 2, pp: 1

Pipeline parallelism:
-train_steps: 426
-dp: 1, tp: 1, pp: 2
```


