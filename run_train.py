"""
Nanotron training script example using a custom dataloader.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=2 examples/custom-dataloader/run_train.py --config-file examples/custom-dataloader/config_custom_dl.yaml
```
"""
import argparse
import os
import pickle
from typing import Dict, cast

import datasets
import numpy as np
from nanotron import logging
from nanotron.config import (
    DataArgs,
    DatasetStageArgs,
    PretrainDatasetsArgs,
)
from nanotron.data.clm_collator import DataCollatorForCLMWithPositionIds
from nanotron.data.dataloader import (
    get_dataloader_worker_init,
    get_train_dataloader,
)
from nanotron.data.processing import (
    clm_process,
    get_datasets,
)
from nanotron.helpers import (
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from nanotron.utils import main_rank_first
from torch.utils.data import DataLoader

from datasets import Dataset

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)

# --- Custom Data Loading Helper Functions ---

# Utility function to read entire files as single documents
def read_whole_file(file_path):
    """Reads an entire file and returns its content as a single string."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    except Exception as e:
        log_rank(f"Error reading file {file_path}: {e}", logger=logger, level=logging.ERROR, rank=0)
        return "" # Return empty string on error

def load_metadata(split_state_path, data_folder):
    """Loads file paths from the split state pickle file."""
    try:
        with open(split_state_path, "rb") as f:
            split_state = pickle.load(f)
        train_files_info = split_state[0]
        # eval_files_info = split_state[1] # Not needed for training dataloader

        # Convert relative paths to absolute paths
        train_file_paths = [os.path.join(data_folder, f['file_path']) for f in train_files_info]
        # eval_file_paths = [os.path.join(data_folder, f['file_path']) for f in eval_files_info]
        return train_file_paths #, eval_file_paths
    except Exception as e:
        log_rank(f"Error loading split state from {split_state_path}: {e}", logger=logger, level=logging.ERROR, rank=0)
        return [] #, []

def create_hf_dataset(file_paths):
    """Loads texts from file paths and creates a Hugging Face Dataset."""
    texts = [read_whole_file(fp) for fp in file_paths if fp] # Read non-empty paths
    # Filter out potential None values if read_whole_file failed silently before
    valid_texts = [text for text in texts if text is not None and text != ""]
    if not valid_texts:
        log_rank("Warning: No valid text data found for dataset creation.", logger=logger, level=logging.WARNING, rank=0)
        # Return an empty dataset with the expected structure
        return Dataset.from_dict({"text": []})
    return Dataset.from_dict({"text": valid_texts})

# --- End Custom Data Loading Helper Functions ---


def get_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    consumed_train_samples: int,
    num_remaining_train_steps: int,
):
    """
    Returns a dataloader for a given data stage.

    data: The data configuration for the current stage.
    consumed_train_samples: The number of samples consumed by the model in the this stage (each stage starts from zero).
    num_remaining_train_steps: The number of remaining training steps for this stage.
    """
    assert consumed_train_samples >= 0, "consumed_train_samples should be greater than 0"
    assert num_remaining_train_steps >= 0, "num_remaining_train_steps should be greater than 0"

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Case 1: custom data generator
    if data.dataset is None:
        log_rank("Using custom data generator", logger=logger, level=logging.INFO, rank=0)

        ###########################################################################################################
        # This can be replaced with your own tokenized data generator
        ###########################################################################################################
        data_folder = "./extracted_climate_text_dataset"
        split_state_path = "./split_state.pkl"
        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path # Use tokenizer from main config

        log_rank(f"Loading metadata from {split_state_path} relative to {data_folder}", logger=logger, level=logging.INFO, rank=0)
        train_file_paths = load_metadata(split_state_path, data_folder)

        if not train_file_paths:
            log_rank("ERROR: Failed to load train file paths. Exiting.", logger=logger, level=logging.ERROR, rank=0)
            # In a distributed setting, a more robust error handling/exit mechanism might be needed
            # For now, returning an empty dataloader might be better than sys.exit
            # sys.exit(1) # Avoid hard exit in library code
            raise RuntimeError("Failed to load training data file paths.")
        
        log_rank(f"Creating raw Hugging Face dataset from {len(train_file_paths)} files...", logger=logger, level=logging.INFO, rank=0)

        # Create dataset (datasets library often handles distributed loading/caching)
        raw_train_dataset = create_hf_dataset(train_file_paths)
        log_rank(f"Raw dataset created. Size: {len(raw_train_dataset)}", logger=logger, level=logging.INFO, rank=0)

        if len(raw_train_dataset) == 0:
            raise RuntimeError("Created training dataset is empty.")

        # --- Tokenizer ---
        log_rank(f"Loading tokenizer from {tokenizer_path}", logger=logger, level=logging.INFO, rank=0)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            log_rank("Warning: Tokenizer has no pad token. Setting pad_token = eos_token.", logger=logger, level=logging.WARNING, rank=0)
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Ensure padding is on the left side for CLM
        sequence_sep_tokens = [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token, tokenizer.unk_token]

        # Using trainer.sequence_length for max length
        log_rank(f"Tokenizer sequence length: {trainer.sequence_length}", logger=logger, level=logging.INFO, rank=0)

        train_dataset = clm_process(
            raw_dataset=raw_train_dataset,
            tokenizer=tokenizer,
            text_column_name="text",
            dataset_processing_num_proc_per_process=1,
            dataset_overwrite_cache=False,
            sequence_length=trainer.sequence_length,
        )

        # Log final size
        log_rank(f"Column selection complete. Final Train size: {len(train_dataset)}", logger=logger, level=logging.INFO, rank=0)

        if len(train_dataset) == 0:
            raise RuntimeError("Training dataset is empty after tokenization and chunking.")
        ###########################################################################################################

        # data_collator = DataCollatorForCLMWithPositionIds(
        #     sequence_length=trainer.sequence_length,
        #     input_pp_rank=input_pp_rank,
        #     output_pp_rank=output_pp_rank,
        #     parallel_context=trainer.parallel_context,
        # )

        return get_train_dataloader(
            train_dataset=train_dataset,
            sequence_length=trainer.sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=trainer.micro_batch_size,
            consumed_train_samples=consumed_train_samples,
            dataloader_num_workers=data.num_loading_workers,
            seed_worker=data.seed,
            dataloader_drop_last=True,
            use_position_ids=True,
            sequence_sep_tokens=sequence_sep_tokens,  # Used to generate position ids
        )

    # Case 2: HuggingFace datasets
    elif isinstance(data.dataset, PretrainDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)
        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # We need to the 1st device to process dataset and cache it, then other devices load from cache
        with main_rank_first(trainer.parallel_context.world_pg):
            # We load the raw dataset
            raw_dataset = get_datasets(
                hf_dataset_or_datasets=data.dataset.hf_dataset_or_datasets,
                hf_dataset_config_name=data.dataset.hf_dataset_config_name,
                splits=data.dataset.hf_dataset_splits,
            )["train"]

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            # We apply the Causal Language Modeling preprocessing
            train_dataset = clm_process(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                text_column_name=data.dataset.text_column_name,
                dataset_processing_num_proc_per_process=data.dataset.dataset_processing_num_proc_per_process,
                dataset_overwrite_cache=data.dataset.dataset_overwrite_cache,
                sequence_length=trainer.sequence_length,
            )

            # We load the processed dataset on the ranks requiring it
            dataloader = get_train_dataloader(
                train_dataset=train_dataset,
                sequence_length=trainer.sequence_length,
                parallel_context=trainer.parallel_context,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=consumed_train_samples,
                dataloader_num_workers=data.num_loading_workers,
                seed_worker=data.seed,
                dataloader_drop_last=True,
            )

            # Check if we have enough samples for train_steps
            total_tokens_dataset = len(dataloader.dataset) * trainer.sequence_length
            num_tokens_needed_for_training = (
                num_remaining_train_steps * trainer.global_batch_size * trainer.sequence_length
            )
            assert num_tokens_needed_for_training <= total_tokens_dataset, (
                f"Dataset is too small for steps ({total_tokens_dataset} < {num_tokens_needed_for_training}), "
                f"Try train_steps<={len(dataloader.dataset) // trainer.global_batch_size + trainer.iteration_step}"
            )
    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}")

    return dataloader


def get_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    dataloaders = {}

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        consumed_train_samples, _ = get_consumed_train_samples_of_a_data_stage_from_ckp(stage, trainer.metadata)
        assert (
            consumed_train_samples is not None
        ), f"Cannot find consumed_train_samples for stage {stage.start_training_step} in the checkpoint"

        num_remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
            stage, trainer.config, trainer.metadata
        )
        log_rank(
            f"[Training Plan] Stage {stage.name} has {num_remaining_train_steps} remaining training steps and has consumed {consumed_train_samples} samples",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        dataloader = (
            get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
            if stage_idx == 0
            else lambda stage=stage: get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
        )
        dataloaders[stage.name] = dataloader
    return dataloaders


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)

