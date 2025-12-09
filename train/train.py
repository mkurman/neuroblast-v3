import sys
import os

sys.path.append("../")

from transformers import Trainer, TrainingArguments, AutoTokenizer
from transformers.data import DataCollatorForLanguageModeling
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset, load_from_disk
from argparse import ArgumentParser
from glob import glob
from datetime import datetime
from datasets import interleave_datasets, concatenate_datasets
from typing import Union, Optional
import random
import torch
from accelerate.logging import get_logger
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from transformers.trainer_pt_utils import get_parameter_names
from trl import SFTTrainer, SFTConfig
from model.pytorch.modeling_neuroblast import NeuroBLASTForCausalLM, NeuroBLASTConfig
import warnings
import wandb

from collator import TokenizationDataCollator
from accelerate import Accelerator

warnings.filterwarnings("ignore")


def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_parameters_count(model) -> tuple[int, int]:
    total_numel = 0
    trainable_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()

            if p.requires_grad:
                trainable_numel += p.numel()

    return total_numel, trainable_numel


def tokenize_function(examples, tokenizer, max_length: int = 512):
    return tokenizer(
        examples["text"], max_length=max_length, padding="max_length", truncation=True
    )


def filter_item(example, length: int = 100):
    return len(example["text"].strip()) > length


def filter_synth(x):
    return "query" in x and "synthetic_reasoning" in x and "synthetic_answer" in x


def map_synth(x):
    return {
        "text": "<|im_start|>user\n"
        + x["query"]
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + "\n<think>\n"
        + x["synthetic_reasoning"]
        + "\n</think>\n\n"
        + x["synthetic_answer"]
        + "<|im_end|>",
    }

def map_synth_chat(x):
    return {"messages": [
        {
            "role": "user", 
            "content": x["query"]
        },
        {
            "role": "assistant",
            "content": "<think>\n"
            + x["synthetic_reasoning"]
            + "\n</think>\n\n"
            + x["synthetic_answer"]
        }
    ]}

def train(
    model_name: str,
    checkpoint: str,
    trainer_checkpoint: str,
    tokenizer_name: str,
    dataset_train: str,
    dataset_validation: Optional[str],
    max_length: int = 512,
    batch_size: int = 32,
    accumulation_iter: int = 128,
    epochs: int = 1,
    lr: float = 5e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    logging_dir: str = "./logs",
    output_dir: str = "./results",
    save_steps: int = 500,
    train_test_split: Union[int, float] = 0.1,
    num_layers: int = 24,
    num_heads: int = 24,
    hidden_size: int = 1024,
    hidden_state_scaler: int = 1,
    skip: int = 0,
    take: Optional[int] = None,
    seed: int = 42,
    device: str = "gpu",
    max_samples: Optional[int] = None,
    **kwargs,
):
    print(f"Training {model_name} on {dataset_train} with {tokenizer_name} tokenizer.")

    set_all_seeds(seed)
    logger = get_logger(__name__)

    print(f"Loading tokenizer {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    is_masked = False

    accelerator = Accelerator()

    if checkpoint is None:
        print(f"Creating {model_name}")
    else:
        print(f"Loading {model_name}")
   
    if accelerator.is_main_process:
        wandb.init(
            project="neuroblast_v3",
            name=f"{model_name}_{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}",
        )

        if checkpoint is None:
            config = NeuroBLASTConfig(
                vocab_size=len(tokenizer),
                hidden_size=512,
                intermediate_size=3072,
                num_associative_layers=32,
                num_sensory_layers=24,
                num_motor_layers=16,
                num_hidden_layers=72,
                num_attention_heads=16,
                num_key_value_heads=8,
                head_dim=128,
                max_position_embeddings=32768,
                pad_token_id=tokenizer.pad_token_id,
                tie_word_embeddings=True,
            )

            model = NeuroBLASTForCausalLM(config)
            model.save_pretrained(os.path.join(output_dir, "base"))
        else:
            model = NeuroBLASTForCausalLM.from_pretrained(checkpoint)

        parameters, trainable_parameters = get_parameters_count(model)
        print(
            "Model parameters:",
            parameters,
            "Trainable parameters:",
            trainable_parameters,
        )
        print(model)

    accelerator.wait_for_everyone()

    print("Loading model")
    if checkpoint is None:
        model = NeuroBLASTForCausalLM.from_pretrained(os.path.join(output_dir, "base"))
    else:
        model = NeuroBLASTForCausalLM.from_pretrained(checkpoint)

    accelerator.wait_for_everyone()

    print("Model loaded.")

    synth = load_dataset(
        "PleIAs/SYNTH",
        streaming=True,
        split="train",
        data_files=["synth_*.parquet"],
        cache_dir="hf_cache",
    )

    datasets = [
        synth,
    ]

    dataset = (
        concatenate_datasets(datasets)
        .shuffle(seed=seed)
        .filter(filter_synth)
        .map(map_synth_chat)
    )

    collator = TokenizationDataCollator(
        tokenizer=tokenizer, pad_to_multiple_of=max_length
    )

    print(f"Batch size {batch_size}")

    if trainer_checkpoint is None:
        output_dir = os.path.join(
            os.getcwd(),
            output_dir,
            model_name,
            datetime.now().strftime("%Y-%m-%d %H_%M_%S"),
        )
    else:
        output_dir = "/".join(trainer_checkpoint.split("/")[:-1])

    train_samples = 100000000 if max_samples is None else max_samples

    num_devices = accelerator.num_processes

    max_steps = train_samples // (batch_size * accumulation_iter * num_devices) * epochs

    print(f"Max steps {max_steps}")
    training_args = TrainingArguments(
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        save_strategy="steps",
        save_steps=save_steps,
        disable_tqdm=False,
        push_to_hub=False,
        logging_strategy="steps",
        logging_dir=os.path.join(
            os.getcwd(),
            logging_dir,
            model_name,
            datetime.now().strftime("%Y-%m-%d %H_%M_%S"),
        ),
        logging_steps=1,
        logging_nan_inf_filter=True,
        gradient_accumulation_steps=accumulation_iter,
        output_dir=output_dir,
        max_steps=max_steps,
        eval_strategy="steps",
        eval_steps=save_steps,
        eval_accumulation_steps=accumulation_iter,
        seed=seed,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=False,
        fp16=False,
        fp16_full_eval=False,
        save_total_limit=1,
        save_on_each_node=False,
        save_only_model=False,
        optim="adamw_torch",
        include_num_input_tokens_seen=True,
        use_cpu=True if device == "cpu" else False,
        use_mps_device=False,
        warmup_ratio=0.0001,
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs=dict(num_decay_steps=20000),
        max_grad_norm=1.0,
        save_safetensors=True,
        report_to="wandb" if accelerator.is_main_process else None,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

    train_ds = dataset.skip(100)
    test_ds = dataset.take(100)

    print("Training model.")
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=trainer_checkpoint)

    print("Training complete.")
    trainer.save_model(output_dir)


def main():
    print("Training model.")
    train(
        model_name="neuroblast3",
        checkpoint=None,
        trainer_checkpoint=None,
        dataset_train=None,
        dataset_validation=None,
        max_length=1280,
        tokenizer_name="PleIAs/Baguettotron",
        batch_size=2,
        accumulation_iter=32,
        epochs=1,
        lr=4e-3,
        weight_decay=0.0,
        logging_dir="logs",
        output_dir="results",
        save_steps=10000,
        seed=3407,
        max_samples=100000000,
        device="gpu",
    )


if __name__ == "__main__":
    main()
