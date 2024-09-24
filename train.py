import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import  TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
        set_seed,

)

from trl import (
   SFTTrainer,
)
from data_collator import CustomDualMaskCollator as DataCollatorForCompletionOnlyLM

@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=4000, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )

def formatting_prompts_func(example):
    return example["text"] # Our dataset is already formatted with chat template


def training_function(script_args, training_args):
    
    # Dataset
    train_dataset  = load_dataset("json", data_files=script_args.dataset_path, split="train")

    response_template = "<reflection>" # train from the reflection token

    # Model & Tokenizer

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    special_tokens_dict = dict()
    special_tokens_dict["additional_special_tokens"] = ["<thinking>","</thinking>","<output>","</output>","<reflection>","</reflection>"]
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    # Model
    torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        attn_implementation="sdpa", # use sdpa, alternatively use "flash_attention_2"
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )
    model.resize_token_embeddings(len(tokenizer))

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    
    # Training
    
    trainer = SFTTrainer(
        model=script_args.model_id,
        args=training_args,
        train_dataset=train_dataset,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        dataset_kwargs={
            "add_special_tokens": True,  # We have added special tokens
        },
    )

    ##########
    # Train model
    ##########
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ##########
    # SAVE MODEL
    ##########
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)
