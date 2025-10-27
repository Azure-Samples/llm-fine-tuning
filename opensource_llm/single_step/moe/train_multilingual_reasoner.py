"""
Fine-tuning script for multilingual reasoning with openai/gpt-oss-20b
Based on the fine-tune-transformers.ipynb notebook
Adapted for Azure ML job execution
"""

import os
import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mxfp4Config,
    set_seed,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune multilingual reasoning model")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="openai/gpt-oss-20b",
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Use MXFP4 quantization"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceH4/Multilingual-Thinking",
        help="The dataset to use for training"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="The dataset split to use"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--use_peft_lora",
        action="store_true",
        default=True,
        help="Use PEFT LoRA for training"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="Rank of the LoRA update matrices"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Scaling factor for LoRA"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="Dropout probability for LoRA layers (must be 0.0 when using target_parameters for MoE)"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="all-linear",
        help="Target modules for LoRA"
    )
    parser.add_argument(
        "--lora_target_parameters",
        type=str,
        default="7.mlp.experts.gate_up_proj,7.mlp.experts.down_proj,15.mlp.experts.gate_up_proj,15.mlp.experts.down_proj,23.mlp.experts.gate_up_proj,23.mlp.experts.down_proj",
        help="Comma-separated list of target parameters for MoE layers"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for model and checkpoints"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device during training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Ratio of warmup steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every X updates steps"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        help="Save checkpoint strategy"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Reporting integration (tensorboard, wandb, mlflow, etc.)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
    )
    
    return parser.parse_args()


def create_model_and_tokenizer(args):
    """Create model and tokenizer."""
    # Handle Azure ML registry model path
    model_path = args.model_name_or_path
    if os.path.exists(model_path):
        # Azure ML registry models need to load from data/model subdirectory

        model_path = os.path.join(model_path, "model_artifact", "model")
        print("content inside model_path with model_artifact:", os.listdir(model_path))
        print(f"Azure ML registry model detected, using path: {model_path}")
    
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model from {model_path}...")
    
    # Detect if we're in distributed mode
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    
    # Configure model kwargs
    model_kwargs = {
        "attn_implementation": "eager",
        "torch_dtype": torch.bfloat16,
        "use_cache": False,
    }
    
    # Only use device_map='auto' in non-distributed mode
    # In distributed mode, let the trainer/accelerator handle device placement
    if not is_distributed:
        model_kwargs["device_map"] = "auto"
        print("Single GPU mode: using device_map='auto'")
    else:
        print(f"Distributed mode detected (world_size={world_size}): device placement handled by accelerator")
    
    # Add quantization config if specified
    if args.use_quantization:
        print("Using MXFP4 quantization...")
        quantization_config = Mxfp4Config(dequantize=True)
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    return model, tokenizer


def create_dataset(args):
    """Load and return the dataset."""
    print(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    print(f"Dataset loaded with {len(dataset)} examples")
    return dataset


def create_peft_config(args):
    """Create PEFT LoRA configuration."""
    if not args.use_peft_lora:
        return None
    
    print("Creating LoRA configuration...")
    
    # Parse target parameters for MoE layers
    target_parameters = None
    if args.lora_target_parameters:
        target_parameters = [p.strip() for p in args.lora_target_parameters.split(",")]
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        target_parameters=target_parameters,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    return peft_config


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Set random seed to {args.seed}")
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(args)
    
    # Load dataset
    dataset = create_dataset(args)
    
    # Create LoRA config (do NOT wrap model manually when using FSDP)
    peft_config = create_peft_config(args)
    if peft_config is not None:
        print("LoRA configuration created - will be applied by Trainer/Accelerate")
    
    # Create training configuration
    print("Creating training configuration...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Non-reentrant for DDP+LoRA
        bf16=args.bf16,
        max_length=args.max_seq_length,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        report_to=args.report_to,
        ddp_find_unused_parameters=False,  # All parameters are used
    )
    
    # Create trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # Enable DDP static graph mode for gradient checkpointing with LoRA
    # This prevents "parameter ready twice" errors when using checkpointing with MoE experts
    if hasattr(trainer.model, 'module') and hasattr(trainer.model, '_set_static_graph'):
        print("Enabling DDP static graph mode for gradient checkpointing compatibility...")
        trainer.model._set_static_graph()
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
