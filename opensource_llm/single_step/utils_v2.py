import os
from enum import Enum

import packaging.version
import torch
import transformers
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Mxfp4Config
)


from peft import LoraConfig


DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    def preprocess(samples):
        batch = []
        for conversation in samples["messages"]:
            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        return {"content": batch}

    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        try:
            # Try first if dataset on a Hub repo
            dataset = load_dataset(data_args.dataset_name, split=split)
        except DatasetGenerationError:
            # If not, check local dataset
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split))

        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data


def create_and_prepare_model(args, data_args, training_args):
    
    # Detect if this is a GPT-OSS model
    is_gpt_oss = "gpt-oss" in args.model_name_or_path.lower()
    model_family = "gpt-oss" if is_gpt_oss else "qwen" if "qwen" in args.model_name_or_path.lower() else "other"
    
    print(f"Detected model family: {model_family}")
    print(f"Is GPT-OSS: {is_gpt_oss}")
    
    # Check if the model path is already pointing to a model directory
    # or if we need to append the standard Azure ML model structure
    original_path = args.model_name_or_path
    
    # Check if it's a local path (not a HuggingFace Hub model)
    if os.path.sep in args.model_name_or_path or os.path.exists(args.model_name_or_path):
        # This appears to be a local path, check Azure ML structure
        if os.path.exists(os.path.join(args.model_name_or_path, "data", "model")):
            args.model_name_or_path = os.path.join(args.model_name_or_path, "data", "model")
            print(f"Using Azure ML data/model structure: {args.model_name_or_path}")
        elif not (os.path.exists(os.path.join(args.model_name_or_path, "config.json")) or 
                  os.path.exists(os.path.join(args.model_name_or_path, "pytorch_model.bin")) or
                  os.path.exists(os.path.join(args.model_name_or_path, "model.safetensors"))):
            # Try model_artifact structure
            artifact_path = os.path.join(args.model_name_or_path, "model_artifact", "model")
            if os.path.exists(artifact_path):
                args.model_name_or_path = artifact_path
                print(f"Using Azure ML model_artifact structure: {args.model_name_or_path}")
            else:
                # Fallback to original path for HuggingFace Hub models
                args.model_name_or_path = original_path
                print(f"Fallback to original path for HuggingFace Hub model: {args.model_name_or_path}")
        else:
            print(f"Using direct model path: {args.model_name_or_path}")
    else:
        # This looks like a HuggingFace Hub model (e.g., "Qwen/Qwen3-30B-A3B")
        print(f"Using HuggingFace Hub model: {args.model_name_or_path}")

    # Set dtype based on model family
    if is_gpt_oss:
        torch_dtype = torch.bfloat16  # Always use bfloat16 for GPT-OSS as recommended
        print("torch_dtype for GPT-OSS:", torch_dtype)
    else:
        # For Qwen and other models, use the specified dtype or default to bfloat16
        torch_dtype = torch.bfloat16
        print("torch_dtype for Qwen/other models:", torch_dtype)
    
    # For GPT-OSS models, use Mxfp4Config as recommended by OpenAI
    # Don't mix BitsAndBytesConfig with Mxfp4Config as they conflict
    bnb_config = None
    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if is_gpt_oss:
        # For GPT-OSS models, use Mxfp4Config as recommended by OpenAI
        quantization_config = Mxfp4Config(dequantize=True)
        print("Using Mxfp4Config for GPT-OSS model")
    else:
        # For Qwen and other models, use BitsAndBytesConfig
        quantization_config = bnb_config
        print("Using BitsAndBytesConfig for Qwen/other model")
    print("quantization_config:", quantization_config)

    # Set attention implementation based on model family and user preference
    if is_gpt_oss:
        attn_implementation = "eager"  # Use eager for GPT-OSS as recommended
        print("Using eager attention for GPT-OSS model")
    else:
        # For Qwen models, allow Flash Attention if requested and available
        if args.use_flash_attn:
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                print("Using Flash Attention 2 for Qwen model")
            except ImportError:
                print("Flash Attention not available, falling back to eager")
                attn_implementation = "eager"
        else:
            attn_implementation = "eager"
            print("Using eager attention for Qwen model")

    model_kwargs = {
        "quantization_config": quantization_config,
        "attn_implementation": attn_implementation,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,  # Reduce CPU memory usage during loading
        "device_map": None,  # Let FSDP handle device placement
    }

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    )

    # Configure LoRA based on model family
    peft_config = None
    if args.use_peft_lora:
        if is_gpt_oss:
            # For GPT-OSS models, use expert targeting if available
            print("Configuring LoRA for GPT-OSS model with expert targeting")
            target_modules = args.lora_target_modules.split(",") if args.lora_target_modules != "all-linear" else args.lora_target_modules
            
            lora_config_args = {
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "r": args.lora_r,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": target_modules,
            }
            
            # Add target_parameters if specified (for expert targeting)
            if hasattr(args, 'lora_target_parameters') and args.lora_target_parameters:
                target_parameters = [param.strip() for param in args.lora_target_parameters.split(",")]
                lora_config_args["target_parameters"] = target_parameters
                print(f"Using expert targeting with target_parameters: {target_parameters}")
            
        else:
            # For Qwen and other models, use standard target modules
            print("Configuring LoRA for Qwen/other model with standard target modules")
            
            # Standard Qwen LoRA target modules
            if args.lora_target_modules == "all-linear":
                target_modules = "all-linear"
            else:
                # Common Qwen model modules - adjust based on actual architecture
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                print(f"Using standard Qwen target modules: {target_modules}")
            
            lora_config_args = {
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "r": args.lora_r,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": target_modules,
            }
        
        peft_config = LoraConfig(**lora_config_args)
        print(f"LoRA config created for {model_family} model")

    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template

        # make embedding resizing configurable?
        # For GPT-OSS models, we use standard embedding resizing
        uses_transformers_4_46 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.46.0")
        uses_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "none").lower() == "true"
        print("uses_fsdp:", uses_fsdp)
        
        # Use mean_resizing=False with FSDP for compatibility
        if uses_fsdp and uses_transformers_4_46:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
        else:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    # Apply PEFT configuration to the model if using LoRA
    if args.use_peft_lora:
        from peft import get_peft_model
        model = get_peft_model(model, peft_config)
        peft_config = None  # Set to None since model is now already a PEFT model

    return model, peft_config, tokenizer