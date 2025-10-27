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
    
    # Check if the model path is already pointing to a model directory
    # or if we need to append the standard Azure ML model structure
    if os.path.exists(os.path.join(args.model_name_or_path, "data", "model")):
        args.model_name_or_path = os.path.join(args.model_name_or_path, "data", "model")
    elif not (os.path.exists(os.path.join(args.model_name_or_path, "config.json")) or 
              os.path.exists(os.path.join(args.model_name_or_path, "pytorch_model.bin")) or
              os.path.exists(os.path.join(args.model_name_or_path, "model.safetensors"))):
        # If it's not a direct model path and doesn't have the /data/model structure,
        # try to use it as-is (for HuggingFace Hub models or other formats)
        args.model_name_or_path = os.path.join(args.model_name_or_path, "model_artifact", "model")

    # For GPT-OSS models, we use Mxfp4Config instead of BitsAndBytes quantization
    # This is the recommended approach by OpenAI for their GPT-OSS models
    
    torch_dtype = torch.bfloat16  # Always use bfloat16 for GPT-OSS as recommended
    print("torch_dtype:", torch_dtype)
    
    # For GPT-OSS models, use Mxfp4Config as recommended by OpenAI
    # Don't mix BitsAndBytesConfig with Mxfp4Config as they conflict
    quantization_config = Mxfp4Config(dequantize=True)
    print("quantization_config:", quantization_config)

    model_kwargs = {
        "quantization_config": quantization_config,
        "attn_implementation": "eager",  # Use eager for GPT-OSS as recommended
        "torch_dtype": torch_dtype,
        "use_cache": False,  # Set to False for training with gradient checkpointing
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,  # Reduce CPU memory usage during loading
        "device_map": None,  # Let FSDP handle device placement
    }

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    )

    # INSPECT MODEL ARCHITECTURE - Print expert layers for debugging
    print("\n" + "="*80)
    print("INSPECTING GPT-OSS MODEL ARCHITECTURE")
    print("="*80)
    
    expert_layers = []
    mlp_layers = []
    all_layer_names = []
    layer_numbers_with_experts = set()
    
    # Let's do a more comprehensive search for different naming patterns
    for name, module in model.named_modules():
        all_layer_names.append(name)
        
        # Look for different expert patterns
        if "expert" in name.lower():
            expert_layers.append(name)
            print(f"Found expert pattern: {name}")
        
        # Look for MLP patterns (might contain experts)
        if "mlp" in name.lower():
            mlp_layers.append(name)
        
        # Look for gate, up, down projections (common in MoE)
        if any(pattern in name.lower() for pattern in ["gate", "up_proj", "down_proj", "gate_up", "gate_proj"]):
            expert_layers.append(name)
    
    print(f"Total model parameters: {model.num_parameters():,}")
    print(f"Model config key info:")
    print(f"  - num_hidden_layers: {getattr(model.config, 'num_hidden_layers', 'Unknown')}")
    print(f"  - num_local_experts: {getattr(model.config, 'num_local_experts', 'Unknown')}")
    print(f"  - experts_per_token: {getattr(model.config, 'experts_per_token', 'Unknown')}")
    print(f"  - num_experts_per_tok: {getattr(model.config, 'num_experts_per_tok', 'Unknown')}")
    
    print(f"\nDirect expert search results:")
    print(f"  - Expert-named layers found: {len(expert_layers)}")
    print(f"  - MLP layers found: {len(mlp_layers)}")
    
    # Show sample of MLP layers to understand structure
    if mlp_layers:
        print(f"\nMLP layer patterns (first 10):")
        for i, layer_name in enumerate(mlp_layers[:10]):
            print(f"  {i+1}. {layer_name}")
        if len(mlp_layers) > 10:
            print(f"  ... and {len(mlp_layers) - 10} more MLP layers")
    
    # Show all layers with specific patterns for debugging
    interesting_patterns = ["gate", "up", "down", "proj", "expert", "mlp", "router"]
    print(f"\nLayers with interesting patterns:")
    for pattern in interesting_patterns:
        matching_layers = [name for name in all_layer_names if pattern in name.lower()]
        if matching_layers:
            print(f"\n  {pattern.upper()} pattern ({len(matching_layers)} found):")
            for layer in matching_layers[:5]:  # Show first 5
                print(f"    - {layer}")
            if len(matching_layers) > 5:
                print(f"    ... and {len(matching_layers) - 5} more")
    
    # Try to understand the actual model structure by looking at a specific layer
    print(f"\nInspecting layer structure...")
    try:
        # Look at the model's structure more directly
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            sample_layer = model.model.layers[0]
            print(f"Sample layer (layer 0) structure:")
            for name, module in sample_layer.named_modules():
                if name:  # Skip empty names
                    print(f"    - {name}: {type(module).__name__}")
            
            # DETAILED EXPERT INSPECTION - Look inside the experts module
            print(f"\nDETAILED EXPERT MODULE INSPECTION:")
            print(f"=" * 60)
            if hasattr(sample_layer, 'mlp') and hasattr(sample_layer.mlp, 'experts'):
                experts_module = sample_layer.mlp.experts
                print(f"Experts module type: {type(experts_module).__name__}")
                print(f"Experts module structure:")
                for name, module in experts_module.named_modules():
                    if name:  # Skip empty names
                        print(f"    experts.{name}: {type(module).__name__}")
                
                # Check for parameters in experts
                print(f"\nExpert parameters:")
                for name, param in experts_module.named_parameters():
                    print(f"    experts.{name}: {param.shape}")
                
                # Also check if there are individual expert layers
                if hasattr(experts_module, 'experts') and isinstance(experts_module.experts, torch.nn.ModuleList):
                    print(f"\nIndividual expert structure (first expert):")
                    first_expert = experts_module.experts[0]
                    for name, module in first_expert.named_modules():
                        if name:
                            print(f"    expert[0].{name}: {type(module).__name__}")
                
                # Generate correct target_parameters based on actual structure
                print(f"\nCORRECTED TARGET PARAMETERS:")
                print(f"=" * 60)
                actual_expert_params = []
                for name, param in experts_module.named_parameters():
                    if any(pattern in name for pattern in ['gate', 'up', 'down', 'proj']):
                        actual_expert_params.append(f"experts.{name}")
                
                if actual_expert_params:
                    print(f"Found actual expert parameters in layer 0:")
                    for param in actual_expert_params[:5]:  # Show first 5
                        print(f"    - {param}")
                    
                    # Generate target parameters for selected layers
                    target_layers = [6, 12, 18, 24, 30]
                    corrected_target_params = []
                    for layer_num in target_layers:
                        for param_name in actual_expert_params:
                            corrected_target_params.append(f"{layer_num}.mlp.{param_name}")
                    
                    print(f"\nCORRECTED YAML CONFIGURATION:")
                    print(f"--lora_target_modules \"all-linear\"")
                    print(f"--lora_target_parameters \"{','.join(corrected_target_params[:10])}...\"")  # Show first 10
                    print(f"(Total of {len(corrected_target_params)} target parameters)")
                else:
                    print("No specific expert parameters found - may need different approach")
            
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            sample_layer = model.transformer.h[0]
            print(f"Sample layer (layer 0) structure:")
            for name, module in sample_layer.named_modules():
                if name:  # Skip empty names
                    print(f"    - {name}: {type(module).__name__}")
    except Exception as e:
        print(f"Could not inspect layer structure: {e}")
    
    # Based on the official example pattern, suggest using all-linear for now
    print(f"\nRECOMMENDATION:")
    print(f"Since expert layers are not easily identifiable with current search,")
    print(f"let's use the OFFICIAL APPROACH from the working example:")
    print(f"")
    print(f"For GPT-OSS 120B (36 layers), following the official pattern:")
    print(f"--lora_target_modules \"all-linear\"")
    print(f"")
    print(f"Based on the official 20B example that targets layers 7, 15, 23,")
    print(f"for 36-layer model, we should target layers distributed across depth:")
    
    # Calculate layer distribution for 36-layer model
    total_layers = 36
    # Target every ~6th layer to get good coverage (similar to 7,15,23 pattern in 32-layer model)
    suggested_layers = [6, 12, 18, 24, 30]  # 5 layers distributed across 36 layers
    
    # Generate target_parameters in the format shown in working example
    target_parameters = []
    for layer_num in suggested_layers:
        target_parameters.extend([
            f"{layer_num}.mlp.experts.gate_up_proj",
            f"{layer_num}.mlp.experts.down_proj"
        ])
    
    print(f"--lora_target_parameters \"{','.join(target_parameters)}\"")
    print(f"")
    print(f"This follows the exact pattern from the working example!")
    
    print("="*80)
    print("END MODEL ARCHITECTURE INSPECTION")
    print("="*80 + "\n")

    peft_config = None
    chat_template = None
    if args.use_peft_lora:
        # Follow the working example pattern for GPT-OSS expert targeting
        target_modules = args.lora_target_modules.split(",") if args.lora_target_modules != "all-linear" else args.lora_target_modules
        
        # Configure LoRA following the official working example
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
            print(f"Using target_modules: {target_modules}")
            print("Following the official GPT-OSS expert targeting pattern!")
        
        peft_config = LoraConfig(**lora_config_args)

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