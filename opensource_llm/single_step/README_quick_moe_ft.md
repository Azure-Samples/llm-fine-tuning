# Quick MOE Fine-tuning with Accelerate

This is a refactored version of the quick MOE fine-tuning script that works with Azure ML and accelerate for distributed training.

## Files Created/Modified

1. **quick_moe_ft.py** - Refactored training script with proper argument parsing
2. **run_quick_moe_ft.yml** - Azure ML job configuration for single GPU training
3. **run_quick_moe_ft_fsdp.yml** - Azure ML job configuration for multi-GPU FSDP training
4. **conda.yml** - Conda environment with required dependencies
5. **configs/accelerate_config.yaml** - Accelerate config for single GPU
6. **configs/fsdp_accelerate_config.yaml** - Accelerate config for FSDP multi-GPU

## Key Changes from Original Script

### Parameterization
- All hardcoded values are now configurable via command line arguments
- Organized into logical argument groups: ModelArguments, DataTrainingArguments, LoRAArguments
- Default values match the original script settings

### Accelerate Integration
- Script now works with `accelerate launch` for distributed training
- Supports both single GPU and multi-GPU FSDP training
- Proper argument parsing compatible with HuggingFace ecosystem

### Azure ML Ready
- Created corresponding YAML files for Azure ML job submission
- Includes proper environment setup with conda.yml
- Supports both single-node single-GPU and single-node multi-GPU training

## Usage

### Single GPU Training
```bash
az ml job create --file run_quick_moe_ft.yml --workspace-name <workspace> --resource-group <rg>
```

### Multi-GPU FSDP Training
```bash
az ml job create --file run_quick_moe_ft_fsdp.yml --workspace-name <workspace> --resource-group <rg>
```

### Local Testing
```bash
# Single GPU
accelerate launch --config_file configs/accelerate_config.yaml quick_moe_ft.py

# Multi-GPU FSDP
accelerate launch --config_file configs/fsdp_accelerate_config.yaml quick_moe_ft.py
```

## Key Parameters

### Model Arguments
- `--model_name_or_path`: Model to fine-tune (default: "Qwen/Qwen3-30B-A3B")
- `--use_4bit_quantization`: Enable 4-bit quantization (default: true)
- `--bnb_4bit_compute_dtype`: Compute dtype for quantization (default: "bfloat16")

### Data Arguments
- `--dataset_name`: Dataset to use (default: "FreedomIntelligence/medical-o1-reasoning-SFT")
- `--dataset_split`: Dataset split (default: "train[0:2000]")
- `--max_seq_length`: Maximum sequence length (default: 2048)

### LoRA Arguments
- `--lora_r`: LoRA rank (default: 64)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--lora_dropout`: LoRA dropout (default: 0.05)
- `--lora_target_modules`: Target modules (default: "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

### Training Arguments
All standard HuggingFace TrainingArguments are supported, with defaults matching the original script.

## Notes

- The script removes the inference/generation portion from the original to focus on training
- Memory optimization techniques (gc.collect(), torch.cuda.empty_cache()) are included
- Compatible with both Azure ML and local training environments
- FSDP configuration allows training of large models across multiple GPUs
