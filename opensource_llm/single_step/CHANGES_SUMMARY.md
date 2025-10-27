# Changes Summary: GPT-OSS vs Qwen Model Support

## Overview
This document summarizes the changes made to support both GPT-OSS and Qwen models with proper model-specific configurations.

## Key Changes Made

### 1. **Model Family Detection (utils_v2.py)**
- Added automatic detection of model family based on model name
- `is_gpt_oss = "gpt-oss" in args.model_name_or_path.lower()`
- `model_family = "gpt-oss" if is_gpt_oss else "qwen" if "qwen" in args.model_name_or_path.lower() else "other"`

### 2. **Quantization Configuration**
- **GPT-OSS models**: Use `Mxfp4Config(dequantize=True)` as recommended by OpenAI
- **Qwen/Other models**: Use `BitsAndBytesConfig` for standard quantization
- Conditional logic ensures the right quantization method is applied

### 3. **Attention Implementation**
- **GPT-OSS models**: Use "eager" attention as recommended
- **Qwen models**: Allow Flash Attention 2 if requested and available
- Fallback to "eager" if Flash Attention is not available

### 4. **LoRA Configuration Simplification**
- **GPT-OSS models**: Support expert targeting with `lora_target_parameters`
- **Qwen models**: Use standard target modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- Removed complex model inspection code that was GPT-OSS specific

### 5. **YAML Configuration Updates (run_fsdp_nc80h100_Qwen.yml)**
- Updated display name: `qwen-sft-lora-multigpu`
- Updated experiment name: `qwen-sft-training`
- Changed target modules for Qwen: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Fixed distribution setting: `process_count_per_instance: 2` (matches `--num_processes 2`)

### 6. **Docker Dependencies Cleanup**
- Removed `vllm` from requirements.txt (unnecessary for training)
- Kept `flash-attn` for potential Flash Attention support

### 7. **Parameter Definitions**
- Confirmed `lora_target_parameters` is already defined in train_v2.py
- All required arguments are properly defined

## Benefits

1. **Model-Specific Optimization**: Each model family gets the most appropriate configuration
2. **Simplified Logic**: Removed complex model inspection code
3. **Better Performance**: Qwen models can use Flash Attention, GPT-OSS uses recommended settings
4. **Cleaner Dependencies**: Removed unnecessary packages
5. **Consistent Configuration**: Fixed mismatched process counts

## Usage

### For Qwen Models:
- The current configuration will automatically detect Qwen models
- Uses standard LoRA target modules
- Allows Flash Attention for better performance

### For GPT-OSS Models:
- Change model name to include "gpt-oss"
- Will automatically use Mxfp4Config and expert targeting
- Uses eager attention as recommended

## Next Steps

1. Test with actual Qwen model to verify target modules are correct
2. For GPT-OSS models, add `--lora_target_parameters` with appropriate expert layer specifications
3. Monitor training performance and adjust batch sizes if needed
