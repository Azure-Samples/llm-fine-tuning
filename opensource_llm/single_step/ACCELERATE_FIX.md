# Fix Summary: Accelerate GPU Detection and Use Cache Removal

## Changes Made

### 1. Fixed Process Count in YAML (run_fsdp_nc80h100_Qwen.yml)
**Changed:**
```yaml
process_count_per_instance: 1 # Let accelerate auto-detect GPUs
```

**Reason:**
- Accelerate automatically detects the number of GPUs on a single node
- Setting this to match the number of GPUs can cause conflicts
- Azure ML + Accelerate works best when this is set to 1

### 2. Removed use_cache from Model Loading (utils_v2.py)
**Removed:**
```python
"use_cache": False,  # Set to False for training with gradient checkpointing
```

**Reason:**
- The use_cache parameter can cause issues during model initialization
- It's better to let the model handle caching internally
- FSDP and gradient checkpointing will handle cache behavior appropriately

### 3. Removed use_cache Configuration (train_v2.py)
**Removed:**
```python
model.config.use_cache = not training_args.gradient_checkpointing
```

**Reason:**
- This explicit cache configuration can interfere with FSDP
- Modern training frameworks handle cache settings automatically
- Removes potential source of configuration conflicts

## Expected Results

### ✅ **GPU Detection**
- Accelerate will automatically detect all available GPUs on the node
- No manual GPU count specification needed
- Proper FSDP distribution across detected GPUs

### ✅ **Cache Behavior**
- No explicit cache configuration conflicts
- Model will handle caching appropriately for training
- FSDP will manage memory efficiently without cache interference

### ✅ **Training Stability**
- Reduced configuration conflicts
- More reliable FSDP initialization
- Better compatibility with Azure ML infrastructure

## Technical Notes

- **Multi-GPU**: With `process_count_per_instance: 1`, accelerate spawns processes equal to the number of detected GPUs
- **FSDP**: Works best when not explicitly constrained by cache settings
- **Azure ML**: This configuration follows Azure ML best practices for distributed training
