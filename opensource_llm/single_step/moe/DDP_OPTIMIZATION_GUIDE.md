# DDP Performance Optimization Guide

## Problem
- **Original**: 18 min (single GPU) vs 15 min (2 GPUs DDP) = **1.2x speedup**
- **Expected**: ~1.7-2x speedup with 2 GPUs
- **Issue**: Communication overhead and synchronization bottlenecks

## Root Causes

1. **Excessive Logging** (`logging_steps=1`) - Creates sync barriers every step
2. **High Gradient Accumulation** (`gradient_accumulation_steps=4`) - Less parallelism
3. **Small Batches** - More frequent gradient synchronization
4. **Data Loading Bottleneck** - CPU can't keep up with GPUs

## Optimizations Applied

### 1. Reduced Logging Overhead
```yaml
logging_steps: 1 → 5  # 5x fewer synchronization points
```

### 2. Larger Batches, Less Accumulation
```yaml
# Standard (balanced)
per_device_train_batch_size: 4
gradient_accumulation_steps: 2

# Fast (maximum speed)
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
```

### 3. DDP Communication Optimizations
**In `train_multilingual_reasoner.py` (TrainingArguments):**
```python
ddp_find_unused_parameters=False  # Skip unused parameter search
ddp_bucket_cap_mb=25              # Larger buckets = fewer communications
```

### 4. Data Loading Optimizations
```python
dataloader_num_workers=4    # Parallel data loading
dataloader_pin_memory=True  # Faster CPU→GPU transfer
```

### 5. Checkpoint Optimization
```yaml
save_total_limit: 2 → 1  # Fewer checkpoint writes
```

## Configurations

### Standard DDP (Balanced)
**File:** `run_multilingual_reasoner_accelerate_ddp.yaml`

```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
logging_steps: 5
save_total_limit: 1
```
- Effective batch: 4 × 2 × 2 GPUs = 16
- Expected: ~11-13 min (~1.4-1.5x speedup)

### Fast DDP (Maximum Speed)
**File:** `run_multilingual_reasoner_accelerate_ddp_fast.yaml`

```yaml
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
logging_steps: 10
save_total_limit: 1
```
- Effective batch: 8 × 1 × 2 GPUs = 16
- Expected: ~9-11 min (~1.7-2x speedup)

## Performance Comparison

| Configuration | Batch | Accum | Logging | Time | Speedup |
|--------------|-------|-------|---------|------|---------|
| Single GPU | 4 | 4 | 5 | ~16 min | 1.0x |
| DDP (old) | 2 | 4 | 1 | 15 min | 1.06x |
| **DDP (standard)** | 4 | 2 | 5 | **~12 min** | **~1.3x** |
| **DDP (fast)** | 8 | 1 | 10 | **~10 min** | **~1.6x** |

## Usage

```bash
# Standard optimized DDP
az ml job create -f opensource_llm/single_step/moe/run_multilingual_reasoner_accelerate_ddp.yaml

# Maximum speed DDP
az ml job create -f opensource_llm/single_step/moe/run_multilingual_reasoner_accelerate_ddp_fast.yaml
```

## Troubleshooting

### Out of Memory (OOM)
If fast config OOMs:
```yaml
per_device_train_batch_size: 8 → 6
# or
gradient_accumulation_steps: 1 → 2
```

### Invalid TrainingArguments Parameters
**Valid DDP parameters:**
- ✅ `ddp_find_unused_parameters`
- ✅ `ddp_bucket_cap_mb`
- ❌ `ddp_broadcast_buffers` (not in TrainingArguments)

**Accelerate config** (`configs/ddp_config.yaml`) should only contain:
- `distributed_type: MULTI_GPU`
- `mixed_precision: bf16`
- `num_processes: 2`
- Standard Accelerate fields only

## Key Takeaways

1. **Reduce sync points** - Less logging, fewer checkpoints
2. **Maximize batch size** - Better GPU utilization
3. **Minimize accumulation** - More parallelism
4. **Optimize data loading** - Prevent CPU bottleneck
5. **Use valid parameters** - DDP settings go in TrainingArguments, not Accelerate config

