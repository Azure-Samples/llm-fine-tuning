# Running Training Jobs with Azure ML

## Prerequisites
Make sure you're logged in to Azure ML:
```bash
az login
az account set --subscription <your-subscription-id>
az configure --defaults workspace=<your-workspace> group=<your-resource-group>
```

## Single GPU Training

### Standard (Optimized)
```bash
az ml job create -f opensource_llm/single_step/moe/run_multilingual_reasoner.yaml
```

**Configuration:**
- Batch size: 4
- Gradient accumulation: 4
- Effective batch size: 16
- Logging every 5 steps
- Uses quantization (MXFP4)

## DDP Training (2 GPUs)

### Standard (Optimized)
```bash
az ml job create -f opensource_llm/single_step/moe/run_multilingual_reasoner_accelerate_ddp.yaml
```

**Configuration:**
- Batch size: 4 per GPU
- Gradient accumulation: 2
- Effective batch size: 16 (4 × 2 × 2)
- Logging every 5 steps
- DDP optimizations enabled

### Fast (Maximum Speed)
```bash
az ml job create -f opensource_llm/single_step/moe/run_multilingual_reasoner_accelerate_ddp_fast.yaml
```

**Configuration:**
- Batch size: 8 per GPU
- Gradient accumulation: 1
- Effective batch size: 16 (8 × 1 × 2)
- Logging every 10 steps
- Maximum parallelism

## FSDP Training (2 GPUs) - Memory Efficient

**Note:** FSDP does not support `target_parameters` for MoE experts. Use only if you remove expert targeting.

```bash
az ml job create -f opensource_llm/single_step/moe/run_multilingual_reasoner_accelerate_fsdp.yaml
```

## Monitor Job

After creating a job, you'll get a job URL. You can also monitor via CLI:

```bash
# List recent jobs
az ml job list --max-results 5 -o table

# Show specific job
az ml job show --name <job-name>

# Stream logs
az ml job stream --name <job-name>

# Download outputs
az ml job download --name <job-name> --output-name output_dir
```

## Expected Performance (Optimized Configs)

| Configuration | Time | Samples/sec | Speedup |
|--------------|------|-------------|---------|
| Single GPU | ~15-17 min | ~60 | 1.0x |
| DDP Standard | ~11-13 min | ~80 | ~1.4x |
| DDP Fast | ~9-11 min | ~100 | ~1.7x |

**Note:** DDP versions don't use quantization (MXFP4) to avoid compatibility issues. Single GPU uses quantization for memory efficiency.

## Troubleshooting

### Out of Memory (OOM)
If you get OOM errors with the fast config:
1. Reduce batch size: `per_device_train_batch_size: 8 → 6`
2. Or increase gradient accumulation: `gradient_accumulation_steps: 1 → 2`

### Slow Data Loading
If you see CPU bottleneck (low GPU utilization):
- Increase `dataloader_num_workers` in the training script
- Ensure data is cached properly

### Job Not Starting
Check compute availability:
```bash
az ml compute show --name nc80h100
```

## Quick Reference

**From current directory:**
```bash
# Single GPU
az ml job create -f run_multilingual_reasoner.yaml

# DDP optimized
az ml job create -f run_multilingual_reasoner_accelerate_ddp.yaml

# DDP fast
az ml job create -f run_multilingual_reasoner_accelerate_ddp_fast.yaml
```

**From workspace root:**
```bash
az ml job create -f opensource_llm/single_step/moe/run_multilingual_reasoner_accelerate_ddp.yaml
```
