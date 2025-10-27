# Fix Summary: HfArgumentParser Cache Directory Error

## Problem
The training job was failing with the error:
```
ValueError: Some specified arguments are not used by the HfArgumentParser: ['--cache_dir', '/mnt/azureml/cr/j/f5e92066bbb1448b9ff970d3aa3370bc/cap/data-capability/wd/cache_dir']
```

## Root Cause
The `--cache_dir` argument was being passed in the YAML file but was not defined in any of the argument dataclasses in `train_v2.py`.

## Fixes Applied

### 1. Added `cache_dir` Argument Definition (train_v2.py)
Added the missing argument to the `ModelArguments` dataclass:
```python
cache_dir: Optional[str] = field(
    default=None,
    metadata={"help": "Directory to store downloaded models and datasets"},
)
```

### 2. Updated Model Loading (utils_v2.py)
Added cache_dir support to model loading:
```python
# Add cache_dir if specified
if hasattr(args, 'cache_dir') and args.cache_dir:
    model_kwargs["cache_dir"] = args.cache_dir
    print(f"Using cache directory: {args.cache_dir}")
```

### 3. Updated Tokenizer Loading (utils_v2.py)
Updated both tokenizer loading paths to support cache_dir:
- For special tokens case
- For default case

### 4. Updated Dataset Loading (utils_v2.py)
Updated `create_datasets` function to:
- Accept `model_args` parameter
- Use cache_dir for dataset downloading
- Updated function call in train_v2.py

### 5. Fixed Process Count Mismatch (YAML)
Fixed inconsistency between command line and distribution settings:
- Command line: `--num_processes 2`
- Distribution: `process_count_per_instance: 2` (was 1)

## Files Modified

1. **train_v2.py**
   - Added `cache_dir` field to `ModelArguments`
   - Updated `create_datasets` call to pass `model_args`

2. **utils_v2.py**
   - Added cache_dir support to model loading
   - Added cache_dir support to tokenizer loading (both paths)
   - Updated `create_datasets` function signature and implementation

3. **run_fsdp_nc80h100_Qwen.yml**
   - Fixed process count: `process_count_per_instance: 2`

## Expected Result
The training job should now start without the HfArgumentParser error, and the cache directory will be properly used for:
- Model downloads and caching
- Tokenizer downloads and caching  
- Dataset downloads and caching

This ensures efficient caching in Azure ML environments and avoids repeated downloads.
