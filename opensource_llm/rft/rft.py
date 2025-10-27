"""
Fine-tune a model with GRPO where the reward signal is the official BFCL
checker (1 = everything correct; 0 = anything wrong).

• ‟category”      … one of the keys in TEST_FILE_MAPPING  (e.g. "simple")
• ‟base_model”    … any causal-LM that the TRL stack can load
• This script trains only on *single-turn* categories to keep it simple.
  (multi-turn works the same way – you’d just have to format the prompt so
  that the model emits the whole conversation in one completion.)
"""

from __future__ import annotations

from pathlib import Path
import json, os, random, torch
from typing import Dict, List

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from bfcl.constants.category_mapping import TEST_FILE_MAPPING
from bfcl.constants.eval_config import PROMPT_PATH
from bfcl.utils import load_file
from transformers.integrations import MLflowCallback  
import json  
import pandas as pd  
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl  
import mlflow  
from bfcl.eval_checker.ast_eval.ast_checker import ast_checker

# --------------------------------------------------------------------------- #
# 1.  pick a BFCL category and build a HuggingFace dataset
# --------------------------------------------------------------------------- #
CATEGORY     = os.getenv("BFCL_CATEGORY", "parallel_multiple")          # override via env-var
BASE_MODEL   = os.getenv("BASE_MODEL"   , "Qwen/Qwen3-1.7B")

prompt_path  = PROMPT_PATH / TEST_FILE_MAPPING[CATEGORY]
raw_entries  = load_file(prompt_path, sort_by_id=True)

MY_METRICS = [  
    'loss', 'grad_norm', 'learning_rate', 'num_tokens',  
    'completions/mean_length', 'completions/min_length', 'completions/max_length',  
    'completions/clipped_ratio', 'completions/mean_terminated_length', 'completions/min_terminated_length', 'completions/max_terminated_length',  
    'rewards/reward_function/mean', 'rewards/reward_function/std', 'reward', 'reward_std', 'frac_reward_zero_std',  
    'kl', 'clip_ratio/low_mean', 'clip_ratio/low_min', 'clip_ratio/high_mean', 'clip_ratio/high_max', 'clip_ratio/region_mean', 'epoch'  
]  

class CustomMlflowCallback(TrainerCallback):  
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):  
        if logs is None:  
            return  
        filtered_logs = {k: v for k, v in logs.items() if k in MY_METRICS and isinstance(v, (int, float))}  
        if filtered_logs:  
            mlflow.log_metrics(filtered_logs, step=int(state.global_step))  

# 1. Load both files into dictionaries keyed by id  
def load_jsonl(filename):  
    data = {}  
    with open(filename, "r", encoding="utf-8") as f:  
        for line in f:  
            row = json.loads(line)  
            data[row["id"]] = row  
    return data  
  
file1 = load_jsonl("data/BFCL_v3_parallel_multiple.json")  
file2 = load_jsonl("data/possible_answer/BFCL_v3_parallel_multiple.json")  
  
# 2. Merge by id  
records = []  
for id_, row1 in file1.items():  
    if id_ not in file2:  
        continue  
    row2 = file2[id_]  
      
    # Extract question, function, ground_truth  
    # question is [[{"role": ..., "content": ...}]], so flatten  
    messages = []  
    for msg_group in row1["question"]:  
        for msg in msg_group:  
            messages.append(msg)  
    functions = row1["function"]  
    ground_truth = row2["ground_truth"] 
    entry_id = row1["id"]  # Use the id from the first file 
  
    # 3. Format the prompt according to your template  
    # We'll only do the core part: system message, tool definition, user prompt  
  
    # Step 1: System message + tools  
    system_msg = "<|im_start|>system\n"  
    system_msg += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"  
    system_msg += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"  
    for tool in functions:  
        system_msg += "\n" + json.dumps(tool)  
    system_msg += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n"  
  
    # Step 2: Add messages in chat history style  
    prompt_str = system_msg  
    for msg in messages:  
        role = msg["role"]  
        content = msg["content"]  
        prompt_str += f"<|im_start|>{role}\n{content}\n"  
  
    # Step 3: Assume assistant follows (model will generate assistant's turn)  
    prompt_str += "istant\n"  
  
    # Collate all in a record  
    records.append({  
        "prompt": prompt_str,  
        "entry_id": entry_id,
        "function": json.dumps(functions, ensure_ascii=False),  
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False)  
    })  
  
dataset = Dataset.from_list(records)

# ---------------------------------------------------------------------------  
# 2.  Reward function (dense version)  
# ---------------------------------------------------------------------------  
import math  
import re  
import json  
from typing import List  
  
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)  
  
  
def _extract_tool_calls(text: str):  
    """  
    Returns a list[dict] like  
        [{"name": "...", "arguments": {...}}, …]  
    If nothing can be parsed, returns [].  
    """  
    calls = []  
    # Preferred path: <tool_call> … </tool_call>  
    for snippet in _TOOL_CALL_RE.findall(text):  
        try:  
            calls.append(json.loads(snippet))  
        except Exception:  
            pass  
  
    if calls:                         # success  
        return calls  
  
    # Fallback: the whole completion *is* a JSON list  
    try:  
        maybe_list = json.loads(text)  
        if isinstance(maybe_list, list):  
            return maybe_list  
    except Exception:  
        pass  
  
    return []                         # can’t parse at all  
  
  
def _arg_overlap_score(pred_args: dict, ref_args: dict) -> float:  
    """  
    Jaccard-like overlap between two *flat* argument dictionaries.  
    Gives   1.0  if all keys & values match,  
            0.0  if nothing matches.  
    """  
    if not ref_args:  
        return 1.0                    # degenerate case  
  
    matches = 0  
    for k, v in ref_args.items():  
        if k in pred_args and pred_args[k] == v:  
            matches += 1  
    return matches / len(ref_args)  
  
  
def reward_function(  
    completions,  
    entry_id,  
    function,  
    ground_truth,  
    **_  
):  
    """  
    Dense reward version.  
    """  
    rewards: List[float] = []  
  
    for comp, fid, fn_doc_json, gt_json in zip(  
        completions, entry_id, function, ground_truth  
    ):  
        # ------------------------------------------------------------  
        # 0. Format check (if we cannot even parse → heavy penalty)  
        # ------------------------------------------------------------  
        pred_calls = _extract_tool_calls(comp)  
  
        if not pred_calls:            # completely un-parseable  
            rewards.append(-1.0)  
            continue  
  
        # ------------------------------------------------------------  
        # 1. Exact validity check – same as before  
        # ------------------------------------------------------------  
        fn_doc   = json.loads(fn_doc_json)  
        gt_calls = json.loads(gt_json)  
  
        checker_out = ast_checker(  
            func_description = fn_doc,  
            model_output     = pred_calls,  
            possible_answer  = gt_calls,  
            language         = "Python",  
            test_category    = CATEGORY,  
            model_name       = BASE_MODEL,  
        )  
  
        if checker_out["valid"]:  
            rewards.append(1.0)  
            continue                  # already perfect → skip partial scoring  
  
        # ------------------------------------------------------------  
        # 2. Partial credit  
        # ------------------------------------------------------------  
        n_ref = len(gt_calls)  
        fn_name_score   = 0.0         # fraction of correct function names  
        arg_exact_score = 0.0         # arg overlap per matched function  
  
        for ref in gt_calls:  
            ref_name      = list(ref.keys())[0]  
            ref_args      = ref[ref_name]  
            # find *first* prediction that has the same name  
            pred          = next(  
                (p for p in pred_calls  
                 if p.get("name") == ref_name or list(p.keys())[0] == ref_name),  
                None  
            )  
  
            if pred is None:  
                continue              # missing call  
            fn_name_score += 1  
  
            # Unify arg dictionaries  
            if "arguments" in pred:  
                pred_args = pred["arguments"]  
            else:  
                pred_args = pred[ref_name]  
            arg_exact_score += _arg_overlap_score(pred_args, ref_args)  
  
        fn_name_score   /= max(1, n_ref)  
        arg_exact_score /= max(1, n_ref)  
  
        # Extra calls that are not in the reference are a (small) penalty  
        extra_calls = max(0, len(pred_calls) - n_ref)  
  
        reward = (  
            0.4 * fn_name_score           # 40 % credit for picking the right function(s)  
          + 0.6 * arg_exact_score         # 60 % credit for matching the arguments  
          - 0.1 * extra_calls             # tiny punishment per spurious call  
        )  
  
        # Don’t let the partial reward become positive if nothing matches  
        # (rare corner case when only spurious calls exist)  
        if math.isclose(reward, 0.0):  
            reward = -0.3  
  
        # Clamp to [-1, +1] for stability  
        rewards.append(float(max(min(reward, 1.0), -1.0)))  
  
    return rewards  
# --------------------------------------------------------------------------- #
# 3.  GRPO-Trainer
# --------------------------------------------------------------------------- #
training_args = GRPOConfig(
    output_dir   = f"{Path.cwd()}/{BASE_MODEL.split('/')[-1]}-{CATEGORY}-grpo",
    logging_steps= 10,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps= 1,
    max_steps    = 1000,          # small demo run – adapt to your compute
    learning_rate= 2e-5,
    fp16         = torch.cuda.is_available(),
)

trainer = GRPOTrainer(
    model         = BASE_MODEL,
    reward_funcs  = reward_function,
    args          = training_args,
    train_dataset = dataset,
    #  instruct GRPO which column to feed as **query**
)
trainer.remove_callback(MLflowCallback)  
trainer.add_callback(CustomMlflowCallback())           # add your custom one  


if __name__ == "__main__":
    trainer.train()