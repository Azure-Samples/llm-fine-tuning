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
from single_eval import evaluate_single           # ← the helper we wrote


# --------------------------------------------------------------------------- #
# 1.  pick a BFCL category and build a HuggingFace dataset
# --------------------------------------------------------------------------- #
CATEGORY     = os.getenv("BFCL_CATEGORY", "simple")          # override via env-var
BASE_MODEL   = os.getenv("BASE_MODEL"   , "Qwen/Qwen2-0.5B-Instruct")

prompt_path  = PROMPT_PATH / TEST_FILE_MAPPING[CATEGORY]
raw_entries  = load_file(prompt_path, sort_by_id=True)

def _serialise_prompt(entry: Dict) -> str:
    """
    Turn the structured chat-prompt into a plain text string that works for
    vanilla causal-LMs.  Very simple:  'role: content\n' … concatenation.
    """
    msg_lines = []
    for m in entry["question"]:
        msg_lines.append(f"{m['role']}: {m['content']}")
    # include the function list – again, super naive
    fn_doc = json.dumps(entry["function"], ensure_ascii=False)
    msg_lines.append(f"FUNCTIONS: {fn_doc}")
    msg_lines.append("ASSISTANT:")              # let the model fill in
    return "\n".join(msg_lines)

dataset = Dataset.from_list(
    [
        {"prompt": _serialise_prompt(e), "id": e["id"]}
        for e in raw_entries
    ]
)


# --------------------------------------------------------------------------- #
# 2.  Reward function wrapper around our `evaluate_single`
# --------------------------------------------------------------------------- #
def reward_bfcl(completions: List[str], samples: List[Dict], **_) -> List[float]:
    """
    `completions`  – the model outputs for the batch
    `samples`      – the batch rows (each row is the *dataset* dict)
    Returns one reward per completion in [0,1].
    """
    rewards = []
    for comp, sample in zip(completions, samples):
        score = evaluate_single(
            category       = CATEGORY,
            model_response = comp,
            entry_id       = sample["id"],
        )
        rewards.append(1.0 if score["valid"] else 0.0)
    return rewards


# --------------------------------------------------------------------------- #
# 3.  GRPO-Trainer
# --------------------------------------------------------------------------- #
training_args = GRPOConfig(
    output_dir   = f"{Path.cwd()}/{BASE_MODEL.split('/')[-1]}-{CATEGORY}-grpo",
    logging_steps= 10,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps= 1,
    max_steps    = 300,          # small demo run – adapt to your compute
    learning_rate= 2e-5,
    fp16         = torch.cuda.is_available(),
)

trainer = GRPOTrainer(
    model         = BASE_MODEL,
    reward_funcs  = reward_bfcl,
    args          = training_args,
    train_dataset = dataset,
    #  instruct GRPO which column to feed as **query**
    query_column  = "prompt",
)

if __name__ == "__main__":
    trainer.train()