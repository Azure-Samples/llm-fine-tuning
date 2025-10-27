# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers.integrations import MLflowCallback  

dataset = load_dataset("trl-lib/tldr", split="train")
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl  
import mlflow  

# --------------------------------------------------------------------------- #
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


# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen3-1.7B-GRPO", logging_steps=10,     max_steps    = 1000,          # small demo run – adapt to your compute
)
trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.remove_callback(MLflowCallback)  
trainer.add_callback(CustomMlflowCallback())           # add your custom one  

trainer.train()