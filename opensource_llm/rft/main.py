import json
import unittest
from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer # Example imports

# --- Function to load BFCL dataset ---
def load_bfcl_dataset(file_path="BFCL_v3_multiple.json"):
    """
    Loads the BFCL dataset from a JSONL file into a Hugging Face Dataset.
    Each line in the JSONL file is expected to be a valid JSON object.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        datasets.Dataset: The loaded dataset.
    """
    try:
        # load_dataset with "json" type can handle JSONL files directly.
        dataset = load_dataset("json", data_files=file_path, split="train")
        return dataset
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        # Fallback or alternative loading if needed, e.g., manual parsing:
        # def generate_examples():
        #     with open(file_path, 'r', encoding='utf-8') as f:
        #         for line in f:
        #             yield json.loads(line)
        # return Dataset.from_generator(generate_examples)
        raise


# --- Reward Function and Helpers ---
def _make_hashable(value):
    """Recursively converts a value to a hashable type."""
    if isinstance(value, dict):
        # Sort items by key for consistent frozenset creation
        return frozenset((k, _make_hashable(v_item)) for k, v_item in sorted(value.items()))
    elif isinstance(value, list):
        return tuple(_make_hashable(v_item) for v_item in value)
    elif isinstance(value, set):
        # Sort items for consistent frozenset creation
        return frozenset(_make_hashable(v_item) for v_item in sorted(list(value)))
    return value

def _canonicalize_call(call):
    """
    Converts a function call dictionary to a canonical, hashable representation.
    The representation is a tuple: (function_name, frozenset_of_canonical_arguments).
    Returns None if the call is malformed.
    """
    if not isinstance(call, dict) or "name" not in call or "arguments" not in call:
        return None
    if not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
        return None

    # Canonicalize arguments by making them hashable and order-insensitive at each level
    # Sort argument items by key before creating the frozenset for consistency
    try:
        canonical_args = frozenset(
            (k, _make_hashable(v)) for k, v in sorted(call["arguments"].items())
        )
        return (call["name"], canonical_args)
    except TypeError: # If an argument value is unhashable despite _make_hashable
        return None


def compute_reward(predictions, references):
    """
    Computes a reward for function call predictions.
    The reward is 1.0 if the set of predicted function calls (name and arguments)
    exactly matches the set of reference function calls. Order of calls within the list
    and order of arguments within a call do not matter. Otherwise, the reward is 0.0.

    Args:
        predictions (list): A list of predicted function calls.
                            Each call is a dict: {"name": str, "arguments": dict}
        references (list): A list of reference (ground truth) function calls.
                           Same structure as predictions.
    Returns:
        float: Reward value (0.0 or 1.0).
    """
    if not isinstance(predictions, list) or not isinstance(references, list):
        return 0.0 # Invalid input format

    # Convert predictions and references to sets of canonical call representations
    # Filter out any None results from _canonicalize_call (malformed calls)
    canonical_preds = set(filter(None, (_canonicalize_call(p) for p in predictions)))
    canonical_refs = set(filter(None, (_canonicalize_call(r) for r in references)))

    if canonical_preds == canonical_refs:
        return 1.0
    else:
        return 0.0

# --- Unit Tests for the Reward Function ---
class TestRewardFunction(unittest.TestCase):
    def test_exact_match_single_function(self):
        pred = [{"name": "func1", "arguments": {"arg1": "val1", "arg2": 10}}]
        ref = [{"name": "func1", "arguments": {"arg2": 10, "arg1": "val1"}}] # Arg order different
        self.assertEqual(compute_reward(pred, ref), 1.0)

    def test_exact_match_multiple_functions_order_agnostic(self):
        pred = [
            {"name": "func1", "arguments": {"a": 1}},
            {"name": "func2", "arguments": {"b": "two"}}
        ]
        ref = [
            {"name": "func2", "arguments": {"b": "two"}},
            {"name": "func1", "arguments": {"a": 1}}
        ] # Call order different
        self.assertEqual(compute_reward(pred, ref), 1.0)

    def test_exact_match_nested_arguments(self):
        pred = [{"name": "func1", "arguments": {"param_str": "val1", "param_nested": {"n_key1": 1, "n_key2": [10, 20], "n_key3": {"sub": "s"}}}}]
        ref = [{"name": "func1", "arguments": {"param_nested": {"n_key2": [10, 20], "n_key1": 1, "n_key3": {"sub": "s"}}, "param_str": "val1"}}]
        self.assertEqual(compute_reward(pred, ref), 1.0)

        pred_wrong_nested_val = [{"name": "func1", "arguments": {"param_str": "val1", "param_nested": {"n_key1": 1, "n_key2": [10, 30]}}}] # n_key2 different
        ref_for_wrong = [{"name": "func1", "arguments": {"param_str": "val1", "param_nested": {"n_key1": 1, "n_key2": [10, 20]}}}]
        self.assertEqual(compute_reward(pred_wrong_nested_val, ref_for_wrong), 0.0)

    def test_no_match_wrong_function_name(self):
        pred = [{"name": "func_wrong", "arguments": {"arg1": "val1"}}]
        ref = [{"name": "func1", "arguments": {"arg1": "val1"}}]
        self.assertEqual(compute_reward(pred, ref), 0.0)

    def test_no_match_wrong_argument_value(self):
        pred = [{"name": "func1", "arguments": {"arg1": "val_wrong"}}]
        ref = [{"name": "func1", "arguments": {"arg1": "val1"}}]
        self.assertEqual(compute_reward(pred, ref), 0.0)

    def test_no_match_wrong_argument_key(self):
        pred = [{"name": "func1", "arguments": {"arg_wrong": "val1"}}]
        ref = [{"name": "func1", "arguments": {"arg1": "val1"}}]
        self.assertEqual(compute_reward(pred, ref), 0.0)

    def test_no_match_extra_argument(self):
        pred = [{"name": "func1", "arguments": {"arg1": "val1", "extra_arg": "foo"}}]
        ref = [{"name": "func1", "arguments": {"arg1": "val1"}}]
        self.assertEqual(compute_reward(pred, ref), 0.0)

    def test_no_match_missing_argument(self):
        pred = [{"name": "func1", "arguments": {"arg1": "val1"}}]
        ref = [{"name": "func1", "arguments": {"arg1": "val1", "arg2": "bar"}}]
        self.assertEqual(compute_reward(pred, ref), 0.0)

    def test_no_match_missing_function_in_prediction(self):
        pred = [{"name": "func1", "arguments": {"a": 1}}]
        ref = [
            {"name": "func1", "arguments": {"a": 1}},
            {"name": "func2", "arguments": {"b": "two"}}
        ]
        self.assertEqual(compute_reward(pred, ref), 0.0)

    def test_no_match_extra_function_in_prediction(self):
        pred = [
            {"name": "func1", "arguments": {"a": 1}},
            {"name": "func2", "arguments": {"b": "two"}}
        ]
        ref = [{"name": "func1", "arguments": {"a": 1}}]
        self.assertEqual(compute_reward(pred, ref), 0.0)

    def test_empty_predictions_and_references(self):
        self.assertEqual(compute_reward([], []), 1.0, "Both empty should match")

    def test_empty_predictions_non_empty_references(self):
        ref = [{"name": "func1", "arguments": {"a": 1}}]
        self.assertEqual(compute_reward([], ref), 0.0)

    def test_non_empty_predictions_empty_references(self):
        pred = [{"name": "func1", "arguments": {"a": 1}}]
        self.assertEqual(compute_reward(pred, []), 0.0)
        
    def test_malformed_prediction_call_structure(self):
        # Test various malformations
        pred_no_name = [{"arguments": {"a": 1}}]
        pred_no_args = [{"name": "func1"}]
        pred_wrong_type_name = [{"name": 123, "arguments": {"a": 1}}]
        pred_wrong_type_args = [{"name": "func1", "arguments": "not_a_dict"}]
        pred_not_a_dict = ["not_a_dict_call"]
        
        ref = [{"name": "func1", "arguments": {"a": 1}}]
        
        self.assertEqual(compute_reward(pred_no_name, ref), 0.0)
        self.assertEqual(compute_reward(pred_no_args, ref), 0.0)
        self.assertEqual(compute_reward(pred_wrong_type_name, ref), 0.0)
        self.assertEqual(compute_reward(pred_wrong_type_args, ref), 0.0)
        self.assertEqual(compute_reward(pred_not_a_dict, ref), 0.0)

    def test_malformed_reference_call_structure(self):
        pred = [{"name": "func1", "arguments": {"a": 1}}]
        ref_malformed = [{"name_typo": "func1", "arguments": {"a": 1}}]
        self.assertEqual(compute_reward(pred, ref_malformed), 0.0)

    def test_mixed_malformed_and_valid_calls(self):
        # If malformed calls are filtered out, the valid parts should still be compared.
        pred_mixed = [
            {"name": "func1", "arguments": {"a": 1}}, # Valid
            {"name_typo": "func2", "arguments": {"b": 2}} # Malformed
        ]
        ref_valid_part = [
            {"name": "func1", "arguments": {"a": 1}}
        ]
        # _canonicalize_call for the malformed part returns None, which is filtered out.
        # So, canonical_preds becomes {('func1', frozenset({('a',1)}))}
        # canonical_refs becomes {('func1', frozenset({('a',1)}))}
        self.assertEqual(compute_reward(pred_mixed, ref_valid_part), 1.0)

        pred_valid_part = [
            {"name": "func1", "arguments": {"a": 1}}
        ]
        ref_mixed = [
            {"name": "func1", "arguments": {"a": 1}}, # Valid
            {"arguments": {"b": 2}} # Malformed (missing name)
        ]
        self.assertEqual(compute_reward(pred_valid_part, ref_mixed), 1.0)

    def test_invalid_top_level_input(self):
        self.assertEqual(compute_reward(None, []), 0.0)
        self.assertEqual(compute_reward([], None), 0.0)
        self.assertEqual(compute_reward("not a list", [{"name": "f", "arguments": {}}]), 0.0)


# --- Main execution for demonstration and testing ---
def main():
    # 1. Load dataset
    print("Attempting to load BFCL dataset from 'BFCL_v3_multiple.json'...")
    try:
        bfcl_dataset = load_bfcl_dataset()
        print(f"Dataset loaded successfully. Number of examples: {len(bfcl_dataset)}")
        print(f"Features: {bfcl_dataset.features}")
        if len(bfcl_dataset) > 0:
            print(f"First example: {json.dumps(bfcl_dataset[0], indent=2)}")
    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Please ensure 'BFCL_v3_multiple.json' is in the same directory as this script.")
        return

    # --- Placeholder for actual training code ---
    # The loaded `bfcl_dataset` contains 'question' and 'function' (schemas).
    # For training, you would typically need to:
    # 1. Preprocess this data:
    #    - Format the 'question' and available 'function' schemas into a prompt for your model.
    #    - Have corresponding target labels, which would be the *actual function call(s)*
    #      (e.g., [{"name": "func_name", "arguments": {"param": "value"}}]) that the model
    #      should generate. This target label is NOT directly in BFCL_v3_multiple.json.
    #      You might need another dataset or a way to derive these target calls.
    # 2. Tokenize inputs and labels.
    # 3. Set up a model (e.g., from Hugging Face Transformers).
    # 4. Use the Trainer API or a custom training loop.
    # 5. The `compute_reward` function can be used in `compute_metrics` for evaluation,
    #    but it requires model predictions and ground truth labels in the specified format.

    print("\n" + "="*30)
    print("TRAINING PIPELINE NOTES:")
    print("The loaded dataset (BFCL_v3_multiple.json) provides function *schemas*.")
    print("The `compute_reward` function and its tests assume that you have target *function calls*")
    print("(i.e., function name with filled arguments) for comparison against model predictions.")
    print("This is a common setup for function calling tasks, but the target calls are not in the")
    print("provided JSON. For a full training pipeline, these target labels would be necessary.")
    print("Placeholder for actual training logic would go here.")
    print("="*30 + "\n")


    # 2. Run unit tests for the reward function
    print("Running unit tests for the `compute_reward` function...")
    suite = unittest.TestSuite()
    # Add all tests from TestRewardFunction class to the suite
    suite.addTest(unittest.makeSuite(TestRewardFunction))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\nAll reward function unit tests passed successfully!")
    else:
        print("\nSome reward function unit tests failed.")

if __name__ == "__main__":
    # Create a dummy BFCL_v3_multiple.json for the script to run if it doesn't exist
    # In a real scenario, this file would be provided.
    try:
        with open("BFCL_v3_multiple.json", "r") as f:
            pass # File exists
    except FileNotFoundError:
        print("Creating a dummy 'BFCL_v3_multiple.json' for demonstration purposes...")
        dummy_data = [
            {"id": "multiple_0", "question": [[{"role": "user", "content": "Triangle sides 5, 4, 3. Properties?"}]], "function": [{"name": "triangle_properties.get", "description": "Desc1", "parameters": {"type": "dict", "properties": {"side1": {"type": "integer"}}}}]},
            {"id": "multiple_1", "question": [[{"role": "user", "content": "Capital of Brazil?"}]], "function": [{"name": "country_info.capital", "description": "Desc2", "parameters": {"type": "dict", "properties": {"country": {"type": "string"}}}}]}
        ]
        with open("BFCL_v3_multiple.json", "w", encoding='utf-8') as f:
            for item in dummy_data:
                f.write(json.dumps(item) + "\n")
    
    main()