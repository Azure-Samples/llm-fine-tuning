"""
Light-weight evaluation of ONE sample without running the full bfcl pipeline.

Example
-------
>>> from bfcl.local_single_eval import evaluate_single
>>> evaluate_single(
...     category       = "simple",
...     entry_id       = "simple_20",
...     model_response = '[{"get_weather": {"location":"Berkeley"}}]'
... )
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from bfcl.constants.category_mapping import TEST_FILE_MAPPING
from bfcl.constants.eval_config import PROMPT_PATH, POSSIBLE_ANSWER_PATH
from bfcl.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
from bfcl.utils import (
    extract_test_category_from_id,
    is_empty_output,
    is_function_calling_format_output,
    is_java,
    is_js,
    is_multi_turn,
    is_relevance_or_irrelevance,
    load_file,
)

# ──────────────────────────────────────────────────────────────────────────────
# Minimal decoder helpers
# ──────────────────────────────────────────────────────────────────────────────
_FUNC_RE = re.compile(
    r"""
    (?P<name>[A-Za-z0-9_.]+)      # foo or Foo.bar
    \s*\(                         # (
    (?P<args>.*?)                 # everything up to
    \)                            # )
""",
    re.VERBOSE | re.DOTALL,
)


def _string_args_to_dict(arg_src: str) -> Dict[str, Any]:
    """
    Turns `'a=1, b="x"'`  into  `{'a':1, 'b':'x'}`  using `ast`.
    Empty string ⇒ `{}`.
    """
    arg_src = arg_src.strip()
    if not arg_src:
        return {}
    # feed into dummy fn so we can use ast.
    fake = f"f({arg_src})"
    tree = ast.parse(fake, mode="eval")
    kw = tree.body.keywords
    payload = {}
    for k in kw:
        payload[k.arg] = ast.literal_eval(k.value)
    return payload


def _dict_call_to_string(item: Dict[str, Any]) -> str:
    """
    {"name": "Foo.bar", "arguments": {...}}  OR  {"Foo.bar": {...}}
       →  "Foo.bar(arg=value, ...)"
    """
    if "name" in item and "arguments" in item:
        func, args = item["name"], item["arguments"]
    else:  # first (and only) key is the func name
        func, args = next(iter(item.items()))
    arg_str = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return f"{func}({arg_str})"


def _extract_calls(text: str) -> List[str]:
    """
    Extract every `Foo.bar(...)` inside *text* or parse JSON variants.
    """
    text = text.strip()

    # 1️⃣  JSON array of dicts
    if text.startswith("[") and ("arguments" in text or "name" in text):
        try:
            obj = json.loads(text)
        except Exception:
            obj = ast.literal_eval(text)
        if isinstance(obj, list):
            return [_dict_call_to_string(o) for o in obj]

    # 2️⃣  Remove surrounding [`[` ... `]`] if present
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    # 3️⃣  Regex for func-calls
    out: List[str] = []
    for m in _FUNC_RE.finditer(text):
        fn, arg_src = m.group("name"), m.group("args")
        arg_dict = _string_args_to_dict(arg_src)
        out.append(_dict_call_to_string({fn: arg_dict}))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Tiny dummy handler with our own decoders
# ──────────────────────────────────────────────────────────────────────────────
class _DummyHandler:
    """
    Implements only what BFCL evaluators call (`decode_ast`, `decode_execute`).
    """

    # ----------  AST decoder (single-turn)  -------------------------------- #
    def decode_ast(self, raw: Any, language: str = "Python"):
        """
        Returns a `List[Dict[str, Dict]]` exactly as bfcl checkers expect.

        Accepts:
            • python objects already in that form
            • JSON / python-literal strings
            • plain "Foo.bar(...)" or "[Foo.bar(...), …]"
        """
        # Already decoded
        if isinstance(raw, list):
            # assume caller passed correct shape
            return raw

        # First attempt: JSON / python-literal
        if isinstance(raw, str):
            txt = raw.strip()
            # try structured
            for parser in (json.loads, ast.literal_eval):
                try:
                    obj = parser(txt)
                    if isinstance(obj, list):
                        # Need to convert  [{"name":..,"arguments":{..}}, …]
                        decoded = []
                        for itm in obj:
                            if isinstance(itm, dict) and "name" in itm:
                                decoded.append({itm["name"]: itm["arguments"]})
                            else:
                                decoded.append(itm)
                        return decoded
                except Exception:
                    pass
            # fallback: regex extraction
            calls = _extract_calls(txt)
            out: List[Dict[str, Dict]] = []
            for c in calls:
                m = _FUNC_RE.match(c)
                func, arg_src = m.group("name"), m.group("args")
                out.append({func: _string_args_to_dict(arg_src)})
            return out

        raise TypeError("Unsupported model_output type for decode_ast()")

    # ----------  multi-turn execute decoder  ------------------------------ #
    def decode_execute(self, raw: Any) -> List[str]:
        """
        Takes ONE assistant message, returns `List[str]` of call-strings.
        """
        if isinstance(raw, list):
            # treat as already list of strings
            return raw
        if isinstance(raw, str):
            return _extract_calls(raw)
        raise TypeError("Unsupported model_output type for decode_execute()")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_single(path: Path, entry_id: str) -> Dict:
    for row in load_file(path):
        if row["id"] == entry_id:
            return row
    raise KeyError(f"{entry_id!r} not found in {path.name}")


def _format_multi_turn(raw: Any, handler: _DummyHandler) -> List[List[List[str]]]:
    """
    Converts raw GPT-4o completion into  list[turn][step][call-str].
    Supported shapes:
        • str                 – whole conversation in one assistant message
        • list[str]           – one assistant reply per turn
        • list[list[str]]     – explicit turn × msg
    """
    if isinstance(raw, str):
        return [[handler.decode_execute(raw)]]

    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            return [
                [handler.decode_execute(msg) for msg in turn_msgs]
                for turn_msgs in raw
            ]
        else:  # list[str]   -> one msg per turn
            return [[handler.decode_execute(msg)] for msg in raw]

    raise TypeError(
        "`model_response` must be str | list[str] | list[list[str]] for multi-turn."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_single(
    category: str,
    model_response: Any,
    entry_id: str,
    model_name: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Grade ONE record (any category) using the official BFCL checkers but without
    running the big *generate → evaluate* pipeline.

    Returns the same dict that the full evaluator stores for a single sample.
    """

    if extract_test_category_from_id(entry_id) != category:
        raise ValueError("entry_id does not match category.")

    prompt_file  = PROMPT_PATH / TEST_FILE_MAPPING[category]
    answer_file  = (
        POSSIBLE_ANSWER_PATH / TEST_FILE_MAPPING[category]
        if not is_relevance_or_irrelevance(category)
        else None
    )

    prompt_row = _load_single(prompt_file, entry_id)
    answer_row = _load_single(answer_file, entry_id) if answer_file else None

    handler = _DummyHandler()

    # (Ir)Relevance  ─────────────────────────────────────────────────────── #
    if is_relevance_or_irrelevance(category):
        decoded = handler.decode_ast(model_response)
        has_call = (
            is_function_calling_format_output(decoded) and not is_empty_output(decoded)
        )
        success = (not has_call) if "irrelevance" in category else has_call
        return {
            "id": entry_id,
            "valid": success,
            "decoded": decoded,
            "error": None if success else "Wrong (ir)relevance decision.",
        }

    # Single-turn AST  ───────────────────────────────────────────────────── #
    if not is_multi_turn(category):
        lang = "Python"
        if is_java(category):
            lang = "Java"
        elif is_js(category):
            lang = "JavaScript"

        decoded = handler.decode_ast(model_response, language=lang)
        check = ast_checker(
            func_description = prompt_row["function"],
            model_output     = decoded,
            possible_answer  = answer_row["ground_truth"],
            language         = lang,
            test_category    = category,
            model_name       = model_name,
        )
        return {"id": entry_id, "decoded": decoded, **check}

    # Multi-turn  ────────────────────────────────────────────────────────── #
    decoded_mt = _format_multi_turn(model_response, handler)
    check = multi_turn_checker(
        multi_turn_model_result_list_decoded = decoded_mt,
        multi_turn_ground_truth_list         = answer_row["ground_truth"],
        test_entry   = prompt_row,
        test_category= category,
        model_name   = model_name,
    )
    return {"id": entry_id, **check}

def main():
    # Example usage
    # category = "multiple"
    # model_response= """[{"guitar_price_find": {"model": "Gibson Les Paul", "condition": "Excellent", "location": "Chicago"}}]"""

    # entry_id = "multiple_165"  # Example entry ID

    # result = evaluate_single(category, model_response, entry_id)
    # print(result)
#    multi-turn
    conversation = [
        '[{"MessageAPI.send_message": {"recipient":"Bob","body":"hi"}}]',
        '[{"MessageAPI.check_inbox": {}}]'
    ]
    print(
        evaluate_single(
            category       = "multi_turn_base",
            entry_id       = "multi_turn_base_7",
            model_response = conversation,
        )
    )



if __name__ == "__main__":
    main()
    print("Evaluation completed successfully.")