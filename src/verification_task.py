import re
from typing import Mapping, Any

from dafny_file import DafnyFile, Dafny
from diff_merger import DiffMergeError, merge_diff, validate_diff_json

SYSTEM_MESSAGE = """You are an LLM specialized in formal verification. 
Your task is to take existing Dafny code and produce the minimal, fully annotated
changes required for the Dafny verifier to prove total correctness.

You MUST:
- Add assert statements, loop invariants, decreases clauses, pre/postconditions,
  frame specifications, and any auxiliary lemmas necessary for verification.
- Preserve the structure of the original program unless correctness requires a
  modification. Justify removals by replacing them with equivalent commented code.
- Return your modifications as a structured diff rather than the full file.

STRICT FORMAT:
You MUST think inside <think>...</think> tags.
Your final output MUST appear ONLY inside <answer>...</answer> tags.
No commentary, no explanations, no meta text outside the required tags.

ABSOLUTE RULES FOR <answer>:
- The content inside <answer> MUST be valid JSON describing Git-style hunks:
  {
    "hunks": [
      {
        "original_start": 12,
        "original_length": 3,
        "patched_start": 12,
        "patched_length": 4,
        "lines": [
          {"type": "context", "text": "unchanged line"},
          {"type": "remove", "text": "old line"},
          {"type": "add", "text": "new line"}
        ]
      }
    ]
  }
- Line numbers are 1-indexed. `text` entries MUST omit trailing newline characters.
- Only the keys shown above are permitted; do not include summaries or comments.
- If no changes are required, output {"hunks": []}.
- NO markdown. NO ``` fences. NO natural language. NO comments.

<think> is hidden from the user in deployment but will be visible during training.
Inside <think> you must:
- reason step-by-step to infer required invariants or assertions,
- identify verification barriers,
- plan modifications that guarantee termination and correctness.

Inside <answer> you must:
- output the JSON diff adhering strictly to the schema above.

All invariants must be strong enough for Dafny to verify.
All modifications must preserve semantic meaning.

Your output MUST be deterministic and complete.
"""

USER_TEMPLATE = """Below is Dafny code that needs verification annotations.

Insert all necessary:
- requires/ensures
- assert statements
- loop invariants
- decreases clauses
- modifies/frame specifications
- auxiliary lemmas (only if needed)
- datatype or function constraints

Return ONLY the JSON diff describing the changes, following the system schema.

Follow the system instructions exactly.
Use <think> for chain-of-thought and <answer> for the final JSON diff.

----- BEGIN INPUT CODE -----
{dafny_code_snippet}
----- END INPUT CODE -----

<think>
"""

FORMAT_WEIGHT = 0.1
VERIFICATION_WEIGHT = 1.0
ASSUMTION_WEIGHT = 1.0
DELETION_WEIGHT = 1.0


def format_reward_function(response: str) -> float:
    """The reward function for whether the LLM output is correctly formatted.

    Returns a float between 0.0 and 1.0, where 1.0 indicates perfect formatting.
    """
    think_regex = r"<think>(.*?)</think>"
    answer_regex = r"<answer>(.*?)</answer>"
    full_format = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>$"

    match_think = re.search(think_regex, response, re.DOTALL)
    match_answer = re.search(answer_regex, response, re.DOTALL)
    match_full = re.search(full_format, response, re.DOTALL)

    reward = 0.0
    if match_think:
        reward += 0.2
    if match_answer:
        reward += 0.3

    if not match_answer:
        return reward

    diff_blob = match_answer.group(1).strip()
    if not diff_blob:
        return reward

    if validate_diff_json(diff_blob):
        return 1.0

    # Provide partial credit when structure begins correctly.
    if match_full:
        reward += 0.2
    return min(reward, 0.9)


def get_generated_dafny_code(response: str, original_code: str) -> DafnyFile:
    """Extracts Dafny code from a JSON diff by merging with `original_code`."""
    answer_regex = r"<answer>(.*?)</answer>"
    match_answer = re.search(answer_regex, response, re.DOTALL)
    if not match_answer:
        raise ValueError("No <answer> tags found in the response.")

    diff_blob = match_answer.group(1).strip()
    if not diff_blob:
        raise ValueError("Diff payload inside <answer> is empty.")

    try:
        merged_code = merge_diff(original_code, diff_blob)
    except DiffMergeError as exc:
        raise ValueError(f"Failed to merge diff JSON: {exc}") from exc

    return DafnyFile.from_code(merged_code)


def assume_reward_function(original_code: str, modified_code: str) -> float:
    """Reward function for whether no new 'assume' statements were introduced."""
    no_additional_assume = DafnyFile.validate_no_assume(original_code, modified_code)
    if no_additional_assume:
        return 0.2
    return -1.0


def deletion_reward_function(original_code: str, modified_code: str) -> float:
    """Reward function for whether code deletion occurred."""
    no_deletion = DafnyFile.validate_no_deletion(original_code, modified_code)
    if no_deletion:
        return 0.2
    return -1.0


def verification_reward_function(dafny_file: DafnyFile, dafny: Dafny) -> float:
    """Reward function for whether the modified Dafny code verifies."""
    verifies = dafny.verify(dafny_file)
    if verifies:
        return 1.0
    return 0.0
