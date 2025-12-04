import re
from typing import Mapping, Any

from dafny_file import DafnyFile, Dafny
from diff_merger import DiffMergeError, merge_diff, validate_diff_json

SYSTEM_MESSAGE = """Role: Dafny verification expert.
Goal: supply the minimal edits needed so the program proves while preserving intent.

Required actions:
- Add/adjust requires, ensures, invariants, decreases, modifies, asserts, helper lemmas.
- Do not delete user logic; if something must go, comment why.
- Always return edits as JSON diff hunks (format below), never the full file.

Formatting:
- Think inside <think>...</think> (training only).
- Final output lives entirely inside <answer>...</answer>.
- JSON schema (no extra keys, 1-indexed line numbers, no trailing newline chars):
  {"hunks":[{"original_start":int,"original_length":int,"patched_start":int,"patched_length":int,"lines":[{"type":"context|remove|add","text":"..."}]}]}
- Emit {"hunks": []} if nothing changes.
- Absolutely no markdown, fences, or stray prose.

Be deterministic and ensure the resulting Dafny verifies."""

USER_TEMPLATE = """Below is Dafny code that needs verification annotations.

Add the specs/assertions/lemmas required for proof while keeping user intent.
Return only the JSON diff described in the system message.

Follow the format (<think>, then <answer> with the diff JSON) exactly.

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
