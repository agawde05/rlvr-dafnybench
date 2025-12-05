import re
from typing import Mapping, Any

from dafny_file import DafnyFile, Dafny
from diff_merger import DiffMergeError, merge_diff, validate_diff_json

SYSTEM_MESSAGE = """You are a formal verification specialist. Given Dafny source, produce the minimal annotations required for Dafny to prove total correctness.

Requirements:
- Add or strengthen asserts, loop invariants, decreases clauses, pre/postconditions, frame specifications, and auxiliary lemmas whenever needed.
- Preserve original structure unless correctness forces a change; if code is removed, leave an equivalent commented placeholder.
- Emit only a structured diff of your edits, never the full file.

Protocol:
- Reason exclusively inside <think>...</think>.
- Emit the final diff exclusively inside <answer>...</answer>.
- <answer> must be valid JSON of the form {"hunks": [...]}, using 1-indexed original_start/original_length/patched_start/patched_length and a lines array of objects with keys type ("context"|"remove"|"add") and text without trailing newlines.
- Output {"hunks": []} if no changes are needed.
- Do not add other keys, markdown, or prose outside the required tags.

Your output must be deterministic and complete. Do not use code blocks or markdown.
"""

USER_TEMPLATE = """Verify the Dafny code below and add every annotation needed for total correctness.

Include any missing requires/ensures, asserts, loop invariants, decreases clauses, modifies/frame specifications, and auxiliary lemmas only when required. Do not introduce new assume statements or delete code without a commented equivalent.

Return only the diff JSON that matches the system schema, with reasoning in <think> and the diff in <answer>.

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
