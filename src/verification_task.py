import re
from dafny_file import DafnyFile, Dafny

SYSTEM_MESSAGE = """You are an LLM specialized in formal verification. 
Your task is to take existing Dafny code and produce a fully annotated version 
that enables the Dafny verifier to prove total correctness.

You MUST:
- Add assert statements, loop invariants, decreases clauses, pre/postconditions,
  frame specifications, and any auxiliary lemmas necessary for verification.
- Preserve the structure of the original program unless correctness requires a
  modification--at which point you should comment what was changed.
- Never delete user-provided logic unless it is provably dead or contradictory--
  if such a situation arises, comment out the code as opposed to deleting it.
- Produce ENTIRE updated Dafny file verbatim.

STRICT FORMAT:
You MUST think inside <think>...</think> tags.
Your final output MUST appear ONLY inside <answer>...</answer> tags.
No commentary, no explanations, no meta text in the final answer.

<think> is hidden from the user in deployment but will be visible during training.
Inside <think> you must:
- reason step-by-step to infer required invariants or assertions,
- identify verification barriers,
- plan modifications that guarantee termination and correctness.

Inside <answer> you must:
- output the full final Dafny code file, fully annotated and syntactically valid.

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

Return the entire resulting file.

Follow the system instructions exactly.
Use <think> for chain-of-thought and <answer> for the final code.

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
    full_format = r"<think>(.*?)</think>\n<answer>(.*?)</answer>$"

    match_full = re.search(full_format, response, re.DOTALL)

    if match_full:
        return 1.0

    match_think = re.search(think_regex, response, re.DOTALL)
    match_answer = re.search(answer_regex, response, re.DOTALL)

    reward = 0.0
    if match_think:
        reward += 0.1
    if match_answer:
        reward += 0.7
    return reward


def get_generated_dafny_code(response: str) -> DafnyFile:
    """Extracts the generated Dafny code from the LLM response."""
    answer_regex = r"<answer>(.*?)</answer>"
    match_answer = re.search(answer_regex, response, re.DOTALL)
    if not match_answer:
        raise ValueError("No <answer> tags found in the response.")

    generated_code = match_answer.group(1).strip()
    return DafnyFile.from_code(generated_code)


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
