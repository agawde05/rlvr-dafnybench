from typing import Mapping, Any

from dafny_file import DafnyFile, Dafny

RL_SYSTEM_MESSAGE = "You revise Dafny programs. Respond with Dafny code only."
RL_USER_TEMPLATE = (
    "Update the program below so Dafny verifies it.\n"
    "{dafny_code_snippet}\n"
    "Return the complete Dafny code and nothing else."
)

SFT_SYSTEM_MESSAGE = "You produce verified Dafny programs. Reply with Dafny code only."
SFT_USER_TEMPLATE = (
    "Complete this Dafny implementation so it verifies:\n"
    "{dafny_body}\n"
    "Output only the full Dafny code."
)

FORMAT_WEIGHT = 0.1
VERIFICATION_WEIGHT = 1.0
ASSUMTION_WEIGHT = 1.0
DELETION_WEIGHT = 1.0


def format_reward_function(response: str) -> float:
    """Reward 1.0 when the output looks like raw Dafny code with no meta markers."""
    stripped = response.strip()
    if not stripped:
        return 0.0
    forbidden_markers = ("<think", "<answer", "```")
    if any(marker in stripped for marker in forbidden_markers):
        return 0.0
    return 0.1


def get_generated_dafny_code(response: str, original_code: str) -> DafnyFile:
    """Interpret the raw model response as Dafny code."""
    code = response.strip()
    if not code:
        raise ValueError("Model response was empty; expected Dafny code.")
    return DafnyFile.from_code(code)


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
