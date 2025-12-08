import json
import sys
from pathlib import Path
from typing import Mapping, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.annotation_patch import apply_edits, patch_json_schema, try_parse_edits  # type: ignore
from src.dafny_file import DafnyFile, Dafny  # type: ignore

ANNOTATION_PATCH_SCHEMA = patch_json_schema()
ANNOTATION_PATCH_SCHEMA_JSON = json.dumps(ANNOTATION_PATCH_SCHEMA, indent=2)

RL_SYSTEM_MESSAGE = (
    "You revise Dafny programs by ADDING ONLY annotations (ghost code, specs, invariants). "
    "Respond ONLY with JSON matching the provided schema."
)
RL_USER_TEMPLATE = (
    "Add the minimal Dafny annotations needed for verification.\n"
    "Return ONLY a JSON array that matches this schema:\n"
    "{annotation_schema}\n\n"
    "IMPORTANT:\n"
    "- Use 1-based line numbers from the ORIGINAL code.\n"
    "- Only insert; do NOT delete or rewrite existing lines.\n"
    "- `position` must be `insert_before` or `insert_after` relative to the anchor line.\n"
    "- Keep indentation consistent; set `indent` false only if you supply your own whitespace.\n"
    "- No prose, no code fences, no explanations outside the JSON.\n\n"
    "PROGRAM:\n"
    "{dafny_code_snippet}\n"
)

SFT_SYSTEM_MESSAGE = "You produce verified Dafny programs. Reply with Dafny code only."
SFT_USER_TEMPLATE = (
    "Complete this Dafny implementation so it verifies:\n"
    "{dafny_body}\n"
    "Output only the full Dafny code."
)

FORMAT_WEIGHT = 0.1
COMPILE_WEIGHT = 2.0
VERIFICATION_WEIGHT = 3.0
ASSUMTION_WEIGHT = 1.0
DELETION_WEIGHT = 1.0


def get_generated_dafny_code(response: str, original_code: str) -> DafnyFile:
    """
    Interpret model output as either a JSON patch or full Dafny code.
    """
    payload = response.strip()
    if not payload:
        raise ValueError("Model response was empty; expected Dafny code or patch.")

    edits = try_parse_edits(payload)
    if edits is not None:
        updated = apply_edits(original_code, edits)
        return DafnyFile.from_code(updated)

    return DafnyFile.from_code(payload)


def format_reward_function(response: str) -> float:
    """Reward 1.0 when the output looks like raw Dafny code with no meta markers."""
    stripped = response.strip()
    if not stripped:
        return 0.0
    if try_parse_edits(stripped) is not None:
        return 0.1
    forbidden_markers = ("<think", "<answer", "```")
    return 0.0 if any(marker in stripped for marker in forbidden_markers) else 0.1


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

def compile_reward_function(dafny_file: DafnyFile, dafny: Dafny) -> float:
    """Reward function for whether the modified Dafny code compiles."""
    compiles = dafny.compile_no_verify(dafny_file)
    if compiles:
        return 1.0
    return 0.0

def verification_reward_function(dafny_file: DafnyFile, dafny: Dafny) -> float:
    """Reward function for whether the modified Dafny code verifies."""
    verifies = dafny.verify(dafny_file)
    if verifies:
        return 1.0
    return 0.0
