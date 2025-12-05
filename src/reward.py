from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from src.dafny_file import Dafny, DafnyFile
from src.verification_task import (
    verification_reward_function,
    get_generated_dafny_code,
    format_reward_function,
    assume_reward_function,
    deletion_reward_function,
)

RewardFn = Callable[[str, str, Dict[str, Any]], Tuple[float, Dict[str, Any]]]


def build_verification_reward(dafny_binary: Path) -> RewardFn:
    """
    Build a reward function that scores generations using Dafny verification
    along with auxiliary shaping terms defined in verification_task.
    """

    dafny = Dafny(dafny_binary)

    def reward_fn(
        prompt: str, completion: str, metadata: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        original_code = metadata.get("original_code", prompt)
        print("original_code: ", original_code)
        try:
            dafny_file: DafnyFile = get_generated_dafny_code(completion, original_code)
        except ValueError:
            print(f"Error: Missing <answer> tags in completion")
            # Missing <answer> tags → no verification, but return formatting score.
            format_score = format_reward_function(completion)
            components = {
                "verification": 0.0,
                "format": float(format_score),
                "assume": 0.0,
                "deletion": 0.0,
            }
            return 0.0, components

        generated_code = dafny_file.get_code() or ""
        original_code = metadata.get("original_code", prompt)
        verification_score = verification_reward_function(dafny_file, dafny)
        print(f"Verification score: {verification_score}")
        format_score = format_reward_function(completion)
        print(f"Format score: {format_score}")
        assume_score = assume_reward_function(original_code, generated_code)
        print(f"Assume score: {assume_score}")
        deletion_score = deletion_reward_function(original_code, generated_code)
        print(f"Deletion score: {deletion_score}")
        components: Dict[str, Any] = {
            "verification": float(verification_score),
            "format": float(format_score),
            "assume": float(assume_score),
            "deletion": float(deletion_score),
        }

        # Main scalar reward is verification; other components are logged for analysis.
        return float(verification_score), components

    return reward_fn

