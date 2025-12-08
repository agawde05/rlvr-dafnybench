from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from dafny_file import Dafny, DafnyFile
from verification_task import (
    FORMAT_WEIGHT,
    ASSUMTION_WEIGHT,
    DELETION_WEIGHT,
    COMPILE_WEIGHT,
    VERIFICATION_WEIGHT,
    compile_reward_function,
    verification_reward_function,
    get_generated_dafny_code,
    format_reward_function,
    assume_reward_function,
    deletion_reward_function,
)

RewardFn = Callable[[str, str, Dict[str, Any]], Tuple[float, Dict[str, Any]]]


def build_reward_function(dafny_binary: Path) -> RewardFn:
    """
    Build a reward function that scores generations using Dafny verification
    along with auxiliary shaping terms defined in verification_task.
    """

    dafny = Dafny(dafny_binary)

    def reward_fn(
        prompt: str, response: str, metadata: Dict[str, Any], end_token: int = 0
    ) -> Tuple[float, Dict[str, Any]]:
        original_code = metadata.get("original_code", prompt)
        # print("completion: ", completion)
        # print("original_code: ", original_code)
        try:
            dafny_file: DafnyFile = get_generated_dafny_code(response, original_code)
        except ValueError as exc:
            print(f"Error interpreting Dafny output: {exc}")
            # Invalid code → no verification, but return formatting score.
            format_score = format_reward_function(response)
            components = {
                "compilation": 0.0,
                "verification": 0.0,
                "format": float(format_score),
                "assume": 0.0,
                "deletion": 0.0,
            }
            return 0.0, components

        generated_code = dafny_file.get_code() or ""
        original_code = metadata.get("original_code", prompt)

        # first check if it is formatted
        format_score = format_reward_function(response)
        if format_score < 0.1:
            components = {
                "compilation": 0.0,
                "verification": 0.0,
                "format": float(format_score),
                "assume": 0.0,
                "deletion": 0.0,
            }
            return 0.0, components

        assume_score = assume_reward_function(original_code, generated_code)
        print(f"Assume score: {assume_score}")

        deletion_score = deletion_reward_function(original_code, generated_code)
        print(f"Deletion score: {deletion_score}")

        compile_score = compile_reward_function(dafny_file, dafny)
        print(f"Compile score: {compile_score}")
        if compile_score < 1.0:
            components: Dict[str, Any] = {
                "compilation": float(compile_score),
                "verification": 0.0,
                "format": float(format_score),
                "assume": float(assume_score),
                "deletion": float(deletion_score),
            }

        verification_score = verification_reward_function(dafny_file, dafny)
        print(f"Verification score: {verification_score}")
        # format_score = format_reward_function(completion)
        # print(f"Format score: {format_score}")
        components: Dict[str, Any] = {
            "compilation": float(compile_score),
            "verification": float(verification_score),
            "format": float(format_score),
            "assume": float(assume_score),
            "deletion": float(deletion_score),
        }

        # Main scalar reward is verification; other components are logged for analysis.
        return (
            FORMAT_WEIGHT * format_score
            + COMPILE_WEIGHT * compile_score
            + VERIFICATION_WEIGHT * verification_score
            + ASSUMTION_WEIGHT * assume_score
            + DELETION_WEIGHT * deletion_score,
            components,
        )

    return reward_fn
