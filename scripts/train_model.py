from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_types import GrpoConfig
from scripts.get_data import (
    DEFAULT_SAVE_PATH,
    load_dafnybench_data,
    load_saved_dafnybench_data,
    save_dafnybench_data,
)
from src.trainer import CustomRLTrainer


def _prepare_prompts(split: str = "test") -> List[str]:
    """
    Ensure the DafnyBench split is available locally and return a list of
    prompts suitable for sampling by the trainer.
    """
    if DEFAULT_SAVE_PATH.exists():
        df = load_saved_dafnybench_data(DEFAULT_SAVE_PATH)
    else:
        df = load_dafnybench_data(split=split)
        save_dafnybench_data(df, DEFAULT_SAVE_PATH)

    if "body" not in df.columns:
        raise ValueError("Expected 'body' column to be present in DafnyBench data.")

    headers = df.get_column("header") if "header" in df.columns else None
    bodies = df.get_column("body")

    prompts: List[str] = []
    for idx in range(len(df)):
        header = headers[idx] if headers is not None else None
        body = bodies[idx]

        parts = [part for part in (header, body) if isinstance(part, str) and part.strip()]
        if not parts:
            continue
        prompts.append("\n".join(parts))

    if not prompts:
        raise ValueError("No usable prompts were extracted from DafnyBench.")

    return prompts


def main() -> None:
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Model loaded successfully")

    dataset = _prepare_prompts(split="test")

    config = GrpoConfig(
        batch_size=8,
        group_size=4,
        num_ppo_epochs=4,
        max_new_tokens=256,
        learning_rate=1e-6,
        advantage_whitening=True,
        mixed_precision=True,
    )

    trainer = CustomRLTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("Starting training...")

    trainer.train(dataset, num_steps=1, checkpoint_dir="checkpoints")


if __name__ == "__main__":
    main()
