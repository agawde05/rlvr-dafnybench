from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.reward import build_verification_reward

from src.data_types import GrpoConfig
from scripts.get_data import (
    DEFAULT_SAVE_PATH,
    load_dafnybench_data,
    load_saved_dafnybench_data,
    save_dafnybench_data,
)
from src.trainer import CustomRLTrainer


def _load_or_prepare_dataframe(split: str = "test"):
    """
    Ensure the DafnyBench split is available locally and return the dataframe.
    """
    if DEFAULT_SAVE_PATH.exists():
        return load_saved_dafnybench_data(DEFAULT_SAVE_PATH)

    df = load_dafnybench_data(split=split)
    save_dafnybench_data(df, DEFAULT_SAVE_PATH)
    return df


def _prepare_prompts(df) -> List[str]:
    """
    Convert the DafnyBench dataframe into prompt strings suitable for RL.
    """
    if "body" not in df.columns:
        raise ValueError("Expected 'body' column to be present in DafnyBench data.")

    headers = df.get_column("header") if "header" in df.columns else None
    bodies = df.get_column("body")
    annotated_bodies = df.get_column("annotated_body")

    prompts: List[str] = []
    for idx in range(len(df)):
        header = headers[idx] if headers is not None else None
        body = bodies[idx]
        annotated_body = annotated_bodies[idx]

        parts = [part for part in (header, body, annotated_body) if isinstance(part, str) and part.strip()]
        if not parts:
            continue
        prompts.append("\n".join(parts))

    if not prompts:
        raise ValueError("No usable prompts were extracted from DafnyBench.")

    return prompts


def _prepare_supervised_pairs(df) -> List[Dict[str, str]]:
    """
    Produce (body, annotated_body) pairs for supervised fine-tuning.
    """
    if "body" not in df.columns or "annotated_body" not in df.columns:
        raise ValueError("DafnyBench data is missing required columns for supervised fine-tuning.")

    bodies = df.get_column("body")
    annotated_bodies = df.get_column("annotated_body")

    pairs: List[Dict[str, str]] = []
    for idx in range(len(df)):
        body = bodies[idx]
        annotated = annotated_bodies[idx]
        if isinstance(body, str) and isinstance(annotated, str):
            body_text = body.strip()
            annotated_text = annotated.strip()
            if body_text and annotated_text:
                pairs.append({"body": body_text, "annotated_body": annotated_text})

    if not pairs:
        raise ValueError("No valid supervised fine-tuning pairs were found in DafnyBench.")

    return pairs


def _prepare_writer_pairs(df) -> List[Dict[str, str]]:
    """
    Produce (unimplemented_body, body) pairs for the task-writer codegen pipeline.
    """
    required = ("unimplemented_body", "body")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"DafnyBench data is missing required columns for writer training: {missing}"
        )

    unimplemented_bodies = df.get_column("unimplemented_body")
    bodies = df.get_column("body")

    pairs: List[Dict[str, str]] = []
    for idx in range(len(df)):
        stub = unimplemented_bodies[idx]
        body = bodies[idx]
        if isinstance(stub, str) and isinstance(body, str):
            stub_text = stub.strip()
            body_text = body.strip()
            if stub_text and body_text:
                pairs.append(
                    {"unimplemented_body": stub_text, "body": body_text}
                )

    if not pairs:
        raise ValueError("No valid writer training pairs were found in DafnyBench.")

    return pairs


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    reward_fn = build_verification_reward(Path("dafny"))

    print("Model loaded successfully")

    df = _load_or_prepare_dataframe(split="test")
    dataset = _prepare_prompts(df)
    supervised_pairs = _prepare_supervised_pairs(df)

    config = GrpoConfig(
        batch_size=8,
        microbatch_size=2,
        group_size=4,
        max_new_tokens=256,
        learning_rate=1e-5,
        advantage_whitening=True,
        mixed_precision=True,
    )

    trainer = CustomRLTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        config=config,
        device=device,
    )

    print("Starting supervised fine-tuning warmup...")
    sft_metrics = trainer.supervised_fine_tune(
        supervised_pairs,
        epochs=1,
        batch_size=config.batch_size,
    )
    print(f"Supervised fine-tuning metrics: {sft_metrics}")

    print("Starting training...")

    trainer.train(dataset, num_steps=10, checkpoint_dir="checkpoints")


if __name__ == "__main__":
    main()
