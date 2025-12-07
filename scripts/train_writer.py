#!/usr/bin/env python
"""
Train the writer pipeline (unimplemented → implementation) with minimal wiring.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.get_data import (
    DEFAULT_SAVE_PATH,
    load_dafnybench_data,
    load_saved_dafnybench_data,
    save_dafnybench_data,
)
from src.data_types import GrpoConfig
from src.trainer import CustomRLTrainer
from src.writer_pipeline import (
    AnnotatorPipeline,
    WriterExample,
    build_writer_examples,
    build_writer_reward,
    writer_examples_to_dataset,
    writer_examples_to_sft_pairs,
)


def _load_dataframe(split: str, limit: Optional[int]):
    if DEFAULT_SAVE_PATH.exists():
        df = load_saved_dafnybench_data(DEFAULT_SAVE_PATH)
    else:
        df = load_dafnybench_data(split=split)
        save_dafnybench_data(df, DEFAULT_SAVE_PATH)
    if limit is not None:
        df = df.head(limit)
    return df


def _filter_examples(
    examples: Sequence[WriterExample],
    max_chars: Optional[int],
) -> list[WriterExample]:
    if max_chars is None:
        return list(examples)

    filtered: list[WriterExample] = []
    for example in examples:
        total = len(example.stub) + len(example.implementation)
        if total <= max_chars:
            filtered.append(example)
    return filtered


def _prepare_examples(
    split: str,
    limit: Optional[int],
    max_chars: Optional[int],
) -> list[WriterExample]:
    df = _load_dataframe(split, limit)
    raw_examples = build_writer_examples(df)
    examples = _filter_examples(raw_examples, max_chars)
    if examples and len(examples) < len(raw_examples):
        print(
            f"[writer-train] Filtered {len(raw_examples) - len(examples)} "
            f"examples over the {max_chars}-char budget."
        )
    return examples


def _save_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    else:
        torch.save(model.state_dict(), output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the writer pipeline.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Drop training examples whose stub+implementation exceed this char count.",
    )
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rl-steps", type=int, default=0)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints") / "writer",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models") / "writer",
        help="Directory where the fine-tuned writer model/tokenizer will be saved.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print dataset stats without initializing a model.",
    )
    parser.add_argument(
        "--annotator-model",
        type=str,
        default=None,
        help="Optional annotator model path for RL cascade rewards.",
    )
    parser.add_argument(
        "--annotator-max-new-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dafny-path",
        type=Path,
        default=None,
        help="Path to the Dafny CLI executable used for verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = _prepare_examples(args.split, args.max_examples, args.max_chars)
    if not examples:
        raise RuntimeError("No writer examples were prepared from DafnyBench.")

    print(f"[writer-train] Loaded {len(examples)} examples from split={args.split}.")
    if args.dry_run:
        print("[writer-train] Dry run complete.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=model_dtype).to(device)

    reward_fn = None
    annotator_pipeline = None
    if args.rl_steps > 0:
        if not args.annotator_model:
            raise ValueError("--annotator-model is required when --rl-steps > 0")
        if args.dafny_path is None:
            raise ValueError("--dafny-path is required when --rl-steps > 0")
        annotator_tokenizer = AutoTokenizer.from_pretrained(args.annotator_model)
        annotator_model = AutoModelForCausalLM.from_pretrained(
            args.annotator_model,
            dtype=model_dtype,
        ).to(device)
        annotator_pipeline = AnnotatorPipeline(
            model=annotator_model,
            tokenizer=annotator_tokenizer,
            device=device,
            max_new_tokens=args.annotator_max_new_tokens,
        )
        reward_fn = build_writer_reward(
            annotator=annotator_pipeline,
            dafny_path=args.dafny_path,
        )

    trainer = CustomRLTrainer(
        model=model,
        tokenizer=tokenizer,
        config=GrpoConfig(batch_size=args.batch_size),
        reward_fn=reward_fn,
        device=device,
    )

    sft_pairs = writer_examples_to_sft_pairs(examples)
    metrics = trainer.supervised_fine_tune(
        sft_pairs,
        epochs=args.sft_epochs,
        batch_size=args.batch_size,
    )
    print(f"[writer-train] SFT metrics: {metrics}")

    if args.rl_steps > 0:
        dataset = writer_examples_to_dataset(examples)
        checkpoint_dir: Optional[str] = None
        if args.checkpoint_dir:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = str(args.checkpoint_dir)
        trainer.train(
            dataset,
            num_steps=args.rl_steps,
            checkpoint_dir=checkpoint_dir,
        )
    print(f"[writer-train] Saving model to {args.output_dir} ...")
    _save_model(trainer.policy_model, tokenizer, args.output_dir)
    print("[writer-train] Done.")


if __name__ == "__main__":
    main()
