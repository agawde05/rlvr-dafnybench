#!/usr/bin/env python
"""
Generate Dafny implementations from unimplemented stubs using a trained writer model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.writer_pipeline import WriterExample, WriterGenerationConfig, generate_implementation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run writer inference on a stub.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--stub-file", type=Path, required=True)
    parser.add_argument("--header-file", type=Path, default=None)
    parser.add_argument("--spec-file", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def _read_optional(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text()


def main() -> None:
    args = parse_args()
    stub = args.stub_file.read_text()
    header = _read_optional(args.header_file)
    spec = _read_optional(args.spec_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    example = WriterExample(
        id="cli",
        stub=stub,
        implementation="",
        header=header,
        spec=spec,
    )
    gen_cfg = WriterGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    output = generate_implementation(model, tokenizer, example, gen_cfg)
    print(output)


if __name__ == "__main__":
    main()
