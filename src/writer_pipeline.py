"""
Utilities for training and evaluating the writer (unimplemented → unannotated) pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import polars as pl
import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from dafny_file import Dafny, DafnyFile
from verification_task import SFT_SYSTEM_MESSAGE, SFT_USER_TEMPLATE

RewardFn = Any  # Alias to avoid circular import with trainer


# --------------------------------------------------------------------------- #
# Data Structures
# --------------------------------------------------------------------------- #
@dataclass
class WriterGenerationConfig:
    """Sampling controls for writer inference."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

    def to_kwargs(self) -> Dict[str, float | int | bool]:
        kwargs: Dict[str, float | int | bool] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
            kwargs["do_sample"] = True
        else:
            kwargs["temperature"] = 1.0
            kwargs["do_sample"] = False
        return kwargs


@dataclass
class WriterExample:
    """Single DafnyBench record tailored for the writer pipeline."""

    id: str
    stub: str
    implementation: str
    header: Optional[str] = None
    spec: Optional[str] = None

    def prompt(self) -> str:
        """Base prompt string used before trainer-level chat formatting."""
        parts: List[str] = []
        if self.spec:
            parts.append(f"// spec: {self.spec.strip()}")
        if self.header:
            parts.append(self.header.strip())
        parts.append(self.stub.strip())
        return "\n".join(part for part in parts if part)


def format_writer_training_prompt(example: WriterExample) -> str:
    """Instruction-following prompt for the writer model."""
    base = example.prompt()
    return f"{WRITER_INSTRUCTION_PREFIX}\n\n{base}"


def build_writer_examples(df: pl.DataFrame) -> List[WriterExample]:
    """Convert a DafnyBench dataframe into writer-ready examples."""
    required = ("unimplemented_body", "body")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"DafnyBench dataframe is missing columns: {missing}")

    ids = df.get_column("id") if "id" in df.columns else None
    headers = df.get_column("header") if "header" in df.columns else None
    specs = df.get_column("spec") if "spec" in df.columns else None
    stubs = df.get_column("unimplemented_body")
    impls = df.get_column("body")

    examples: List[WriterExample] = []
    for idx in range(len(df)):
        stub = stubs[idx]
        impl = impls[idx]
        if not isinstance(stub, str) or not isinstance(impl, str):
            continue
        stub_text = stub.strip()
        impl_text = impl.strip()
        if not stub_text or not impl_text:
            continue
        example_id = str(ids[idx]) if ids is not None else str(idx)
        header = headers[idx] if headers is not None else None
        spec = specs[idx] if specs is not None else None
        examples.append(
            WriterExample(
                id=example_id,
                stub=stub_text,
                implementation=impl_text,
                header=header.strip() if isinstance(header, str) else None,
                spec=spec.strip() if isinstance(spec, str) else None,
            )
        )

    return examples


def writer_examples_to_sft_pairs(examples: Iterable[WriterExample]) -> List[Dict[str, str]]:
    """Produce dictionaries shaped for `CustomRLTrainer.supervised_fine_tune`."""
    return [
        {"body": example.stub, "annotated_body": example.implementation}
        for example in examples
    ]


def writer_examples_to_dataset(examples: Iterable[WriterExample]) -> List[Dict[str, Any]]:
    """
    Produce dataset entries for RL training.

    Each entry provides the text prompt plus metadata (the originating example).
    """
    dataset: List[Dict[str, Any]] = []
    for example in examples:
        dataset.append(
            {"prompt": format_writer_training_prompt(example), "example": example}
        )
    return dataset


def writer_examples_to_prompts(examples: Iterable[WriterExample]) -> List[str]:
    """Compatibility helper returning raw prompt strings."""
    return [format_writer_training_prompt(example) for example in examples]


# --------------------------------------------------------------------------- #
# Annotator Interface
# --------------------------------------------------------------------------- #
def _format_chat_prompt(
    tokenizer: Optional[PreTrainedTokenizerBase], system_text: str, user_text: str
) -> str:
    system_clean = system_text.strip()
    user_clean = user_text.strip()

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_clean},
            {"role": "user", "content": user_clean},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            pass

    return f"{system_clean}\n\n{user_clean}"


def _stitch_header_body(header: Optional[str], body: str) -> str:
    if header:
        return f"{header.rstrip()}\n{body.lstrip()}"
    return body


def format_annotator_prompt(
    tokenizer: Optional[PreTrainedTokenizerBase],
    example: WriterExample,
    writer_body: str,
) -> str:
    pieces: List[str] = []
    if example.spec:
        pieces.append(f"// spec: {example.spec.strip()}")
    pieces.append(_stitch_header_body(example.header, writer_body))
    dafny_body = "\n".join(part for part in pieces if part)
    user_text = SFT_USER_TEMPLATE.format(dafny_body=dafny_body)
    return _format_chat_prompt(tokenizer, SFT_SYSTEM_MESSAGE, user_text)


class AnnotatorPipeline:
    """Thin inference wrapper over an annotator model/tokenizer pair."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def annotate(self, example: WriterExample, writer_body: str) -> str:
        prompt = format_annotator_prompt(self.tokenizer, example, writer_body)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        else:
            generation_kwargs["temperature"] = 1.0

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                **generation_kwargs,
            )

        generated = output[0][encoded["input_ids"].shape[1] :]
        annotated = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not annotated:
            raise RuntimeError("Annotator model produced empty output.")
        return annotated


# --------------------------------------------------------------------------- #
# Writer Reward
# --------------------------------------------------------------------------- #
def build_writer_reward(
    annotator: AnnotatorPipeline,
    dafny: Optional[Dafny] = None,
    dafny_path: Optional[Path] = None,
) -> RewardFn:
    """
    Construct a reward function that scores writer outputs by cascading through
    the annotator and verifier.
    """
    if dafny is None:
        if dafny_path is None:
            raise ValueError("Either `dafny` or `dafny_path` must be provided.")
        dafny = Dafny(dafny_path)

    def _reward(prompt: str, completion: str, metadata: Mapping[str, Any]):
        example = metadata.get("example")
        if not isinstance(example, WriterExample):
            return 0.0, {"error": "missing_example"}

        writer_body = completion.strip()
        if not writer_body:
            return 0.0, {"error": "empty_writer_output"}

        try:
            annotated = annotator.annotate(example, writer_body)
        except Exception as exc:  # pragma: no cover - handled in tests via stub
            return 0.0, {"error": f"annotator_failed: {exc}"}

        full_program = _stitch_header_body(example.header, annotated)

        try:
            dafny_file = DafnyFile.from_code(full_program)
            verified = dafny.verify(dafny_file)
        except Exception as exc:  # pragma: no cover
            return 0.0, {"error": f"dafny_failed: {exc}", "annotated_code": annotated}

        reward = 1.0 if verified else 0.0
        components = {
            "verified": bool(verified),
            "annotated_code": annotated,
        }
        return reward, components

    return _reward


# --------------------------------------------------------------------------- #
# Writer Inference
# --------------------------------------------------------------------------- #
def generate_implementation(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    example: WriterExample,
    gen_cfg: Optional[WriterGenerationConfig] = None,
    device: Optional[torch.device] = None,
) -> str:
    """Generate an unannotated implementation for a given stub."""
    prompt = example.prompt()
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        target_device = device or next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        target_device = torch.device("cpu")
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    cfg = gen_cfg or WriterGenerationConfig()
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            **cfg.to_kwargs(),
        )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


__all__ = [
    "AnnotatorPipeline",
    "WriterExample",
    "WriterGenerationConfig",
    "build_writer_examples",
    "build_writer_reward",
    "format_annotator_prompt",
    "format_writer_training_prompt",
    "generate_implementation",
    "writer_examples_to_dataset",
    "writer_examples_to_prompts",
    "writer_examples_to_sft_pairs",
]
WRITER_INSTRUCTION_PREFIX = (
    "Fill in the missing Dafny method bodies below. Only produce Dafny code."
)
