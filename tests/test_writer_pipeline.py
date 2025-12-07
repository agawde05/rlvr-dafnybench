"""
Lightweight tests for writer pipeline helpers.
"""
from __future__ import annotations

import torch

from src.writer_pipeline import (
    WriterExample,
    WriterGenerationConfig,
    build_writer_examples,
    build_writer_reward,
    format_annotator_prompt,
    format_writer_training_prompt,
    generate_implementation,
    writer_examples_to_dataset,
    writer_examples_to_prompts,
    writer_examples_to_sft_pairs,
)


def test_build_writer_examples_filters_rows():
    import polars as pl

    df = pl.DataFrame(
        {
            "id": ["a", "b"],
            "unimplemented_body": ["method Foo();", None],
            "body": ["method Foo() { }", "  "],
            "header": ["class C {}", None],
            "spec": [None, "desc"],
        }
    )
    examples = build_writer_examples(df)
    assert len(examples) == 1
    assert examples[0].id == "a"
    assert examples[0].prompt() == "class C {}\nmethod Foo();"


def test_writer_pairs_and_prompts_formatting():
    examples = [
        WriterExample(id="1", stub="method Foo();", implementation="method Foo() { }"),
        WriterExample(id="2", stub="method Bar();", implementation="method Bar() { }", spec="desc"),
    ]
    prompts = writer_examples_to_prompts(examples)
    pairs = writer_examples_to_sft_pairs(examples)
    assert "Fill in the missing Dafny method bodies" in prompts[0]
    assert "// spec: desc" in prompts[1]
    assert pairs[0] == {"body": "method Foo();", "annotated_body": "method Foo() { }"}


def test_writer_dataset_includes_examples():
    examples = [
        WriterExample(id="1", stub="method Foo();", implementation="method Foo() { }"),
    ]
    dataset = writer_examples_to_dataset(examples)
    assert dataset[0]["prompt"] == format_writer_training_prompt(examples[0])
    assert dataset[0]["example"] is examples[0]


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors=None):
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        return {"input_ids": input_ids.tolist()}

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens
        return " ".join(str(int(t)) for t in token_ids)


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_dummy", torch.tensor(0.0))

    def generate(self, input_ids, **_):
        pad = torch.full((input_ids.size(0), 2), 4, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, pad], dim=1)


def test_generate_implementation_returns_new_tokens():
    tokenizer = _DummyTokenizer()
    model = _DummyModel()
    example = WriterExample(id="1", stub="method Foo();", implementation="", header="class C {}", spec="desc")
    result = generate_implementation(
        model=model,
        tokenizer=tokenizer,
        example=example,
        gen_cfg=WriterGenerationConfig(max_new_tokens=10),
    )
    assert result == "4 4"


def test_generation_config_handles_deterministic_mode():
    cfg = WriterGenerationConfig(max_new_tokens=5, temperature=0.0, top_p=0.5)
    kwargs = cfg.to_kwargs()
    assert kwargs["do_sample"] is False


def test_format_annotator_prompt_includes_context():
    example = WriterExample(
        id="1",
        stub="method Foo();",
        implementation="",
        header="class C {}",
        spec="desc",
    )
    prompt = format_annotator_prompt(None, example, "method Foo() { var x := 1; }")
    assert "// spec: desc" in prompt
    assert "class C" in prompt


def test_writer_reward_uses_cascade():
    example = WriterExample(id="1", stub="method Foo();", implementation="", header="class C {}", spec=None)

    class AnnotatorStub:
        def annotate(self, example, writer_body):
            return f"{writer_body}\n// pass"

    class DummyDafny:
        def verify(self, dafny_file):
            code = dafny_file.get_code() or ""
            return "// pass" in code

    reward_fn = build_writer_reward(AnnotatorStub(), dafny=DummyDafny())
    reward, components = reward_fn("prompt", "method Foo() { }", {"example": example})
    assert reward == 1.0
    assert components["verified"] is True
