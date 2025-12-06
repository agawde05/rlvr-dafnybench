import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from grpo import DafnyGRPOTrainer  # noqa: E402
from data_types import Response  # noqa: E402


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 11, hidden_size: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, vocab_size)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        return SimpleNamespace(logits=logits)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(t) for t in token_ids)

    def tokenize(self, text: str):
        return text.split()

    def encode(self, text: str, add_special_tokens: bool = False):
        tokens = list(range(2, 2 + len(text.split())))
        return tokens


class DummyVerifier:
    def verify(self, dafny_file, timeout_seconds=30):
        return True


def make_trainer():
    config = {
        "num_answers_per_question": 1,
        "max_gen_len": 5,
        "micro_batch_size": 2,
        "kl_beta": 0.05,
    }
    model = DummyModel()
    tokenizer = DummyTokenizer()
    return DafnyGRPOTrainer(model, tokenizer, DummyVerifier(), config)


def make_response(prompt_ids, gen_ids, reward=1.0):
    dummy_components = {
        "reward": reward,
        "formatting_success": 1.0,
        "compilation_success": 1.0,
        "verification_success": reward,
        "timeout_or_failure": "",
    }
    return Response(
        prompt="p",
        full_text="",
        prompt_token_ids=prompt_ids,
        prompt_tokens=["p"] * len(prompt_ids),
        generated_token_ids=gen_ids,
        is_complete=True,
        reward=reward,
        reward_components=dummy_components,
    )


def test_prepare_training_batch_masks_prompt_tokens():
    trainer = make_trainer()
    responses = [
        make_response([2, 3], [4, 5, 1]),
        make_response([5, 6, 7], [8, 9, 1]),
    ]

    batch = trainer._prepare_training_batch(responses)
    assert batch is not None

    labels = batch["labels"].cpu()
    prompt_lengths = batch["prompt_lengths"].cpu().tolist()

    for row_idx, prompt_len in enumerate(prompt_lengths):
        assert torch.all(labels[row_idx, :prompt_len] == -100)
        assert labels[row_idx, prompt_len].item() != -100


def test_compute_grpo_loss_ignores_prompt_tokens():
    trainer = make_trainer()
    responses = [make_response([2, 3], [4, 5, 6, 1], reward=0.5)]
    batch = trainer._prepare_training_batch(responses)
    assert batch is not None

    seq_len = batch["input_ids"].size(1)
    vocab_size = trainer.model.vocab_size
    logits = torch.randn(1, seq_len, vocab_size)
    ref_logits = logits.clone()

    loss, metrics = trainer._compute_grpo_loss(
        logits,
        ref_logits,
        batch["labels"],
        batch["attention_mask"],
        batch["advantages"],
        batch["prompt_lengths"],
    )
    assert metrics["num_valid_tokens"] == 4

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    prompt_len = int(batch["prompt_lengths"][0].item())
    seq_tokens = int(batch["attention_mask"][0].sum().item())
    advantage = float(batch["advantages"][0].item())
    manual_sum = 0.0
    counted = 0
    for token_idx in range(prompt_len, seq_tokens):
        label = int(batch["labels"][0, token_idx].item())
        if label == -100:
            continue
        shift_pos = token_idx - 1
        manual_sum += float(log_probs[0, shift_pos, label]) * advantage
        counted += 1
    manual_loss = -manual_sum / counted
    assert pytest.approx(manual_loss, rel=1e-5) == loss.item()


def test_compute_grpo_loss_reports_kl_term():
    trainer = make_trainer()
    responses = [make_response([2, 3], [4, 5, 1], reward=0.8)]
    batch = trainer._prepare_training_batch(responses)
    assert batch is not None

    seq_len = batch["input_ids"].size(1)
    vocab_size = trainer.model.vocab_size
    logits = torch.randn(1, seq_len, vocab_size)
    ref_logits = torch.randn(1, seq_len, vocab_size)

    loss, metrics = trainer._compute_grpo_loss(
        logits,
        ref_logits,
        batch["labels"],
        batch["attention_mask"],
        batch["advantages"],
        batch["prompt_lengths"],
    )
    assert math.isfinite(loss.item())

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
    prompt_len = int(batch["prompt_lengths"][0].item())
    seq_tokens = int(batch["attention_mask"][0].sum().item())

    kl_sum = 0.0
    total = 0
    for token_idx in range(prompt_len, seq_tokens):
        label = int(batch["labels"][0, token_idx].item())
        if label == -100:
            continue
        shift_pos = token_idx - 1
        kl_sum += float(
            log_probs[0, shift_pos, label] - ref_log_probs[0, shift_pos, label]
        )
        total += 1
    manual_kl = kl_sum / total
    assert pytest.approx(manual_kl, rel=1e-5) == metrics["kl_value"]


def test_prepare_training_batch_shapes_stable_across_batches():
    trainer = make_trainer()
    responses = [
        make_response([2, 3], [4, 5, 6, 1]),
        make_response([7], [8, 1]),
    ]
    batch = trainer._prepare_training_batch(responses)
    assert batch is not None

    input_shape = batch["input_ids"].shape
    label_shape = batch["labels"].shape
    assert input_shape == label_shape
    assert input_shape[0] == len(responses)
    assert batch["attention_mask"].shape == input_shape
    assert batch["prompt_lengths"].shape[0] == len(responses)
    second_seq_len = len(responses[1].prompt_token_ids + responses[1].generated_token_ids)
    assert torch.all(batch["attention_mask"][1, second_seq_len:] == 0)
