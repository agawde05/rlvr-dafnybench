from __future__ import annotations

import copy
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
import torch.nn.functional as F
import re


from dafny_file import Dafny
from data_types import GrpoConfig, Response
from verification_task import (
    ASSUMTION_WEIGHT,
    DELETION_WEIGHT,
    FORMAT_WEIGHT,
    VERIFICATION_WEIGHT,
    assume_reward_function,
    deletion_reward_function,
    format_reward_function,
    get_generated_dafny_code,
    SYSTEM_MESSAGE,
    USER_TEMPLATE,
    verification_reward_function,
)

try:
    from torch.cuda.amp import autocast
except ImportError:  # pragma: no cover - CPU-only environments
    autocast = None  # type: ignore


RewardFn = Callable[[str, str, Mapping[str, Any]], Tuple[float, Dict[str, Any]]]

if TYPE_CHECKING:
    from data_types import Minibatch


@dataclass
class RolloutItem:
    """Internal container that ties a response to its originating metadata."""

    response: Response
    metadata: Mapping[str, Any]


@dataclass
class PolicyBatch:
    """Micro-batch container for GRPO updates."""

    input_ids: Tensor
    attention_mask: Tensor
    target_mask: Tensor
    advantages: Tensor
    num_target_tokens: int


class CustomRLTrainer:
    """
    Modular GRPO-style trainer that orchestrates sampling, reward computation,
    and policy updates for the Dafny verification task.

    Parameters
    ----------
    model:
        HuggingFace-compatible causal LM representing the trainable policy π.
    tokenizer:
        Tokenizer aligned with `model`.
    reward_fn:
        Callable with signature `(prompt: str, completion: str, metadata: Mapping)`
        that returns `(scalar_reward, component_dict)`. If None, a default reward
        built from `verification_task.py` utilities is used.
    config:
        GRPO configuration. Defaults to `GrpoConfig()` when omitted.
    optimizer:
        Optional PyTorch optimizer. Defaults to AdamW over `model.parameters()`.
    ref_model:
        Optional initial clone of `model` to serve as a frozen reference policy.
    scheduler:
        Optional learning-rate scheduler.
    dafny:
        Optional `Dafny` verifier handle used by the default reward function.
    dafny_path:
        Alternative to `dafny`: path to Dafny binary to instantiate `Dafny`.
    logger:
        Callable that receives a dict of metrics every logging step.
    device:
        Preferred torch device string; falls back to CUDA when available else CPU.

    Dataset Expectations
    --------------------
    The trainer expects each dataset element to be either:
      * a raw prompt string, or
      * a mapping containing at least the key `"prompt"`, plus optional
        fields such as `"original_code"` for reward shaping.

    Reward Integration
    ------------------
    The default reward wiring combines the helpers from `verification_task.py`:
      total = FORMAT_WEIGHT  * format_reward
             + VERIFICATION_WEIGHT * verification_reward
             + ASSUMTION_WEIGHT * assume_reward
             + DELETION_WEIGHT * deletion_reward
    Missing metadata (e.g., original code) degrades gracefully by treating
    the corresponding component as zero.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        reward_fn: Optional[RewardFn] = None,
        config: Optional[GrpoConfig] = None,
        optimizer: Optional[Optimizer] = None,
        ref_model: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        dafny: Optional[Dafny] = None,
        dafny_path: Optional[str] = None,
        logger: Optional[Callable[[Dict[str, Any]], None]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or GrpoConfig()
        self.device = self._resolve_device(device or self.config.device)

        self.tokenizer = tokenizer
        self.policy_model = model.to(self.device)
        self.policy_model.train()
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        self.use_autocast = bool(
            self.config.mixed_precision and autocast and torch.cuda.is_available()
        )

        self.ref_model = (ref_model or self._clone_model(self.policy_model)).to(
            self.device
        )
        self._freeze_model(self.ref_model)

        self.optimizer = optimizer or AdamW(
            self.policy_model.parameters(), lr=self.config.learning_rate
        )
        self.scheduler = scheduler

        self.dafny = dafny or (Dafny(Path(dafny_path)) if dafny_path else None)
        self.reward_fn: RewardFn = reward_fn or self._build_default_reward_fn()

        self.logger = logger or (lambda metrics: print(f"[trainer] {metrics}"))
        self._rng = random.Random()
        self._step = 0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def train(
        self,
        dataset: Sequence[Any],
        num_steps: int,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        """
        Run GRPO training for a fixed number of steps.

        Parameters
        ----------
        dataset:
            Sequence of training examples (strings or mappings with `prompt` key).
        num_steps:
            Number of outer GRPO steps. Each step samples `config.batch_size`
            prompts and collects `group_size` rollouts per prompt.
        checkpoint_dir:
            Optional path for periodic checkpoints every `config.checkpoint_freq`.
        """
        if not dataset:
            raise ValueError("Dataset is empty; cannot start training.")

        dataset = list(dataset)
        checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None

        for i in range(num_steps):
            print(f"Step {i} of {num_steps}")
            minibatch, metadata = self._sample_minibatch(dataset)
            print(f"Sampled minibatch")
            if not minibatch.prompts:
                continue

            rollouts = self._collect_rollouts(minibatch, metadata)
            print(f"Collected rollouts")
            if not rollouts:
                continue

            torch.cuda.empty_cache()
                        
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

            self._score_rollouts(rollouts)

            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

            print(f"Scored rollouts")
            normalized = self._normalize_rewards(rollouts)

            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

            print(f"Normalized rewards")
            if self.config.advantage_whitening:
                normalized = self._whiten_advantages(normalized)

                print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

                print(f"Whitened advantages")
            policy_batches = self._build_policy_batch(rollouts, normalized)

            torch.cuda.empty_cache()

            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
            print(f"Built policy batch")
            update_metrics = self._update_policy(policy_batches)
            print(f"Updated policy")
            reward_metrics = self._summarize_rewards(rollouts)
            print(f"Summarized rewards")
            metrics = {**reward_metrics, **update_metrics, "step": self._step}

            if self._step % self.config.log_freq == 0:
                self.logger(metrics)

            if (
                checkpoint_path
                and self.config.checkpoint_freq > 0
                and self._step % self.config.checkpoint_freq == 0
            ):
                self._save_checkpoint(checkpoint_path, self._step)

            if self.config.ref_update_freq > 0 and (
                self._step % self.config.ref_update_freq == 0
            ):
                self._sync_model(self.policy_model, self.ref_model, freeze=True)

            self._step += 1

    # --------------------------------------------------------------------- #
    # Sampling & Rollout Helpers
    # --------------------------------------------------------------------- #
    def _sample_minibatch(
        self, dataset: Sequence[Any]
    ) -> Tuple["Minibatch", List[Mapping[str, Any]]]:
        from data_types import Minibatch  # local import to avoid cycle

        batch_size = min(self.config.batch_size, len(dataset))
        if batch_size == 0:
            return Minibatch(prompts=[], prompt_tokens=[], prompt_token_ids=[]), []

        selected = self._rng.sample(dataset, batch_size)
        raw_prompts: List[str] = []
        prompts: List[str] = []
        prompt_token_ids: List[List[int]] = []

        for item in selected:
            raw_prompt = self._extract_prompt(item)
            formatted_prompt = self._format_prompt(raw_prompt)
            raw_prompts.append(raw_prompt)
            prompts.append(formatted_prompt)

            encoded = self.tokenizer(
                formatted_prompt, add_special_tokens=True, return_tensors=None,
            )
            ids = encoded["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            prompt_token_ids.append(ids)

        minibatch = Minibatch(
            prompts=prompts,
            prompt_tokens=[],
            prompt_token_ids=prompt_token_ids,
        )
        metadata: List[Mapping[str, Any]] = []
        for idx, item in enumerate(selected):
            meta: Dict[str, Any]
            if isinstance(item, Mapping):
                meta = dict(item)
            else:
                meta = {}
            meta.setdefault("original_code", raw_prompts[idx])
            metadata.append(meta)
        return minibatch, metadata

    def _collect_rollouts(
        self,
        minibatch: "Minibatch",
        metadata: List[Mapping[str, Any]],
    ) -> List[RolloutItem]:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        rollouts: List[RolloutItem] = []
        if len(minibatch.prompts) == 0:
            return rollouts

        for prompt_idx, prompt in enumerate(minibatch.prompts):
            print(f"Collecting rollouts for prompt {prompt_idx} of {len(minibatch.prompts)}")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            prompt_ids = minibatch.prompt_token_ids[prompt_idx]
            prompt_tensor = torch.tensor(
                prompt_ids, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            if pad_token_id is not None:
                prompt_attention_mask = (prompt_tensor != pad_token_id).long()
            else:
                prompt_attention_mask = torch.ones_like(prompt_tensor, dtype=torch.long)

            batch_input = prompt_tensor.repeat(self.config.group_size, 1)

            model_was_training = self.policy_model.training
            if model_was_training:
                self.policy_model.eval()
            try:
                with torch.no_grad():
                    generated = self.policy_model.generate(
                        input_ids=batch_input,
                        attention_mask=prompt_attention_mask,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
            finally:
                if model_was_training:
                    self.policy_model.train()

            sequences = generated.sequences.tolist()

            for _ in range(self.config.group_size):

                generated_ids = sequences[_]
                completion_ids = generated_ids[len(prompt_ids) :]

                completion_text = self.tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                )
                full_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                rollout_metadata = dict(metadata[prompt_idx])
                rollout_metadata["diff_json"] = completion_text

                response = Response(
                    prompt=prompt,
                    full_text=full_text,
                    completion_text=completion_text,
                    prompt_token_ids=prompt_ids,
                    prompt_tokens=[],
                    generated_token_ids=completion_ids,
                    is_complete=bool(
                        eos_token_id
                        and eos_token_id in completion_ids
                        and completion_ids[-1] == eos_token_id
                    ),
                    reward=0.0,
                    reward_components={},
                )
                rollouts.append(RolloutItem(response=response, metadata=rollout_metadata))

        return rollouts

    # --------------------------------------------------------------------- #
    # Reward Computation
    # --------------------------------------------------------------------- #
    def _score_rollouts(self, rollouts: List[RolloutItem]) -> None:
        for item in rollouts:
            response = item.response
            reward, components = self.reward_fn(
                response.prompt, response.completion_text, item.metadata
            )
            response.reward = float(reward)
            response.reward_components = components

    def _normalize_rewards(self, rollouts: List[RolloutItem]) -> List[float]:
        group = self.config.group_size
        eps = self.config.reward_norm_eps
        normalized: List[float] = []

        for i in range(0, len(rollouts), group):
            group_rewards = [r.response.reward for r in rollouts[i : i + group]]
            if not group_rewards:
                continue
            mean = statistics.mean(group_rewards)
            std = statistics.pstdev(group_rewards) if len(group_rewards) > 1 else 0.0
            denom = std if std > 0 else 0.0
            for reward in group_rewards:
                normalized.append((reward - mean) / (denom + eps))

        return normalized

    def _whiten_advantages(self, advantages: List[float]) -> List[float]:
        if not advantages:
            return advantages
        mean = statistics.mean(advantages)
        std = statistics.pstdev(advantages) if len(advantages) > 1 else 0.0
        eps = self.config.reward_norm_eps
        return [(adv - mean) / (std + eps) for adv in advantages]

    def _build_policy_batch(
        self, rollouts: List[RolloutItem], advantages: List[float]
    ) -> List[PolicyBatch]:
        sequences: List[Tuple[List[int], int, float]] = []
        for idx, item in enumerate(rollouts):
            prompt_ids = list(item.response.prompt_token_ids)
            completion_ids = list(item.response.generated_token_ids)
            if not completion_ids:
                continue
            seq = prompt_ids + completion_ids
            sequences.append((seq, len(prompt_ids), advantages[idx]))

        if not sequences:
            return []

        sequences.sort(key=lambda x: len(x[0])) # efficient micro batching
        micro_size = self.config.microbatch_size or len(sequences)

        batches: List[PolicyBatch] = []
        for start in range(0, len(sequences), micro_size):
            chunk = sequences[start : start + micro_size]
            chunk_max_len = max(len(seq) for seq, _, _ in chunk)
            batch_size = len(chunk)

            input_ids = torch.full(
                (batch_size, chunk_max_len),
                self.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
            target_mask = torch.zeros_like(input_ids, dtype=torch.float32)
            advantages_tensor = torch.zeros(batch_size, device=self.device, dtype=torch.float32)

            num_target_tokens = 0

            for row_idx, (seq, prompt_len, adv) in enumerate(chunk):
                length = len(seq)
                seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device)
                input_ids[row_idx, :length] = seq_tensor
                attention_mask[row_idx, :length] = 1.0
                target_mask[row_idx, prompt_len:length] = 1.0
                advantages_tensor[row_idx] = float(adv)
                num_target_tokens += max(length - prompt_len, 0)

            batches.append(
                PolicyBatch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_mask=target_mask,
                    advantages=advantages_tensor,
                    num_target_tokens=num_target_tokens,
                )
            )

        return batches

    # --------------------------------------------------------------------- #
    # Policy Update
    # --------------------------------------------------------------------- #
    def _update_policy(
        self, policy_batches: Sequence[PolicyBatch]
    ) -> Dict[str, float]:
        if not policy_batches:
            return {}

        total_target_tokens = sum(
            batch.num_target_tokens for batch in policy_batches if batch.num_target_tokens
        )
        if total_target_tokens == 0:
            return {}

        metrics_accumulator: Dict[str, float] = {}
        grad_accum_steps = max(1, self.config.gradient_accumulation_steps)
        accumulation_step = 0
        self.optimizer.zero_grad()
        pad_token_id = self.pad_token_id
        num_epochs = max(1, getattr(self.config, "num_epochs", 1))

        for micro in policy_batches:
            if micro.num_target_tokens == 0:
                continue

            input_ids = micro.input_ids
            attention_mask = micro.attention_mask
            target_mask = micro.target_mask
            advantages = micro.advantages

            input_token_ids = input_ids[:, :-1]
            target_token_ids = input_ids[:, 1:]
            attention_for_model = attention_mask[:, :-1]
            target_token_mask = target_mask[:, 1:]

            with self._autocast_context():

                print(f"prev Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"prev Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

                outputs = self.policy_model(
                    input_ids=input_token_ids,
                    attention_mask=attention_for_model,
                )

                print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

                logits = outputs.logits

            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_token_ids.reshape(-1),
                ignore_index=pad_token_id,
                reduction="none",
            ).reshape(target_token_mask.shape)

            token_log_probs = -per_token_loss * target_token_mask
            advantages_expanded = advantages.view(-1, 1)
            objective = (token_log_probs * advantages_expanded).sum() / total_target_tokens
            loss = -objective

            step_metrics = {
                "loss": float(loss.detach().cpu()),
            }

            loss = loss / grad_accum_steps
            loss.backward()

            del loss, objective, logits, per_token_loss, token_log_probs, advantages_expanded
            torch.cuda.empty_cache()

        grad_norm = self._clip_gradients(self.policy_model)

        self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
        }

    # --------------------------------------------------------------------- #
    # Logging & Checkpointing
    # --------------------------------------------------------------------- #
    def _summarize_rewards(self, rollouts: List[RolloutItem]) -> Dict[str, float]:
        rewards = [item.response.reward for item in rollouts]
        if not rewards:
            return {}

        mean = statistics.mean(rewards)
        std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        return {
            "reward_mean": float(mean),
            "reward_std": float(std),
            "reward_min": float(min(rewards)),
            "reward_max": float(max(rewards)),
            "num_episodes": float(len(rewards)),
        }

    def _save_checkpoint(self, checkpoint_dir: Path, step: int) -> None:
        step_dir = checkpoint_dir / f"checkpoint_step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        model_dir = step_dir / "policy_model"
        if hasattr(self.policy_model, "save_pretrained"):
            self.policy_model.save_pretrained(model_dir)
        else:
            torch.save(self.policy_model.state_dict(), model_dir / "model.pt")

        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "step": step,
            },
            step_dir / "trainer_state.pt",
        )

    # --------------------------------------------------------------------- #
    # Low-level Utilities
    # --------------------------------------------------------------------- #
    def _clip_gradients(self, model: nn.Module) -> float:
        max_norm = float(self.config.max_grad_norm)
        grad_norm = clip_grad_norm_(model.parameters(), max_norm)
        return float(grad_norm.detach().cpu()) if grad_norm is not None else 0.0

    def _accumulate_metrics(
        self, accumulator: Dict[str, float], metrics: Dict[str, float]
    ) -> None:
        for key, value in metrics.items():
            accumulator.setdefault(key, 0.0)
            accumulator[key] += float(value)
        accumulator["_count"] = accumulator.get("_count", 0.0) + 1.0

    def _finalize_metrics(self, accumulator: Dict[str, float]) -> Dict[str, float]:
        count = accumulator.pop("_count", 0.0)
        if count == 0:
            return {}
        return {key: value / count for key, value in accumulator.items()}

    def _format_prompt(self, prompt: str) -> str:
        system_text = SYSTEM_MESSAGE.strip()
        user_text = USER_TEMPLATE.format(dafny_code_snippet=prompt).strip()

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                # Some tokenizers may have a different signature; fall back to manual formatting.
                pass

        return f"{system_text}\n\n{user_text}"

    def _extract_prompt(self, item: Any) -> str:
        if isinstance(item, Mapping):
            if "prompt" in item:
                return str(item["prompt"])
            if "input" in item:
                return str(item["input"])
        return str(item)

    def _build_default_reward_fn(self) -> RewardFn:
        def _default(prompt: str, completion: str, metadata: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
            original_code = (
                metadata.get("original_code")
                if isinstance(metadata, Mapping)
                else None
            )
            if original_code is None:
                original_code = prompt

            format_score = format_reward_function(completion)
            verification_score = 0.0
            assume_score = 0.0
            deletion_score = 0.0
            modified_code = ""

            try:
                dafny_file = get_generated_dafny_code(completion, original_code)
                modified_code = dafny_file.get_code() or ""
                if self.dafny:
                    verification_score = verification_reward_function(
                        dafny_file, self.dafny
                    )
                assume_score = assume_reward_function(original_code, modified_code)
                deletion_score = deletion_reward_function(original_code, modified_code)
            except ValueError:
                # Missing <answer> tags; keep defaults (all zeros)
                pass

            total_reward = (
                FORMAT_WEIGHT * format_score
                + VERIFICATION_WEIGHT * verification_score
                + ASSUMTION_WEIGHT * assume_score
                + DELETION_WEIGHT * deletion_score
            )

            components = {
                "format": format_score,
                "verification": verification_score,
                "assume": assume_score,
                "deletion": deletion_score,
                "modified_code": modified_code,
            }
            return total_reward, components

        return _default

    def _resolve_device(self, preferred: str) -> torch.device:
        if preferred == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        if preferred == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(preferred)

    def _clone_model(self, model: nn.Module) -> nn.Module:
        clone = copy.deepcopy(model)
        clone.eval()
        return clone

    def _freeze_model(self, model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    def _sync_model(
        self, source: nn.Module, destination: nn.Module, freeze: bool
    ) -> None:
        destination.load_state_dict(source.state_dict())
        if freeze:
            self._freeze_model(destination)
        else:
            for param in destination.parameters():
                param.requires_grad = False
            destination.eval()

    def _autocast_context(self):
        if self.use_autocast and autocast:
            return autocast()
        return _NullContext()


class _NullContext:
    """Fallback context manager when AMP is unavailable."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False