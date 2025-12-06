from __future__ import annotations

import copy
import gc
import random
import statistics
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
import torch.nn.functional as F
import re

from dafny_file import Dafny
from memory_efficient_optimizer import MemoryEfficientAdamW
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
    RL_SYSTEM_MESSAGE,
    RL_USER_TEMPLATE,
    SFT_SYSTEM_MESSAGE,
    SFT_USER_TEMPLATE,
    verification_reward_function,
)

try:
    from torch.amp import autocast
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
        old_model: Optional[nn.Module] = None,
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

        # Enable gradient checkpointing to reduce activation memory
        if self.config.gradient_checkpointing and hasattr(
            self.policy_model, "gradient_checkpointing_enable"
        ):
            self.policy_model.gradient_checkpointing_enable()

        # Setup old_model based on config
        if self.config.use_old_model:
            self.old_model = old_model or self._clone_model(self.policy_model)
            if self.config.offload_models_to_cpu:
                self.old_model.to("cpu")
            else:
                self.old_model.to(self.device)
            self.old_model.eval()
        else:
            self.old_model = None

        # Setup ref_model based on config
        if self.config.use_reference_model:
            self.ref_model = ref_model or self._clone_model(self.policy_model)
            if self.config.offload_models_to_cpu:
                self.ref_model.to("cpu")
            else:
                self.ref_model.to(self.device)
            self._freeze_model(self.ref_model)
        else:
            self.ref_model = None

        # Use memory-efficient optimizer if enabled
        if optimizer:
            self.optimizer = optimizer
        elif self.config.memory_efficient_optimizer:
            self.optimizer = MemoryEfficientAdamW(
                self.policy_model.parameters(),
                lr=self.config.learning_rate,
                enabled=True,
            )
        else:
            self.optimizer = AdamW(
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
            minibatch, metadata = self._sample_minibatch(dataset)
            if not minibatch.prompts:
                continue

            rollouts = self._collect_rollouts(minibatch, metadata)
            if not rollouts:
                continue

            self._cleanup_memory()

            self._score_rollouts(rollouts)

            normalized = self._normalize_rewards(rollouts)

            if self.config.advantage_whitening:
                normalized = self._whiten_advantages(normalized)

            policy_batches = self._build_policy_batch(rollouts, normalized)

            self._cleanup_memory()

            update_metrics = self._update_policy(policy_batches)
            del policy_batches
            reward_metrics = self._summarize_rewards(rollouts)
            del rollouts

            # Add memory metrics if logging is enabled
            if self.config.log_memory_usage:
                memory_metrics = self._log_memory_usage()
                metrics = {**reward_metrics, **update_metrics, **memory_metrics, "step": self._step}
            else:
                metrics = {**reward_metrics, **update_metrics, "step": self._step}

            if self._step % self.config.log_freq == 0:
                self.logger(metrics)

            # Sync old_model with policy if enabled
            if (
                self.config.old_update_freq > 0
                and self.old_model is not None
                and self._step % self.config.old_update_freq == 0
            ):
                if self.config.offload_models_to_cpu:
                    self.old_model.to(self.device)
                self._sync_model(self.policy_model, self.old_model, freeze=False)
                if self.config.offload_models_to_cpu:
                    self.old_model.to("cpu")

            # Sync ref_model with policy if enabled
            if (
                self.config.ref_update_freq > 0
                and self.ref_model is not None
                and self._step % self.config.ref_update_freq == 0
            ):
                if self.config.offload_models_to_cpu:
                    self.ref_model.to(self.device)
                self._sync_model(self.policy_model, self.ref_model, freeze=True)
                if self.config.offload_models_to_cpu:
                    self.ref_model.to("cpu")

            # Checkpoint saving
            if (
                checkpoint_path
                and self.config.checkpoint_freq > 0
                and self._step % self.config.checkpoint_freq == 0
            ):
                self._save_checkpoint(checkpoint_path, self._step)

            self._cleanup_memory()
            self._step += 1

    def supervised_fine_tune(
        self,
        examples: Sequence[Mapping[str, Any]],
        epochs: int = 1,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
    ) -> Mapping[str, float]:
        """
        Run supervised fine-tuning on raw Dafny bodies paired with annotated bodies.

        Each example must provide `body` and `annotated_body` fields. The model is
        trained with cross-entropy loss to generate the annotated body given the raw
        body as context.
        """
        if not examples:
            return {}

        valid_pairs: List[Tuple[str, str]] = []
        for item in examples:
            if not isinstance(item, Mapping):
                continue
            body = str(item.get("body", "") or "").strip()
            annotated = str(item.get("annotated_body", "") or "").strip()
            if body and annotated:
                valid_pairs.append((body, annotated))

        if not valid_pairs:
            raise ValueError(
                "No valid supervised examples found; expected non-empty `body` and "
                "`annotated_body` fields."
            )

        effective_batch_size = max(1, batch_size or self.config.batch_size)
        grad_accum_steps = max(1, self.config.gradient_accumulation_steps)
        ignore_index = -100

        self.policy_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        cumulative_loss = 0.0
        cumulative_tokens = 0
        steps = 0

        for epoch in range(max(1, epochs)):
            if shuffle:
                self._rng.shuffle(valid_pairs)

            for start in range(0, len(valid_pairs), effective_batch_size):
                batch_pairs = valid_pairs[start : start + effective_batch_size]
                batch_tensors = self._build_supervised_batch(batch_pairs, ignore_index)
                if batch_tensors is None:
                    continue

                input_ids, attention_mask, labels = batch_tensors

                with self._autocast_context():
                    outputs = self.policy_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )
                    raw_loss = outputs.loss
                    loss = raw_loss / grad_accum_steps

                loss.backward()
                steps += 1

                cumulative_loss += float(raw_loss.detach())
                cumulative_tokens += int((labels != ignore_index).sum().item())

                if steps % grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

        # Flush remaining gradients if we exited early in an accumulation cycle.
        if steps % grad_accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        update_steps = max(1, (steps + grad_accum_steps - 1) // grad_accum_steps)
        mean_loss = cumulative_loss / update_steps
        return {
            "sft_loss": mean_loss,
            "sft_tokens": float(cumulative_tokens),
            "sft_steps": float(update_steps),
        }

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

        # Determine which model to use for generation
        if self.old_model is not None:
            if self.config.offload_models_to_cpu:
                generation_context = self._model_on_device(self.old_model, self.device)
            else:
                generation_context = nullcontext()
            generation_model = self.old_model
            was_training = False  # old_model is always in eval mode
        else:
            generation_context = nullcontext()
            generation_model = self.policy_model
            was_training = self.policy_model.training

        with generation_context:
            if self.old_model is None and was_training:
                self.policy_model.eval()

            try:
                for prompt_idx, prompt in enumerate(minibatch.prompts):
                    prompt_ids = minibatch.prompt_token_ids[prompt_idx]
                    prompt_tensor = torch.tensor(
                        prompt_ids, dtype=torch.long, device=self.device
                    ).unsqueeze(0)
                    if pad_token_id is not None:
                        prompt_attention_mask = (prompt_tensor != pad_token_id).long()
                    else:
                        prompt_attention_mask = torch.ones_like(
                            prompt_tensor, dtype=torch.long
                        )

                    # Batched generation: expand prompt for group_size samples
                    batch_input = prompt_tensor.expand(self.config.group_size, -1)
                    batch_attention = prompt_attention_mask.expand(
                        self.config.group_size, -1
                    )

                    with torch.no_grad():
                        with self._autocast_context():
                            generated = generation_model.generate(
                                input_ids=batch_input,
                                attention_mask=batch_attention,
                                max_new_tokens=self.config.max_new_tokens,
                                temperature=self.config.temperature,
                                top_p=self.config.top_p,
                                do_sample=True,
                                pad_token_id=pad_token_id,
                                eos_token_id=eos_token_id,
                                return_dict_in_generate=True,
                                output_scores=False,
                            )

                    sequences = generated.sequences.tolist()

                    for g in range(self.config.group_size):
                        generated_ids = sequences[g]
                        completion_ids = generated_ids[len(prompt_ids) :]

                        completion_text = self.tokenizer.decode(
                            completion_ids, skip_special_tokens=True
                        )
                        full_text = self.tokenizer.decode(
                            generated_ids, skip_special_tokens=True
                        )

                        rollout_metadata = dict(metadata[prompt_idx])
                        rollout_metadata["generated_code"] = completion_text

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
                        rollouts.append(
                            RolloutItem(response=response, metadata=rollout_metadata)
                        )
            finally:
                if self.old_model is None and was_training:
                    self.policy_model.train()

        self._cleanup_memory()
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

    def _build_supervised_batch(
        self,
        pairs: Sequence[Tuple[str, str]],
        ignore_index: int,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        if not pairs:
            return None

        pad_token_id = self.pad_token_id
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        encoded_inputs: List[Tensor] = []
        encoded_labels: List[Tensor] = []
        max_length = 0

        for body, annotated in pairs:
            prompt_text = self._format_sft_prompt(body)
            target_text = annotated.strip()
            if not target_text:
                continue

            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=True,
                return_tensors="pt",
            )["input_ids"][0]

            target_ids = self.tokenizer(
                target_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]

            if eos_token_id is not None and (len(target_ids) == 0 or target_ids[-1] != eos_token_id):
                target_ids = torch.cat(
                    [target_ids, torch.tensor([eos_token_id], dtype=torch.long)]
                )

            input_ids = torch.cat([prompt_ids, target_ids], dim=0)
            labels = torch.full_like(input_ids, ignore_index)
            labels[len(prompt_ids) :] = input_ids[len(prompt_ids) :]

            encoded_inputs.append(input_ids)
            encoded_labels.append(labels)
            max_length = max(max_length, input_ids.size(0))

        if not encoded_inputs:
            return None

        batch_size = len(encoded_inputs)
        device = self.device

        input_batch = torch.full(
            (batch_size, max_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_batch = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
            device=device,
        )
        label_batch = torch.full(
            (batch_size, max_length),
            ignore_index,
            dtype=torch.long,
            device=device,
        )

        for row, (seq, lab) in enumerate(zip(encoded_inputs, encoded_labels)):
            length = seq.size(0)
            input_batch[row, :length] = seq.to(device)
            attention_batch[row, :length] = 1
            label_batch[row, :length] = lab.to(device)

        return input_batch, attention_batch, label_batch

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

        grad_accum_steps = max(1, self.config.gradient_accumulation_steps)
        accumulation_step = 0
        self.optimizer.zero_grad()
        pad_token_id = self.pad_token_id

        total_loss = 0.0

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
                outputs = self.policy_model(
                    input_ids=input_token_ids,
                    attention_mask=attention_for_model,
                    use_cache=False
                )

                logits = outputs.logits

                per_token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_token_ids.reshape(-1),
                    ignore_index=pad_token_id,
                    reduction="none",
                ).reshape(target_token_mask.shape)

                del logits
                del outputs
                
            token_log_probs = -per_token_loss * target_token_mask
            advantages_expanded = advantages.view(-1, 1)
            objective = (token_log_probs * advantages_expanded).sum() / total_target_tokens
            
            loss = -objective

            loss = loss / grad_accum_steps

            total_loss += loss.detach().item()
            loss.backward(retain_graph=False)
            del per_token_loss
            del loss
            del objective
            del token_log_probs
            del advantages_expanded
            del advantages

        grad_norm = self._clip_gradients(self.policy_model)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": float(total_loss),
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
    # Memory Management Utilities
    # --------------------------------------------------------------------- #
    def _cleanup_memory(self) -> None:
        """Force garbage collection and clear CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def _model_on_device(
        self, model: nn.Module, device: torch.device
    ) -> Generator[nn.Module, None, None]:
        """Temporarily move a model to a device, then move it back."""
        original_device = next(model.parameters()).device
        try:
            model.to(device)
            yield model
        finally:
            model.to(original_device)
            self._cleanup_memory()

    def _log_memory_usage(self, tag: str = "") -> Dict[str, float]:
        """Log current GPU memory usage."""
        if not torch.cuda.is_available():
            return {}

        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        suffix = f"_{tag}" if tag else ""
        memory_info = {
            f"memory_allocated_gb{suffix}": allocated,
            f"memory_reserved_gb{suffix}": reserved,
            f"memory_max_allocated_gb{suffix}": max_allocated,
        }
        return memory_info

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
        user_text = RL_USER_TEMPLATE.format(dafny_code_snippet=prompt)
        return self._build_chat_prompt(RL_SYSTEM_MESSAGE, user_text)

    def _format_sft_prompt(self, body: str) -> str:
        user_text = SFT_USER_TEMPLATE.format(dafny_body=body)
        return self._build_chat_prompt(SFT_SYSTEM_MESSAGE, user_text)

    def _build_chat_prompt(self, system_text: str, user_text: str) -> str:
        system_clean = system_text.strip()
        user_clean = user_text.strip()

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_clean},
                {"role": "user", "content": user_clean},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                # Some tokenizers may expose a different signature; fall back to manual formatting.
                pass

        return f"{system_clean}\n\n{user_clean}"

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
                # Generated text was not valid Dafny code; keep defaults (all zeros)
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
            return autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return _NullContext()


class _NullContext:
    """Fallback context manager when AMP is unavailable."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
