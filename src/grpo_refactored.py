# grpo.py
import logging
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from dafny_file import DafnyFile
from collections import defaultdict
from dataclasses import replace
from transformers import PreTrainedModel, PreTrainedTokenizer
from data_types import Response, Minibatch


class DafnyGrpoTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dafny_verifier: Any,
        config: Dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dafny_verifier = dafny_verifier
        self.config = config

        self.logger = logging.getLogger(__name__)
        self.num_answers_per_question = config.get("num_answers_per_question", 4)
        self.max_gen_len = config.get("max_gen_len", 512)
        self.micro_batch_size = config.get("micro_batch_size", 2)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.temperature = config.get("temperature", 0.8)
        self.top_p = config.get("top_p", 0.9)
        self.kl_beta = config.get("kl_beta", 0.1)

        self.device = model.device
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.ref_model = deepcopy(model).eval()
        self.ref_model.to(self.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def train_step(
        self, batch: Minibatch, optimizer: torch.optim.Optimizer
    ) -> Dict[str, float | np.floating | int]:
        """
        Purpose:
            Execute one GRPO policy update over a minibatch while logging diagnostics.
        Args:
            batch (Minibatch): contains B prompts/token ids; tokens are variable length.
            optimizer (torch.optim.Optimizer): optimizer covering trainable params.
        Returns:
            Dict[str, float | np.floating | int]: scalar metrics such as loss (1,), KL (1,),
                success counts (scalars), and num_responses (int).
        Assumptions:
            - On-policy rollouts generated with the current model weights.
        Invariants:
            - Prompt tokens remain excluded from the policy gradient loss masking.
        Side Effects:
            - Generates responses, mutates model gradients, performs optimizer.step, logs stats.
        """
        # Generate multiple responses for each prompt
        responses = self.generate_responses(batch)
        if not responses:
            return {
                "loss": 0.0,
                "success_rate": 0.0,
                "num_responses": 0,
                "kl_value": 0.0,
                "avg_reward": 0.0,
                "avg_generation_length": 0.0,
            }

        # Compute GRPO policy update
        loss_metrics = self._compute_and_apply_gradients(responses, optimizer)

        # Compute metrics
        success_rate = np.mean(
            [
                resp.reward_components.get("verification_success", 0)
                for resp in responses
            ]
        )
        rewards = [resp.reward for resp in responses]
        avg_reward = np.mean(rewards)
        avg_generation_length = np.mean(
            [len(resp.generated_token_ids) for resp in responses]
        )

        self.logger.info(
            "Batch stats: reward_mean=%.3f reward_std=%.3f success_rate=%.3f "
            "avg_gen_len=%.2f kl=%.4f",
            float(avg_reward),
            float(np.std(rewards)),
            float(success_rate),
            float(avg_generation_length),
            float(loss_metrics.get("kl_value", 0.0)),
        )

        return {
            **loss_metrics,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_generation_length": avg_generation_length,
            "num_responses": len(responses),
            "num_successful": sum(
                resp.reward_components.get("verification_success", 0)
                for resp in responses
            ),
        }

    def generate_responses(self, batch: Minibatch) -> List[Response]:
        """
        Purpose:
            Sample multiple rollouts per prompt to estimate GRPO advantages.
        Args:
            batch (Minibatch): contains B prompts plus token ids; shapes are
                lists of length B with variable-length token lists.
        Returns:
            List[Response]: length B * num_answers_per_question with token ids
                for prompts and generations.
        Assumptions:
            - On-policy sampling using current model parameters.
        Invariants:
            - Prompt tokens remain excluded from downstream loss masking.
        Side Effects:
            - Generates text via the model and queries the Dafny verifier.
        """
        responses = []

        for i, (prompt, prompt_tokens, prompt_token_ids) in enumerate(
            zip(batch.prompts, batch.prompt_tokens, batch.prompt_token_ids)
        ):
            for j in range(self.num_answers_per_question):
                # Generate completion
                generated_token_ids = self._generate_with_sampling(prompt_token_ids)
                full_token_ids = prompt_token_ids + generated_token_ids
                full_text = self.tokenizer.decode(
                    full_token_ids, skip_special_tokens=True
                )
                generated_text = self.tokenizer.decode(
                    generated_token_ids, skip_special_tokens=True
                )

                # Compute Dafny reward
                reward_info = self._compute_dafny_reward(prompt, generated_text)

                # Create Response object
                response = Response(
                    prompt=prompt,
                    full_text=full_text,
                    prompt_token_ids=prompt_token_ids,
                    prompt_tokens=prompt_tokens,
                    generated_token_ids=generated_token_ids,
                    is_complete=True,  # We handle truncation in generation
                    reward=reward_info["reward"],
                    reward_components=reward_info,
                )
                responses.append(response)

        return responses

    def _generate_with_sampling(self, prompt_token_ids: List[int]) -> List[int]:
        """
        Purpose:
            Sample a completion token-by-token using temperature and top-p decoding.
        Args:
            prompt_token_ids (List[int]): length L_p prompt tokens that seed decoding.
        Returns:
            List[int]: generated token ids (length >=1) including EOS when emitted.
        Assumptions:
            - On-policy sampling from the current model weights.
        Invariants:
            - Prompt tokens remain excluded from downstream loss since only generated ids are returned.
        Side Effects:
            - Switches the model to eval mode for decoding and runs torch.no_grad().
        """
        self.model.eval()

        input_ids = torch.tensor([prompt_token_ids], device=self.device)
        current_ids = input_ids.clone()
        generated_tokens: List[int] = []
        eos_id = self.tokenizer.eos_token_id

        for _ in range(self.max_gen_len):
            with torch.no_grad():
                outputs = self.model(current_ids)
                next_token_logits = outputs.logits[:, -1, :]

            # Apply temperature
            if self.temperature != 1.0:
                next_token_logits = next_token_logits / self.temperature

            # Apply top-p sampling
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(
                    indices_to_remove, float("-inf")
                )

            if (
                self.pad_token_id is not None
                and self.pad_token_id != eos_id
                and self.pad_token_id < next_token_logits.size(-1)
            ):
                next_token_logits[..., self.pad_token_id] = float("-inf")

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token.item()
            if self.pad_token_id is not None:
                assert (
                    next_token_id != self.pad_token_id
                ), "Generated tokens must not include PAD"

            generated_tokens.append(next_token_id)
            current_ids = torch.cat([current_ids, next_token], dim=1)

            if eos_id is not None and next_token_id == eos_id:
                break
        else:
            if eos_id is not None and (not generated_tokens or generated_tokens[-1] != eos_id):
                eos_tensor = torch.tensor([[eos_id]], device=self.device)
                generated_tokens.append(eos_id)
                current_ids = torch.cat([current_ids, eos_tensor], dim=1)

        return generated_tokens

    def _compute_dafny_reward(self, prompt: str, generated_text: str) -> Dict[str, Any]:
        """
        Purpose:
            Evaluate Dafny code quality and emit structured reward components.
        Args:
            prompt (str): scalar string prompt that precedes the generation.
            generated_text (str): scalar decoded generation appended to prompt.
        Returns:
            Dict[str, Any]: reward components with scalar floats plus failure message.
        Assumptions:
            - On-policy samples; Dafny verifier side effects are deterministic per code.
        Invariants:
            - Prompt tokens only influence generation, not loss masking.
        Side Effects:
            - Invokes the Dafny verifier which may spawn subprocesses.
        """
        full_code = prompt + generated_text
        reward_components: Dict[str, Any] = {
            "formatting_success": 0.0,
            "compilation_success": 0.0,
            "verification_success": 0.0,
            "timeout_or_failure": "",
            "full_code": full_code,
        }

        try:
            dafny_file = DafnyFile.from_code(full_code)
            reward_components["formatting_success"] = 1.0

            verification_success = bool(
                self.dafny_verifier.verify(dafny_file, timeout_seconds=30)
            )

            reward_components["verification_success"] = float(verification_success)
            reward_components["compilation_success"] = (
                1.0 if verification_success else 0.0
            )
            reward_components["timeout_or_failure"] = (
                "" if verification_success else "verification_failed"
            )
        except TimeoutError:
            reward_components["timeout_or_failure"] = "timeout"
        except Exception as exc:
            reward_components["timeout_or_failure"] = str(exc)

        reward = (
            0.1 * reward_components["formatting_success"]
            + 0.3 * reward_components["compilation_success"]
            + 0.6 * reward_components["verification_success"]
        )
        reward_components["reward"] = reward
        return reward_components

    def _compute_and_apply_gradients(
        self, responses: List[Response], optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Purpose:
            Backpropagate the GRPO objective with KL regularization and update the optimizer.
        Args:
            responses (List[Response]): length N list with tokenized prompts/generations.
            optimizer (torch.optim.Optimizer): optimizer whose parameter group matches self.model.
        Returns:
            Dict[str, float]: scalar metrics (loss, grad_norm, kl_value) each shape ().
        Assumptions:
            - On-policy samples; rewards are normalized per prompt group already.
        Invariants:
            - Prompt tokens remain excluded from the policy gradient loss masking.
        Side Effects:
            - Modifies model gradients and performs optimizer.step().
        """
        # Normalize rewards per group (by prompt)
        normalized_responses = self._normalize_rewards_per_group(responses)

        total_loss = 0.0
        num_micro_batches = 0
        kl_values = []

        # Process in micro-batches
        for i in range(0, len(normalized_responses), self.micro_batch_size):
            batch_responses = normalized_responses[i : i + self.micro_batch_size]
            batch_data = self._prepare_training_batch(batch_responses)

            if batch_data:
                # Forward pass
                self.model.train()
                outputs = self.model(
                    input_ids=batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                )
                with torch.no_grad():
                    ref_outputs = self.ref_model(
                        input_ids=batch_data["input_ids"],
                        attention_mask=batch_data["attention_mask"],
                    )

                # Compute GRPO loss
                batch_loss, loss_info = self._compute_grpo_loss(
                    outputs.logits,
                    ref_outputs.logits,
                    batch_data["labels"],
                    batch_data["attention_mask"],
                    batch_data["advantages"],
                    batch_data["prompt_lengths"],
                )

                # Backward pass
                batch_loss.backward()
                total_loss += batch_loss.item()
                kl_values.append(loss_info["kl_value"])
                self.logger.debug(
                    "Micro-batch %d loss=%.4f kl=%.4f valid_tokens=%d",
                    num_micro_batches,
                    batch_loss.item(),
                    loss_info["kl_value"],
                    loss_info["num_valid_tokens"],
                )
                num_micro_batches += 1

        if num_micro_batches == 0:
            return {"loss": 0.0, "grad_norm": 0.0, "kl_value": 0.0}

        # Gradient clipping and update
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        return {
            "loss": total_loss / num_micro_batches,
            "grad_norm": grad_norm.item(),
            "kl_value": float(np.mean(kl_values)),
        }

    def _normalize_rewards_per_group(self, responses: List[Response]) -> List[Response]:
        """
        Purpose:
            Normalize rewards per prompt group so each prompt has zero-mean unit-std returns.
        Args:
            responses (List[Response]): list length N with prompt text/generation reward scalars.
        Returns:
            List[Response]: list of same length with normalized reward floats.
        Assumptions:
            - On-policy reward signals grouped by identical prompt strings.
        Invariants:
            - Prompt tokens remain excluded from subsequent loss computations.
        Side Effects:
            - Allocates new Response dataclasses via dataclasses.replace.
        """
        groups = defaultdict(list)
        for response in responses:
            groups[response.prompt].append(response)

        normalized_responses = []
        for group_responses in groups.values():
            rewards = [resp.reward for resp in group_responses]
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards) + 1e-8

            for response in group_responses:
                normalized_reward = (response.reward - mean_reward) / std_reward
                # Create new Response with normalized reward
                normalized_response = replace(response, reward=normalized_reward)
                normalized_responses.append(normalized_response)

        return normalized_responses

    def _prepare_training_batch(
        self, responses: List[Response]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Purpose:
            Pad variable-length sequences and build label/tensor masks for RL fine-tuning.
        Args:
            responses (List[Response]): length N micro-batch with prompt/generated ids.
        Returns:
            Optional[Dict[str, torch.Tensor]]: tensors with shapes
                input_ids [N, T], attention_mask [N, T], labels [N, T],
                advantages [N], prompt_lengths [N]; None when no valid targets.
        Assumptions:
            - On-policy samples; rewards already normalized by prompt group.
        Invariants:
            - Prompt tokens remain masked out of the policy gradient loss.
        Side Effects:
            - Allocates new tensors on the model device for training.
        """
        if not responses:
            return None

        valid_responses = [resp for resp in responses if resp.generated_token_ids]
        if not valid_responses:
            return None

        sequences = []
        prompt_lengths = []
        for response in valid_responses:
            full_tokens = response.prompt_token_ids + response.generated_token_ids
            sequences.append(full_tokens)
            prompt_lengths.append(len(response.prompt_token_ids))

        max_len = max(len(seq) for seq in sequences)
        batch_size = len(sequences)

        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=self.device
        )
        labels = torch.full_like(input_ids, fill_value=-100)

        for idx, (seq, prompt_len) in enumerate(zip(sequences, prompt_lengths)):
            seq_len = len(seq)
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device)
            input_ids[idx, :seq_len] = seq_tensor
            attention_mask[idx, :seq_len] = 1
            labels[idx, :seq_len] = seq_tensor
            if prompt_len > 0:
                labels[idx, :prompt_len] = -100

        advantages = torch.tensor(
            [resp.reward for resp in valid_responses],
            device=self.device,
            dtype=torch.float32,
        )
        prompt_length_tensor = torch.tensor(
            prompt_lengths, device=self.device, dtype=torch.long
        )

        assert input_ids.shape == labels.shape, "Labels and inputs must align"
        assert (
            advantages.shape[0] == input_ids.shape[0]
        ), "Advantages must match batch size"

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantages": advantages,
            "prompt_lengths": prompt_length_tensor,
        }

    def _compute_grpo_loss(
        self,
        logits: torch.Tensor,
        ref_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        advantages: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Purpose:
            Compute the GRPO policy gradient objective with KL regularization.
        Args:
            logits (torch.Tensor): [B, T, V] current model logits.
            ref_logits (torch.Tensor): [B, T, V] frozen reference model logits.
            labels (torch.Tensor): [B, T] token ids with -100 where masked.
            attention_mask (torch.Tensor): [B, T] 1 for real tokens, 0 for pads.
            advantages (torch.Tensor): [B] normalized reward advantages.
            prompt_lengths (torch.Tensor): [B] lengths of prompts (pre-generation).
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: scalar loss tensor and metrics dict
                (kl_value float, num_valid_tokens int).
        Assumptions:
            - On-policy samples; labels already contain prompt masking.
        Invariants:
            - Prompt tokens remain excluded from the loss through masking.
        Side Effects:
            - None (pure computation over provided tensors).
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous().to(logits.dtype)

        safe_shift_labels = shift_labels.clone()
        invalid_label_mask = safe_shift_labels == -100
        safe_shift_labels[invalid_label_mask] = 0

        log_probs = F.log_softmax(shift_logits, dim=-1)
        ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)
        target_log_probs = torch.gather(
            log_probs, -1, safe_shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        ref_target_log_probs = torch.gather(
            ref_log_probs, -1, safe_shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        batch_positions = torch.arange(
            shift_labels.size(-1), device=logits.device, dtype=torch.long
        ).unsqueeze(0)
        prompt_offsets = torch.clamp(prompt_lengths - 1, min=0)
        generation_mask = (
            batch_positions >= prompt_offsets.unsqueeze(1)
        ).to(logits.dtype)
        token_mask = shift_attention_mask * generation_mask
        token_mask = token_mask.masked_fill(invalid_label_mask, 0.0)

        advantages = advantages.to(logits.dtype)
        masked_objective = (target_log_probs * advantages[:, None] * token_mask).sum()

        valid_token_sum = token_mask.sum()
        normalizer = valid_token_sum.clamp(min=1.0)

        kl_terms = (target_log_probs - ref_target_log_probs) * token_mask
        kl_value = kl_terms.sum() / normalizer

        loss = -(masked_objective / normalizer) + self.kl_beta * kl_value
        metrics = {
            "kl_value": float(kl_value.detach().item()),
            "num_valid_tokens": int(valid_token_sum.detach().item()),
        }
        return loss, metrics
