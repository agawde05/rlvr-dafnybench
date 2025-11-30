# grpo.py
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import numpy as np
from dafny_file import DafnyFile
from collections import defaultdict
from dataclasses import replace
from transformers import PreTrainedModel, PreTrainedTokenizer
from data_types import Response, Minibatch


class DafnyGRPOTrainer:
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

        self.num_answers_per_question = config.get("num_answers_per_question", 4)
        self.max_gen_len = config.get("max_gen_len", 512)
        self.micro_batch_size = config.get("micro_batch_size", 2)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.temperature = config.get("temperature", 0.8)
        self.top_p = config.get("top_p", 0.9)

        self.device = model.device
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def train_step(
        self, batch: Minibatch, optimizer: torch.optim.Optimizer
    ) -> Dict[str, float | np.floating | int]:
        """GRPO training step"""
        # Generate multiple responses for each prompt
        responses = self.generate_responses(batch)
        if not responses:
            return {"loss": 0.0, "success_rate": 0.0, "num_responses": 0}

        # Compute GRPO policy update
        loss_metrics = self._compute_and_apply_gradients(responses, optimizer)

        # Compute metrics
        success_rate = np.mean(
            [
                resp.reward_components.get("verification_success", 0)
                for resp in responses
            ]
        )
        avg_reward = np.mean([resp.reward for resp in responses])

        return {
            **loss_metrics,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "num_responses": len(responses),
            "num_successful": sum(
                resp.reward_components.get("verification_success", 0)
                for resp in responses
            ),
        }

    def generate_responses(self, batch: Minibatch) -> List[Response]:
        """Generate multiple responses for each prompt in the minibatch"""
        responses = []

        for i, (prompt, prompt_tokens, prompt_token_ids) in enumerate(
            zip(batch.prompts, batch.prompt_tokens, batch.prompt_token_ids)
        ):
            for j in range(self.num_answers_per_question):
                # Generate completion
                generated_token_ids = self._generate_with_sampling(prompt_token_ids)
                generated_text = self.tokenizer.decode(
                    generated_token_ids, skip_special_tokens=True
                )
                full_text = prompt + generated_text

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
        """Custom generation with sampling"""
        self.model.eval()

        input_ids = torch.tensor([prompt_token_ids], device=self.device)
        current_ids = input_ids.clone()
        generated_tokens = []

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

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token.item()

            if next_token_id == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id)
            current_ids = torch.cat([current_ids, next_token], dim=1)

        return generated_tokens

    def _compute_dafny_reward(self, prompt: str, generated_text: str) -> Dict[str, Any]:
        """Compute reward using Dafny verification"""
        try:
            full_code = prompt + generated_text
            dafny_file = DafnyFile.from_code(full_code)

            verification_success = self.dafny_verifier.verify(
                dafny_file, timeout_seconds=30
            )
            reward = 1.0 if verification_success else 0.0

            return {
                "reward": reward,
                "verification_success": float(verification_success),
                "full_code": full_code,
                "error": None,
            }
        except Exception as e:
            return {"reward": 0.0, "verification_success": 0.0, "error": str(e)}

    def _compute_and_apply_gradients(
        self, responses: List[Response], optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """GRPO policy update"""
        # Normalize rewards per group (by prompt)
        normalized_responses = self._normalize_rewards_per_group(responses)

        total_loss = 0.0
        num_micro_batches = 0

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

                # Compute GRPO loss
                batch_loss = self._compute_grpo_loss(
                    outputs.logits,
                    batch_data["labels"],
                    batch_data["attention_mask"],
                    batch_data["advantages"],
                )

                # Backward pass
                batch_loss.backward()
                total_loss += batch_loss.item()
                num_micro_batches += 1

        if num_micro_batches == 0:
            return {"loss": 0.0, "grad_norm": 0.0}

        # Gradient clipping and update
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad()

        return {"loss": total_loss / num_micro_batches, "grad_norm": grad_norm.item()}

    def _normalize_rewards_per_group(self, responses: List[Response]) -> List[Response]:
        """Normalize rewards within each prompt group"""
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
        """Prepare training batch"""
        if not responses:
            return None

        sequences = []
        for response in responses:
            # Combine prompt + generated tokens
            full_tokens = response.prompt_token_ids + response.generated_token_ids
            sequences.append(full_tokens)

        # Manual padding
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        attention_masks = []

        for seq in sequences:
            padding_len = max_len - len(seq)
            padded_sequences.append(seq + [self.pad_token_id] * padding_len)
            attention_masks.append([1] * len(seq) + [0] * padding_len)

        input_ids = torch.tensor(padded_sequences, device=self.device)
        attention_mask = torch.tensor(attention_masks, device=self.device)
        labels = input_ids.clone()

        advantages = torch.tensor(
            [resp.reward for resp in responses], device=self.device
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantages": advantages,
        }

    def _compute_grpo_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GRPO loss"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = torch.gather(
            log_probs, -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Apply advantages
        advantages_expanded = advantages.unsqueeze(-1).expand(-1, shift_labels.size(-1))
        masked_objective = (
            target_log_probs * advantages_expanded * shift_attention_mask
        ).sum()
        num_target_tokens = shift_attention_mask.sum()

        return -masked_objective / num_target_tokens  # Negative for gradient ascent
