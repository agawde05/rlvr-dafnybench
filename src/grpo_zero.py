"""
GRPO-Zero Implementation for Language Model Training

This module implements the Group Relative Policy Optimization (GRPO) algorithm
in a "Zero" setting - meaning:
    1. No supervised fine-tuning stage
    2. No reward model (only environment rewards from verification)
    3. Direct RL training using group-relative outcome rewards

The algorithm maintains three models:
    - À_¸ (current policy): The model being trained
    - À_old (old policy): A snapshot used for importance sampling
    - À_ref (reference model): A frozen reference for KL penalty

Training uses PPO-style clipped objectives with per-group advantage normalization.
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer
import random

from data_types import Response, Minibatch, TokenIds, GrpoConfig


# ============================================================================
# Rollout Pipeline
# ============================================================================


def sample_questions(
    dataset: List[str], batch_size: int, tokenizer: PreTrainedTokenizer
) -> Minibatch:
    """
    Sample a batch of questions from the dataset.

    This function randomly samples questions from the provided dataset and
    prepares them as a Minibatch for rollout generation.

    Args:
        dataset: List of question strings to sample from.
        batch_size: Number of questions to sample.
        tokenizer: Tokenizer for converting text to token IDs.

    Returns:
        Minibatch containing sampled questions with tokenized prompts.

    Mathematical Definition:
        Sample B questions Q = {q_1, ..., q_B} from dataset D.
        For each q_i, tokenize to get token IDs.

    Notes:
        - Sampling is uniform random without replacement within a batch.
        - Questions are tokenized but not padded yet (that happens in rollout).
    """

    # Sample batch_size questions uniformly at random without replacement
    sampled_prompts = random.sample(dataset, min(batch_size, len(dataset)))

    # Tokenize each prompt
    prompt_token_ids_list = []

    for prompt in sampled_prompts:
        # Tokenize the prompt
        encoded = tokenizer(prompt, add_special_tokens=True, return_tensors=None)

        # Get token IDs as a list
        token_ids = encoded["input_ids"]
        prompt_token_ids_list.append(token_ids)

    # Create and return Minibatch
    # Note: prompt_tokens is left empty as it's redundant (can be reconstructed from token_ids if needed)
    return Minibatch(
        prompts=sampled_prompts,
        prompt_tokens=[],  # for debugging, can be filled if needed
        prompt_token_ids=prompt_token_ids_list,
    )


def rollout_batch(
    model: PreTrainedModel,
    minibatch: Minibatch,
    config: GrpoConfig,
    tokenizer: PreTrainedTokenizer,
    reward_fn: callable,
) -> List[Response]:
    """
    Perform batched autoregressive rollout to generate completions.

    This function takes a minibatch of B questions and generates G completions
    per question using KV-cache-based autoregressive sampling. The result is
    B*G Response objects, each containing the full generation and its reward.

    Args:
        model: The policy model À_old to generate completions from.
        minibatch: Batch of questions to generate completions for.
        config: Configuration containing group_size, max_new_tokens, etc.
        tokenizer: Tokenizer for encoding/decoding.
        reward_fn: Function that computes reward for a completion.
            Signature: reward_fn(prompt: str, completion: str, metadata: Dict) -> (float, Dict)

    Returns:
        List of B*G Response objects, grouped by question.
        Order: [q1_g1, q1_g2, ..., q1_gG, q2_g1, ..., qB_gG]

    Mathematical Definition:
        For each question q in minibatch:
            Sample G completions: c ~ À_old(·|q)
            Compute reward: r = R(q, c) from environment
            Store as Response object

    Tensor Shapes:
        Input:
            - minibatch.prompt_token_ids: List[TokenIds], len B
        Internal:
            - replicated_prompts: [B*G, prompt_len]
            - tokens: [B*G, seq_len] where seq_len grows incrementally
            - attention_mask: [B*G, seq_len]
            - position_ids: [B*G, seq_len]
        Output:
            - List of B*G Response objects

    Algorithm:
        1. Replicate each prompt G times: [B, L] -> [B*G, L]
        2. Initialize token matrix with prompts
        3. For each decoding step (up to max_new_tokens):
            a. Forward pass through model with KV cache
            b. Sample next token using temperature and top-p
            c. Append to token matrix
            d. Update EOS tracking
            e. Break early if all sequences finished
        4. Extract generated tokens (excluding prompt)
        5. Decode to text and compute rewards
        6. Package into Response objects

    Notes:
        - Uses KV caching for efficiency (only compute new token each step).
        - Masks prompt tokens during sampling (only generate after prompt).
        - Detects EOS token per sequence and stops generation.
        - Breaks early when all sequences in batch have finished.
        - Reward computation is environment-based (verification task).
    """

    responses: List[Response] = []

    if len(minibatch.prompts) == 0:
        return responses

    device = next(model.parameters()).device
    group_size = max(1, config.group_size)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    for prompt, prompt_token_ids in zip(
        minibatch.prompts, minibatch.prompt_token_ids
    ):
        input_ids = torch.tensor(
            [prompt_token_ids], dtype=torch.long, device=device
        )

        for _ in range(group_size):
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )

            full_ids = generated[0].tolist()
            completion_ids = full_ids[len(prompt_token_ids) :]
            completion_text = tokenizer.decode(
                completion_ids, skip_special_tokens=False
            )
            full_text = tokenizer.decode(full_ids, skip_special_tokens=False)

            rollout_metadata: Dict[str, Any] = {
                "original_code": prompt,
                "diff_json": completion_text,
            }
            reward, components = reward_fn(prompt, completion_text, rollout_metadata)
            components = components or {}

            responses.append(
                Response(
                    prompt=prompt,
                    full_text=full_text,
                    prompt_token_ids=prompt_token_ids,
                    prompt_tokens=[],
                    generated_token_ids=completion_ids,
                    is_complete=bool(
                        eos_token_id
                        and completion_ids
                        and completion_ids[-1] == eos_token_id
                    ),
                    reward=float(reward),
                    reward_components=components,
                )
            )

    return responses


def replicate_prompts(
    prompt_token_ids: List[TokenIds], group_size: int, pad_token_id: int = 0
) -> torch.Tensor:
    """
    Replicate each prompt G times for group sampling.

    Args:
        prompt_token_ids: List of B token ID lists, one per question.
        group_size: Number of times to replicate each prompt (G).
        pad_token_id: Token ID to use for padding (default: 0).

    Returns:
        Tensor of shape [B*G, max_prompt_len] containing padded and
        replicated prompts.

    Tensor Shapes:
        Input: List of B variable-length sequences
        Output: [B*G, max_prompt_len]

    Algorithm:
        1. Find maximum prompt length in batch
        2. Pad all prompts to max length
        3. Replicate each prompt G times consecutively
        4. Result: [p1, p1, ..., p1, p2, p2, ..., p2, ..., pB, pB, ..., pB]

    Notes:
        - Padding token should be tokenizer.pad_token_id.
        - Replication is consecutive to maintain group structure.
    """
    B = len(prompt_token_ids)

    # Edge case: empty list
    if B == 0:
        return torch.empty(0, 0, dtype=torch.long)

    # Find maximum prompt length
    max_prompt_len = max(len(ids) for ids in prompt_token_ids)

    # Create padded tensor [B, max_prompt_len]
    padded = torch.full((B, max_prompt_len), pad_token_id, dtype=torch.long)
    for i, ids in enumerate(prompt_token_ids):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    # Replicate each prompt G times consecutively using repeat_interleave
    # [p1, p2, p3] -> [p1, p1, ..., p1, p2, p2, ..., p2, p3, p3, ..., p3]
    return padded.repeat_interleave(group_size, dim=0)


def autoregressive_decode_step(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Optional[Tuple],
    temperature: float,
    top_p: float,
    eos_token_id: int,
) -> Tuple[torch.Tensor, Tuple]:
    """
    Perform a single autoregressive decoding step with KV caching.

    Args:
        model: Language model to decode with (HuggingFace PreTrainedModel).
        input_ids: Token IDs, shape [B*G, seq_len] or [B*G, 1] if using cache.
        attention_mask: Attention mask, shape [B*G, seq_len].
        past_key_values: KV cache from previous steps (or None for first step).
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        eos_token_id: EOS token ID for the tokenizer.

    Returns:
        next_tokens: Sampled token IDs, shape [B*G].
        updated_past_key_values: Updated KV cache for next step.

    Tensor Shapes:
        Input:
            - input_ids: [B*G, 1] (only last token when using cache)
            - attention_mask: [B*G, seq_len]
        Output:
            - next_tokens: [B*G]
            - past_key_values: Tuple of cached key-value tensors

    Algorithm:
        1. Forward pass through HuggingFace model to get logits
        2. Extract logits for last position: [B*G, vocab_size]
        3. Apply temperature scaling: logits = logits / temperature
        4. Apply top-p (nucleus) filtering
        5. Sample from filtered distribution
        6. Return sampled tokens and updated cache

    Notes:
        - HuggingFace models return a CausalLMOutput object with .logits and .past_key_values
        - When past_key_values is not None, only pass last token to model.
        - Temperature < 1: more deterministic, > 1: more random.
        - Top-p filters out low-probability tokens before sampling.
    """
    # Forward pass through model
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

    # Extract logits and updated KV cache
    logits = outputs.logits  # [B*G, seq_len, vocab_size]
    updated_past_key_values = outputs.past_key_values

    # Get logits for the last position (next token prediction)
    next_token_logits = logits[:, -1, :]  # [B*G, vocab_size]

    # Apply temperature scaling
    if temperature != 1.0:
        next_token_logits = next_token_logits / temperature

    # Apply top-p filtering
    if top_p < 1.0:
        next_token_logits = apply_top_p_filtering(next_token_logits, top_p)

    # Convert to probabilities
    probs = F.softmax(next_token_logits, dim=-1)  # [B*G, vocab_size]

    # Sample from the distribution
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B*G]

    return next_tokens, updated_past_key_values


def apply_top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) sampling filter to logits.

    Args:
        logits: Unnormalized logits, shape [batch_size, vocab_size].
        top_p: Cumulative probability threshold (e.g., 0.95).

    Returns:
        Filtered logits with low-probability tokens set to -inf,
        shape [batch_size, vocab_size].

    Algorithm:
        1. Sort logits in descending order
        2. Compute cumulative softmax probabilities
        3. Find cutoff index where cumulative prob exceeds top_p
        4. Set all logits below cutoff to -inf
        5. Return filtered logits

    Mathematical Definition:
        Let π(v|s) be the probability of token v given state s.
        Sort tokens by π(v|s) in descending order: v_1, v_2, ...
        Find smallest k such that sum_{i=1}^k π(v_i|s) >= top_p
        Keep only v_1, ..., v_k; set others to -inf

    Notes:
        - Setting to -inf ensures these tokens have 0 probability after softmax.
        - Prevents sampling from the long tail of unlikely tokens.

    TODO: Performance optimizations if needed:
        1. Top-k pre-filtering: First filter to top-k tokens (e.g., k=1000) before
           applying top-p. This reduces sorting from O(V log V) to O(k log k) where
           k << V. Can provide ~50x speedup for large vocabularies.

        2. Skip when top_p >= 1.0: Add early return to avoid expensive operations
           when no filtering is actually needed.

        3. Use HuggingFace's TopPLogitsWarper: Battle-tested, highly optimized
           implementation with additional optimizations.

        Current implementation is adequate for typical use (batch_size * group_size < 128,
        vocab_size ~50k). Top-p filtering takes ~1-2ms vs ~50-200ms for model forward
        pass, so it's not the bottleneck.
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find indices where cumulative probability exceeds top_p
    # We want to remove tokens where cumulative_prob > top_p
    # Shift right by 1 to keep at least the first token (highest prob)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = torch.roll(sorted_indices_to_remove, shifts=1, dims=-1)
    sorted_indices_to_remove[..., 0] = False

    # Create a copy of logits to avoid in-place modification issues
    filtered_logits = logits.clone()

    # Vectorized scatter: map removal mask back to original indices
    filtered_logits.scatter_(
        -1,
        sorted_indices,
        logits.gather(-1, sorted_indices).masked_fill(
            sorted_indices_to_remove, float("-inf")
        ),
    )

    return filtered_logits


# ============================================================================
# Group-Relative Reward Normalization
# ============================================================================


def normalize_rewards_per_group(
    responses: List[Response], group_size: int, eps: float = 1e-8
) -> List[float]:
    """
    Normalize rewards using group-relative statistics.

    This is the core "group-relative" component of GRPO. For each question,
    we normalize the rewards of its G completions using the mean and std
    of just that group, not the global batch statistics.

    Args:
        responses: List of B*G Response objects, grouped by question.
        group_size: Number of completions per question (G).
        eps: Epsilon for numerical stability in normalization.

    Returns:
        List of B*G normalized reward values (floats), same order as responses.

    Mathematical Definition:
        For each question q with completions {c_1, ..., c_G}:
            rewards: {r_1, ..., r_G}
            ¼_q = mean(r_1, ..., r_G)
            Ã_q = std(r_1, ..., r_G)
            normalized_r_i = (r_i - ¼_q) / (Ã_q + eps)

    Algorithm:
        1. Group responses by question: chunks of size G
        2. For each group:
            a. Extract raw rewards: [r_1, ..., r_G]
            b. Compute group mean ¼ and std Ã
            c. Normalize: (r_i - ¼) / (Ã + eps)
        3. Flatten normalized rewards back to list of B*G values

    Notes:
        - This is advantage estimation in GRPO: higher-than-average
          completions get positive advantage, lower get negative.
        - No value function needed - we use group statistics as baseline.
        - If Ã = 0 (all rewards identical), normalization gives 0 advantage.
        - Responses must be ordered as [q1_g1, ..., q1_gG, q2_g1, ..., qB_gG].
    """
    rewards = [response.reward for response in responses]
    normalized: List[float] = []

    if group_size <= 0:
        raise ValueError("group_size must be positive")

    for i in range(0, len(rewards), group_size):
        group = rewards[i : i + group_size]
        if not group:
            continue

        mu = sum(group) / len(group)
        var = sum((x - mu) ** 2 for x in group) / len(group)
        std = var ** 0.5

        denom = std if std > 0 else 0.0
        for r in group:
            normalized.append((r - mu) / (denom + eps))

    return normalized
    


def whiten_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Whiten advantages across entire batch for additional stability.

    This is an optional second normalization step that can be applied after
    group-relative normalization. It ensures advantages have zero mean and
    unit variance across the full batch.

    Args:
        advantages: Tensor of shape [B*G] containing (potentially already
            group-normalized) advantages.
        eps: Epsilon for numerical stability.

    Returns:
        Whitened advantages, shape [B*G].

    Mathematical Definition:
        ¼ = mean(advantages)
        Ã = std(advantages)
        whitened = (advantages - ¼) / (Ã + eps)

    Notes:
        - This is optional and controlled by config.advantage_whitening.
        - Can help with optimization stability but may reduce signal.
        - Applied after group normalization, not instead of it.
    """
    if advantages.numel() == 0:
        return advantages

    mu = advantages.mean()
    std = advantages.std()
    std = std.clamp_min(eps)
    return (advantages - mu) / std


# ============================================================================
# Batch Building for Policy Update
# ============================================================================


@dataclass
class PolicyBatch:
    """
    A batch of data prepared for policy gradient computation.

    Attributes:
        input_ids: Padded token IDs, shape [batch_size, seq_len].
            Contains both prompt and completion tokens.

        attention_mask: Attention mask, shape [batch_size, seq_len].
            1 for real tokens, 0 for padding.

        target_mask: Mask indicating completion tokens, shape [batch_size, seq_len].
            1 for completion tokens (where loss is computed), 0 for prompt/padding.

        advantages: Per-episode advantages, shape [batch_size].
            Normalized rewards (from group-relative normalization).

        old_log_probs: Log probabilities from À_old, shape [batch_size, seq_len].
            Used for importance sampling ratio.
    """

    input_ids: torch.Tensor  # [batch_size, seq_len]
    attention_mask: torch.Tensor  # [batch_size, seq_len]
    target_mask: torch.Tensor  # [batch_size, seq_len]
    advantages: torch.Tensor  # [batch_size]
    old_log_probs: torch.Tensor  # [batch_size, seq_len]


def _token_log_probs_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    shifted_input = input_ids[:, 1:]
    shifted_log_probs = log_probs[:, :-1, :]
    gathered = torch.gather(
        shifted_log_probs, dim=2, index=shifted_input.unsqueeze(-1)
    ).squeeze(-1)

    token_log_probs = torch.zeros(
        input_ids.shape, dtype=log_probs.dtype, device=input_ids.device
    )
    token_log_probs[:, 1:] = gathered

    mask = target_mask.to(token_log_probs.dtype)
    return token_log_probs * mask


def build_policy_batches(
    responses: List[Response],
    normalized_rewards: List[float],
    old_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    micro_batch_size: int,
) -> List[PolicyBatch]:
    """
    Convert responses into batches suitable for policy gradient computation.

    This function takes a list of responses (with completions and normalized
    rewards) and packages them into PolicyBatch objects with proper padding
    and masking for efficient training.

    Args:
        responses: List of B*G Response objects from rollout.
        normalized_rewards: List of B*G normalized reward values (advantages).
        tokenizer: Tokenizer for padding.
        micro_batch_size: Number of episodes per micro-batch.

    Returns:
        List of PolicyBatch objects, each containing a micro-batch of data.

    Algorithm:
        1. Sort responses by total length (prompt + completion) for efficient packing
        2. Divide into chunks of size micro_batch_size
        3. For each chunk:
            a. Find max sequence length in chunk
            b. Pad all sequences to max length
            c. Build input_ids: [prompt tokens + completion tokens + padding]
            d. Build attention_mask: 1 for real tokens, 0 for padding
            e. Build target_mask: 1 for completion tokens only, 0 elsewhere
            f. Extract advantages for this batch
            g. Compute old_log_probs (forward pass through À_old)
        4. Return list of PolicyBatch objects

    Tensor Shapes:
        For each micro-batch of size M:
            - input_ids: [M, max_seq_len]
            - attention_mask: [M, max_seq_len]
            - target_mask: [M, max_seq_len]
            - advantages: [M]
            - old_log_probs: [M, max_seq_len]

    Notes:
        - Sorting by length minimizes padding waste and speeds up training.
        - target_mask is crucial: loss should only be computed on generated tokens.
        - old_log_probs are computed once and cached (not recomputed during PPO epochs).
        - Padding token should not contribute to loss (masked out by target_mask).
    """
    if len(responses) == 0:
        return []

    if len(responses) != len(normalized_rewards):
        raise ValueError("responses and normalized_rewards must have equal length")

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0

    device = next(old_model.parameters()).device
    micro_batch_size = max(1, micro_batch_size)

    episodes: List[Tuple[List[int], int, float]] = []
    for response, advantage in zip(responses, normalized_rewards):
        prompt_ids = list(response.prompt_token_ids)
        completion_ids = list(response.generated_token_ids)
        sequence = prompt_ids + completion_ids
        episodes.append((sequence, len(prompt_ids), float(advantage)))

    episodes.sort(key=lambda item: len(item[0]), reverse=True)

    policy_batches: List[PolicyBatch] = []
    for start in range(0, len(episodes), micro_batch_size):
        chunk = episodes[start : start + micro_batch_size]
        if not chunk:
            continue

        batch_size = len(chunk)
        max_len = max(len(seq) for seq, _, _ in chunk)

        input_ids = torch.full(
            (batch_size, max_len),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        target_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.float32, device=device
        )
        advantages = torch.tensor(
            [adv for _, _, adv in chunk], dtype=torch.float32, device=device
        )

        for row, (sequence, prompt_len, _) in enumerate(chunk):
            seq_len = len(sequence)
            if seq_len == 0:
                continue

            seq_tensor = torch.tensor(sequence, dtype=torch.long, device=device)
            input_ids[row, :seq_len] = seq_tensor
            attention_mask[row, :seq_len] = 1
            target_mask[row, prompt_len:seq_len] = 1.0

        old_log_probs = compute_old_log_probs(
            old_model, input_ids, attention_mask, target_mask
        )

        policy_batches.append(
            PolicyBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_mask=target_mask,
                advantages=advantages,
                old_log_probs=old_log_probs,
            )
        )

    return policy_batches


def compute_old_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log probabilities under the old policy À_old.

    Args:
        model: The old policy model À_old.
        input_ids: Token IDs, shape [batch_size, seq_len].
        attention_mask: Attention mask, shape [batch_size, seq_len].
        target_mask: Mask for completion tokens, shape [batch_size, seq_len].

    Returns:
        Per-token log probabilities for completion tokens,
        shape [batch_size, seq_len]. Masked positions have log_prob = 0.

    Algorithm:
        1. Forward pass through model to get logits: [batch_size, seq_len, vocab_size]
        2. Compute log softmax: log_probs = log_softmax(logits, dim=-1)
        3. Gather log probs for actual tokens (next token prediction):
            - Shift input_ids left by 1 to get targets
            - Gather log_probs at target token indices
        4. Apply target_mask to zero out prompt and padding positions
        5. Return per-token log probs

    Tensor Shapes:
        Input:
            - input_ids: [batch_size, seq_len]
            - attention_mask: [batch_size, seq_len]
            - target_mask: [batch_size, seq_len]
        Internal:
            - logits: [batch_size, seq_len, vocab_size]
            - log_probs: [batch_size, seq_len, vocab_size]
        Output:
            - token_log_probs: [batch_size, seq_len]

    Mathematical Definition:
        For position t with token x_t:
            log À_old(x_t | x_<t) = log_softmax(logits_t)[x_t]

        Only computed for completion tokens (where target_mask = 1).

    Notes:
        - These log probs are cached and used for computing importance ratios.
        - Must use torch.no_grad() when computing for À_old (not training it).
        - Target prediction: logits at position t predict token at position t+1.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

    return _token_log_probs_from_logits(logits, input_ids, target_mask)


# ============================================================================
# Policy Evaluation Functions
# ============================================================================


def compute_policy_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token log probabilities under current policy À_¸.

    This is similar to compute_old_log_probs but for the current (training)
    policy. Gradients flow through this computation.

    Args:
        model: Current policy model À_¸ (with gradients enabled).
        input_ids: Token IDs, shape [batch_size, seq_len].
        attention_mask: Attention mask, shape [batch_size, seq_len].
        target_mask: Mask for completion tokens, shape [batch_size, seq_len].

    Returns:
        Tuple of:
            - Per-token log probabilities, shape [batch_size, seq_len].
            - Raw logits from the model forward pass.

    Tensor Shapes:
        Same as compute_old_log_probs.

    Mathematical Definition:
        For position t with token x_t:
            log À_¸(x_t | x_<t) = log_softmax(logits_t)[x_t]

    Notes:
        - Gradients are enabled (used for backpropagation).
        - Otherwise identical to compute_old_log_probs.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits
    token_log_probs = _token_log_probs_from_logits(
        logits, input_ids, target_mask
    )
    return token_log_probs, logits


def compute_reference_log_probs(
    ref_model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log probabilities under frozen reference model À_ref.

    The reference model is used to compute KL divergence penalty, preventing
    the policy from deviating too far from the initial pretrained model.

    Args:
        ref_model: Frozen reference model À_ref (no gradients).
        input_ids: Token IDs, shape [batch_size, seq_len].
        attention_mask: Attention mask, shape [batch_size, seq_len].
        target_mask: Mask for completion tokens, shape [batch_size, seq_len].

    Returns:
        Per-token log probabilities, shape [batch_size, seq_len].

    Tensor Shapes:
        Same as compute_old_log_probs.

    Mathematical Definition:
        For position t with token x_t:
            log À_ref(x_t | x_<t) = log_softmax(logits_t)[x_t]

    Notes:
        - Must use torch.no_grad() since reference model is frozen.
        - Reference model is typically initialized as a copy of À_¸ at start.
        - May be periodically updated (controlled by config.ref_update_freq).
    """
    with torch.no_grad():
        outputs = ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

    return _token_log_probs_from_logits(logits, input_ids, target_mask)


def compute_entropy(logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token entropy of the policy distribution.

    Entropy measures the randomness of the policy. Higher entropy means
    more exploration. We add entropy bonus to encourage exploration.

    Args:
        logits: Unnormalized logits, shape [batch_size, seq_len, vocab_size].
        target_mask: Mask for completion tokens, shape [batch_size, seq_len].

    Returns:
        Mean entropy across all completion tokens (scalar).

    Mathematical Definition:
        For position t:
            H_t = -£_v À_¸(v|s_t) log À_¸(v|s_t)

        where À_¸(v|s_t) = softmax(logits_t)[v]

        Return: mean(H_t for all t where target_mask[t] = 1)

    Algorithm:
        1. Compute probabilities: probs = softmax(logits, dim=-1)
        2. Compute log probabilities: log_probs = log_softmax(logits, dim=-1)
        3. Compute entropy: entropy = -£ probs * log_probs
        4. Apply target_mask to only include completion tokens
        5. Return mean entropy

    Tensor Shapes:
        Input:
            - logits: [batch_size, seq_len, vocab_size]
            - target_mask: [batch_size, seq_len]
        Internal:
            - probs: [batch_size, seq_len, vocab_size]
            - entropy: [batch_size, seq_len]
        Output:
            - mean_entropy: scalar

    Notes:
        - Higher entropy = more uniform distribution = more exploration.
        - Entropy naturally decreases during training as policy becomes confident.
        - Entropy bonus (config.entropy_coef * entropy) added to loss.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)

    mask = target_mask.to(entropy.dtype)
    total_tokens = mask.sum().clamp_min(1.0)
    return (entropy * mask).sum() / total_tokens


# ============================================================================
# GRPO-Zero Loss Function
# ============================================================================


def compute_grpo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    target_mask: torch.Tensor,
    config: GrpoConfig,
    policy_logits: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute GRPO-Zero loss with PPO-style clipping and KL penalty.

    This is the core loss function for GRPO-Zero. It combines:
        1. Clipped PPO objective for policy improvement
        2. KL penalty vs reference model for stability
        3. Entropy bonus for exploration

    Args:
        policy_log_probs: Log probs under À_¸, shape [batch_size, seq_len].
        old_log_probs: Log probs under À_old, shape [batch_size, seq_len].
        ref_log_probs: Log probs under À_ref, shape [batch_size, seq_len].
        advantages: Per-episode advantages, shape [batch_size].
        target_mask: Mask for completion tokens, shape [batch_size, seq_len].
        config: Configuration with clip_ratio, kl_coef, entropy_coef.

    Returns:
        loss: Scalar loss to minimize (negative of objective).
        metrics: Dictionary with diagnostic metrics:
            - policy_loss: Policy gradient component
            - kl_penalty: KL divergence penalty
            - entropy: Average entropy
            - clip_fraction: Fraction of ratios that were clipped
            - approx_kl: Approximate KL divergence

    Mathematical Definition:
        Per-token importance ratio:
            r_t = exp(log À_¸(x_t | x_<t) - log À_old(x_t | x_<t))

        Per-episode ratio (geometric mean over tokens):
            R = exp(£_t (log À_¸ - log À_old) * mask_t / £_t mask_t)

        Or simpler: sum over tokens then clip:
            advantage_weighted_ratio = £_t A * r_t * mask_t

        Clipped objective:
            L_clip = min(R * A, clip(R, 1-µ, 1+µ) * A)

        KL penalty:
            KL = £_t (log À_¸ - log À_ref) * mask_t / £_t mask_t

        Total loss:
            L = -L_clip + ²_kl * KL - ²_ent * H

        where:
            - A: advantage (normalized reward)
            - µ: clip_ratio
            - ²_kl: kl_coef
            - ²_ent: entropy_coef
            - H: entropy

    Algorithm:
        1. Compute per-token log probability ratios:
            log_ratio_t = policy_log_probs - old_log_probs
        2. Sum ratios over completion tokens per episode:
            log_ratio_sum = £_t log_ratio_t * target_mask_t
        3. Count completion tokens per episode:
            num_tokens = £_t target_mask_t
        4. Compute per-episode ratio:
            ratio = exp(log_ratio_sum / num_tokens)
        5. Compute unclipped term:
            unclipped = ratio * advantages
        6. Compute clipped term:
            clipped_ratio = clip(ratio, 1 - µ, 1 + µ)
            clipped = clipped_ratio * advantages
        7. Policy objective (per episode):
            policy_obj = min(unclipped, clipped)
        8. Compute KL penalty (per token, then mean):
            kl_per_token = (policy_log_probs - ref_log_probs) * target_mask
            kl = mean(kl_per_token[target_mask == 1])
        9. Compute entropy (already computed separately)
        10. Final loss:
            loss = -mean(policy_obj) + kl_coef * kl - entropy_coef * entropy

    Tensor Shapes:
        Input:
            - policy_log_probs: [batch_size, seq_len]
            - old_log_probs: [batch_size, seq_len]
            - ref_log_probs: [batch_size, seq_len]
            - advantages: [batch_size]
            - target_mask: [batch_size, seq_len]
        Internal:
            - log_ratios: [batch_size, seq_len]
            - ratio: [batch_size]
            - policy_obj: [batch_size]
            - kl: scalar
        Output:
            - loss: scalar

    Notes:
        - Clipping prevents large policy updates that could destabilize training.
        - KL penalty keeps policy close to reference, preventing mode collapse.
        - Entropy bonus encourages exploration in early training.
        - All operations must respect target_mask (only use completion tokens).
        - Advantages should already be normalized (group-relative + optional whitening).
    """
    mask = target_mask.to(policy_log_probs.dtype)
    token_counts = mask.sum(dim=1).clamp_min(1.0)

    # PPO ratio computation
    log_ratio = (policy_log_probs - old_log_probs) * mask
    log_ratio_sum = log_ratio.sum(dim=1)
    ratio = torch.exp(log_ratio_sum / token_counts)
    clipped_ratio = torch.clamp(
        ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
    )

    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    policy_obj = torch.minimum(unclipped, clipped)
    policy_loss = -policy_obj.mean()

    # KL term
    kl_per_episode = (
        (policy_log_probs - ref_log_probs) * mask
    ).sum(dim=1) / token_counts
    kl_penalty = kl_per_episode.mean()

    # Entropy bonus
    if policy_logits is not None:
        entropy = compute_entropy(policy_logits, target_mask)
    else:
        entropy = -(
            (policy_log_probs * mask).sum(dim=1) / token_counts
        ).mean()
        if not torch.is_tensor(entropy):
            entropy = torch.tensor(entropy, dtype=policy_log_probs.dtype)

    loss = (
        policy_loss
        + config.kl_coef * kl_penalty
        - config.entropy_coef * entropy
    )

    clip_fraction = (ratio - clipped_ratio).abs() > 1e-8
    clip_fraction = clip_fraction.float().mean()
    approx_kl = (ratio.log() - (ratio - 1.0)).mean()

    metrics = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl_penalty": float(kl_penalty.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
    }

    return loss, metrics


# ============================================================================
# Policy Update Loop
# ============================================================================


def update_policy(
    policy_model: PreTrainedModel,
    old_model: PreTrainedModel,
    ref_model: PreTrainedModel,
    optimizer: Optimizer,
    policy_batches: List[PolicyBatch],
    config: GrpoConfig,
) -> Dict[str, float]:
    """
    Perform policy update using collected rollouts.

    This function implements the inner loop of GRPO training:
        1. Loop through micro-batches
        2. Compute GRPO loss for each
        3. Backpropagate and accumulate gradients
        4. Clip gradients and update parameters

    Args:
        policy_model: Current policy À_¸ to train.
        old_model: Old policy À_old (frozen, for importance sampling).
        ref_model: Reference policy À_ref (frozen, for KL penalty).
        optimizer: Optimizer for policy_model.
        policy_batches: List of PolicyBatch objects from rollout.
        config: Configuration with gradient clipping, num_ppo_epochs, etc.

    Returns:
        Dictionary of training metrics:
            - loss: Average loss across all micro-batches and epochs
            - policy_loss: Average policy gradient component
            - kl_penalty: Average KL divergence penalty
            - entropy: Average entropy
            - grad_norm: Gradient norm (after clipping)
            - clip_fraction: Fraction of ratios that were clipped
            - approx_kl: Approximate KL divergence

    Algorithm:
        For epoch in range(num_ppo_epochs):
            accumulated_loss = 0
            accumulated_metrics = {}

            For each micro_batch in policy_batches:
                # Forward pass
                policy_log_probs = compute_policy_log_probs(
                    policy_model, micro_batch.input_ids, ...)

                ref_log_probs = compute_reference_log_probs(
                    ref_model, micro_batch.input_ids, ...)

                # Compute loss
                loss, metrics = compute_grpo_loss(
                    policy_log_probs,
                    micro_batch.old_log_probs,  # Cached from rollout
                    ref_log_probs,
                    micro_batch.advantages,
                    micro_batch.target_mask,
                    config
                )

                # Backward pass (with gradient accumulation)
                loss = loss / gradient_accumulation_steps
                loss.backward()

                accumulated_loss += loss.item()
                accumulate_metrics(accumulated_metrics, metrics)

                # Update parameters every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    grad_norm = clip_gradients(policy_model, config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

        Return averaged metrics

    Notes:
        - Multiple PPO epochs reuse the same rollout data (off-policy correction).
        - Gradient accumulation allows effective batch size > GPU memory limit.
        - Gradient clipping prevents training instability.
        - old_log_probs are cached and don't need recomputation.
        - Only policy_model is trained; old_model and ref_model are frozen.
    """
    if len(policy_batches) == 0:
        return {}

    policy_model.train()
    old_model.eval()
    ref_model.eval()

    num_epochs = max(1, config.num_ppo_epochs)
    grad_accum_steps = max(1, config.gradient_accumulation_steps)

    metrics_accum: Dict[str, float] = {}
    metrics_count = 0.0
    grad_norms: List[float] = []

    optimizer.zero_grad()
    accumulation_step = 0

    for _ in range(num_epochs):
        for batch in policy_batches:
            policy_log_probs, policy_logits = compute_policy_log_probs(
                policy_model,
                batch.input_ids,
                batch.attention_mask,
                batch.target_mask,
            )
            ref_log_probs = compute_reference_log_probs(
                ref_model,
                batch.input_ids,
                batch.attention_mask,
                batch.target_mask,
            )

            loss, step_metrics = compute_grpo_loss(
                policy_log_probs,
                batch.old_log_probs,
                ref_log_probs,
                batch.advantages,
                batch.target_mask,
                config,
                policy_logits=policy_logits,
            )

            loss = loss / grad_accum_steps
            loss.backward()

            accumulation_step += 1
            for key, value in step_metrics.items():
                metrics_accum[key] = metrics_accum.get(key, 0.0) + float(value)
            metrics_count += 1.0

            if accumulation_step % grad_accum_steps == 0:
                grad_norm = clip_gradients(policy_model, config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                grad_norms.append(grad_norm)

    if accumulation_step % grad_accum_steps != 0:
        grad_norm = clip_gradients(policy_model, config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        grad_norms.append(grad_norm)

    if metrics_count == 0:
        return {}

    averaged = {k: v / metrics_count for k, v in metrics_accum.items()}
    if grad_norms:
        averaged["grad_norm"] = sum(grad_norms) / len(grad_norms)
    return averaged


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    Clip gradients by global norm.

    Args:
        model: Model whose gradients to clip.
        max_norm: Maximum allowed gradient norm.

    Returns:
        Total gradient norm before clipping.

    Notes:
        - Uses torch.nn.utils.clip_grad_norm_.
        - Clipping prevents gradient explosion.
    """
    if max_norm <= 0:
        return 0.0

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if isinstance(grad_norm, torch.Tensor):
        return float(grad_norm.detach().cpu())
    return float(grad_norm)


# ============================================================================
# Training Loop
# ============================================================================


def train_step(
    policy_model: PreTrainedModel,
    old_model: PreTrainedModel,
    ref_model: PreTrainedModel,
    optimizer: Optimizer,
    dataset: List[str],
    tokenizer: PreTrainedTokenizer,
    reward_fn: callable,
    config: GrpoConfig,
    step: int,
) -> Dict[str, Any]:
    """
    Execute one full training step of GRPO-Zero.

    This is the main training step that combines rollout and policy update:
        1. Sample questions
        2. Generate completions (rollout)
        3. Compute rewards
        4. Normalize rewards per group
        5. Build policy batches
        6. Update policy
        7. Update old model if needed
        8. Update reference model if needed

    Args:
        policy_model: Current policy À_¸ to train.
        old_model: Old policy À_old for rollout.
        ref_model: Reference policy À_ref for KL penalty.
        optimizer: Optimizer for policy_model.
        dataset: Dataset of questions to sample from.
        tokenizer: Tokenizer for encoding/decoding.
        reward_fn: Function to compute rewards.
        config: GRPO configuration.
        step: Current training step (for logging/checkpointing).

    Returns:
        Dictionary of metrics:
            - Training metrics from update_policy
            - reward_mean: Average raw reward
            - reward_std: Std of raw rewards
            - reward_max: Maximum reward
            - reward_min: Minimum reward
            - num_episodes: Number of episodes collected (B*G)

    Algorithm:
        1. Sample minibatch of questions
        2. Rollout to get completions and rewards
        3. Normalize rewards per group
        4. (Optional) Whiten advantages
        5. Build policy batches with proper masking
        6. Update policy using GRPO loss
        7. Update old model: À_old  À_¸ (if step % old_update_freq == 0)
        8. Update ref model: À_ref  À_¸ (if step % ref_update_freq == 0)
        9. Collect and return metrics

    Notes:
        - This function is called repeatedly by the outer training loop.
        - Combines all components: rollout, reward, normalization, update.
        - old_model is typically updated every step (on-policy).
        - ref_model is updated less frequently to maintain stable KL reference.
    """
    minibatch = sample_questions(dataset, config.batch_size, tokenizer)
    responses = rollout_batch(
        old_model, minibatch, config, tokenizer, reward_fn
    )

    if len(responses) == 0:
        return {"step": step, "num_episodes": 0}

    normalized_rewards = normalize_rewards_per_group(
        responses, config.group_size, config.reward_norm_eps
    )

    advantages_tensor = torch.tensor(
        normalized_rewards, dtype=torch.float32
    )
    model_device = next(policy_model.parameters()).device
    if config.advantage_whitening and advantages_tensor.numel() > 0:
        advantages_tensor = whiten_advantages(
            advantages_tensor.to(model_device),
            eps=config.reward_norm_eps,
        )
    normalized_advantages = advantages_tensor.detach().cpu().tolist()

    policy_batches = build_policy_batches(
        responses,
        normalized_advantages,
        old_model,
        tokenizer,
        micro_batch_size=config.batch_size,
    )

    update_metrics = update_policy(
        policy_model,
        old_model,
        ref_model,
        optimizer,
        policy_batches,
        config,
    )

    rewards = [resp.reward for resp in responses]
    reward_mean = sum(rewards) / len(rewards)
    reward_std = (
        torch.tensor(rewards).std(unbiased=False).item()
        if len(rewards) > 1
        else 0.0
    )
    metrics = {
        **update_metrics,
        "reward_mean": float(reward_mean),
        "reward_std": float(reward_std),
        "reward_max": float(max(rewards)),
        "reward_min": float(min(rewards)),
        "num_episodes": float(len(rewards)),
        "step": float(step),
    }

    if config.old_update_freq > 0 and (step + 1) % config.old_update_freq == 0:
        update_old_model(policy_model, old_model)

    if config.ref_update_freq > 0 and (step + 1) % config.ref_update_freq == 0:
        update_reference_model(policy_model, ref_model)

    return metrics


def train_loop(
    policy_model: PreTrainedModel,
    dataset: List[str],
    tokenizer: PreTrainedTokenizer,
    reward_fn: callable,
    config: GrpoConfig,
    num_steps: int,
    checkpoint_dir: str,
) -> None:
    """
    Main training loop for GRPO-Zero.

    This is the top-level training function that orchestrates the entire
    GRPO-Zero training process.

    Args:
        policy_model: Model to train (À_¸).
        dataset: Dataset of questions.
        tokenizer: Tokenizer.
        reward_fn: Reward function (verification task).
        config: GRPO configuration.
        num_steps: Number of training steps to run.
        checkpoint_dir: Directory to save checkpoints.

    Algorithm:
        1. Initialize old_model as copy of policy_model
        2. Initialize ref_model as copy of policy_model (frozen)
        3. Initialize optimizer
        4. For step in range(num_steps):
            a. Execute train_step
            b. Log metrics (if step % log_freq == 0)
            c. Save checkpoint (if step % checkpoint_freq == 0)
        5. Save final checkpoint

    Notes:
        - old_model and ref_model are deep copies of policy_model.
        - ref_model is frozen (requires_grad=False for all parameters).
        - Optimizer is typically AdamW with low learning rate (~1e-5).
        - Checkpoints should include: model state, optimizer state, step number.
    """
    if num_steps <= 0:
        return

    optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=config.learning_rate
    )
    old_model = initialize_old_model(policy_model)
    ref_model = initialize_reference_model(policy_model)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    for step in range(num_steps):
        metrics = train_step(
            policy_model,
            old_model,
            ref_model,
            optimizer,
            dataset,
            tokenizer,
            reward_fn,
            config,
            step,
        )

        if config.log_freq > 0 and (step + 1) % config.log_freq == 0:
            log_metrics(metrics, step + 1)

        if (
            config.checkpoint_freq > 0
            and (step + 1) % config.checkpoint_freq == 0
        ):
            save_checkpoint(policy_model, optimizer, step + 1, checkpoint_dir)

    save_checkpoint(policy_model, optimizer, num_steps, checkpoint_dir)


def initialize_old_model(policy_model: PreTrainedModel) -> PreTrainedModel:
    """
    Create a copy of policy model for π_old.

    Args:
        policy_model: Current policy model (HuggingFace PreTrainedModel).

    Returns:
        Deep copy of policy_model (not frozen, but not optimized).

    Notes:
        - old_model is updated periodically by copying from policy_model.
        - Not trainable (not passed to optimizer), but gradients not disabled.
        - Uses copy.deepcopy for HuggingFace models to ensure complete independence.
    """
    old_model = copy.deepcopy(policy_model)
    old_model.eval()
    return old_model


def initialize_reference_model(policy_model: PreTrainedModel) -> PreTrainedModel:
    """
    Create a frozen copy of policy model for π_ref.

    Args:
        policy_model: Current policy model (HuggingFace PreTrainedModel).

    Returns:
        Deep copy of policy_model with all gradients disabled.

    Notes:
        - Reference model is completely frozen (requires_grad=False).
        - Used only for computing KL penalty.
        - Updated less frequently than old_model.
        - Uses copy.deepcopy and then freezes all parameters.
    """
    ref_model = copy.deepcopy(policy_model)
    freeze_model(ref_model)
    return ref_model


def update_old_model(policy_model: PreTrainedModel, old_model: PreTrainedModel) -> None:
    """
    Update old model by copying parameters from policy model.

    Args:
        policy_model: Current policy model (source, HuggingFace PreTrainedModel).
        old_model: Old model to update (destination, HuggingFace PreTrainedModel).

    Notes:
        - Simple parameter copy: old_model.load_state_dict(policy_model.state_dict()).
        - Typically done every step (on-policy updates).
        - Works with HuggingFace PreTrainedModel instances.
    """
    old_model.load_state_dict(policy_model.state_dict())
    old_model.eval()


def update_reference_model(
    policy_model: PreTrainedModel, ref_model: PreTrainedModel
) -> None:
    """
    Update reference model by copying parameters from policy model.

    Args:
        policy_model: Current policy model (source, HuggingFace PreTrainedModel).
        ref_model: Reference model to update (destination, HuggingFace PreTrainedModel).

    Notes:
        - Same as update_old_model, but done less frequently.
        - Maintains frozen status (requires_grad stays False).
        - Works with HuggingFace PreTrainedModel instances.
    """
    ref_model.load_state_dict(policy_model.state_dict())
    freeze_model(ref_model)


def save_checkpoint(
    policy_model: PreTrainedModel, optimizer: Optimizer, step: int, checkpoint_dir: str
) -> None:
    """
    Save model checkpoint.

    Args:
        policy_model: Model to save (HuggingFace PreTrainedModel).
        optimizer: Optimizer to save.
        step: Current training step.
        checkpoint_dir: Directory to save checkpoint.

    Notes:
        - Can use HuggingFace's save_pretrained() or torch.save().
        - Also saves optimizer state_dict and step number.
        - Filename format: checkpoint_step_{step}/
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = path / f"step_{step}.pt"

    torch.save(
        {
            "model": policy_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        checkpoint_file,
    )


def load_checkpoint(
    checkpoint_path: str, policy_model: PreTrainedModel, optimizer: Optimizer
) -> int:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory.
        policy_model: Model to load into (HuggingFace PreTrainedModel).
        optimizer: Optimizer to load into.

    Returns:
        Training step number from checkpoint.

    Notes:
        - Can use HuggingFace's from_pretrained() or torch.load().
        - Loads model state_dict, optimizer state_dict, and step number.
        - Used for resuming training.
    """
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.is_dir():
        raise ValueError("checkpoint_path should point to a checkpoint file")

    device = next(policy_model.parameters()).device
    checkpoint = torch.load(checkpoint_file, map_location=device)

    policy_model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return int(checkpoint.get("step", 0))


# ============================================================================
# Utility Functions
# ============================================================================


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of parameters in model.

    Args:
        model: Model to count parameters of.

    Returns:
        Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def freeze_model(model: nn.Module) -> None:
    """
    Freeze all parameters in model (set requires_grad=False).

    Args:
        model: Model to freeze.
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def log_metrics(metrics: Dict[str, Any], step: int) -> None:
    """
    Log training metrics.

    Args:
        metrics: Dictionary of metrics to log.
        step: Current training step.

    Notes:
        - Can use wandb, tensorboard, or simple print statements.
        - Should log: loss, reward statistics, KL, entropy, grad norm, etc.
    """
    formatted = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
    print(f"[step {step}] {formatted}")
