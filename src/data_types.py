from dataclasses import dataclass, field
from typing import Any, List, Dict

type Tokens = List[str]
type TokenIds = List[int]


@dataclass
class Response:
    """A single model-generated response to a prompt, with associated reward."""

    prompt: str  # Input prompt text
    full_text: str  # prompt + generation
    completion_text: str  # generation only
    prompt_token_ids: TokenIds  # Tokenized prompt
    prompt_tokens: Tokens  # Prompt tokens (for debugging)
    generated_token_ids: TokenIds  # Generated tokens only
    is_complete: bool  # Whether generation finished naturally
    reward: float  # Scalar reward (normalized during training)
    reward_components: Dict[str, Any]  # Breakdown for logging and failure reasons


@dataclass
class Minibatch:
    """Stores a minibatch of questions passed to the model for training."""

    prompts: List[str]  # List of input prompt texts
    prompt_tokens: List[Tokens]  # Tokenized prompts (for debugging)
    prompt_token_ids: List[TokenIds]  # Tokenized prompts


@dataclass
class GrpoConfig:
    """Configuration for GRPO-Zero training. Inline metadata documents each field."""

    # Sampling parameters
    group_size: int = field(
        default=8,
        metadata={
            "help": "Completions sampled per question (G) for group-relative reward normalization."
        },
    )
    microbatch_size: int = field(
        default=8,
        metadata={"help": "Number of completions per microbatch."},
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    temperature: float = field(
        default=0.8,
        metadata={"help": "Sampling temperature; higher values increase randomness."},
    )
    top_p: float = field(
        default=0.95,
        metadata={
            "help": "Nucleus sampling cutoff; sample from tokens whose cumulative probability <= top_p."
        },
    )

    # Training parameters
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate for the policy optimizer."},
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Number of questions per training batch."},
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Micro-batches to accumulate before each optimizer step."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for clipping to stabilize training."},
    )
    num_ppo_epochs: int = field(
        default=2,
        metadata={
            "help": "Number of epochs to train on each batch of collected rollouts."
        },
    )

    # PPO/GRPO hyperparameters
    clip_ratio: float = field(
        default=0.2,
        metadata={
            "help": "PPO clipping parameter epsilon; limits policy update magnitude."
        },
    )
    kl_coef: float = field(
        default=0.05,
        metadata={
            "help": "Coefficient for KL penalty against the reference model to prevent drift."
        },
    )
    entropy_coef: float = field(
        default=0.01,
        metadata={
            "help": "Entropy bonus coefficient encouraging exploration during training."
        },
    )
    value_loss_coef: float = field(
        default=0.0,
        metadata={
            "help": "Value loss weight (unused in GRPO-Zero but kept for extensibility)."
        },
    )

    # Reward normalization
    reward_norm_eps: float = field(
        default=1e-8,
        metadata={
            "help": "Epsilon for numerical stability in group-relative reward normalization."
        },
    )
    advantage_whitening: bool = field(
        default=True,
        metadata={
            "help": "Whether to whiten advantages across the full batch after group normalization."
        },
    )

    # Model management
    ref_update_freq: int = field(
        default=100,
        metadata={
            "help": "Steps between syncing reference model with current policy; 0 disables updates."
        },
    )
    old_update_freq: int = field(
        default=1,
        metadata={
            "help": "Steps between updating the old policy snapshot used for importance sampling."
        },
    )

    # Checkpointing
    checkpoint_freq: int = field(
        default=100,
        metadata={"help": "Steps between saving model checkpoints."},
    )

    # Logging
    log_freq: int = field(
        default=10,
        metadata={"help": "Steps between metric logging."},
    )

    # Device and precision
    device: str = field(
        default="cuda",
        metadata={"help": "Execution device for training (e.g., 'cuda' or 'cpu')."},
    )
    mixed_precision: bool = field(
        default=True,
        metadata={"help": "Enable mixed precision (fp16) training when supported."},
    )
