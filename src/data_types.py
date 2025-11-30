from dataclasses import dataclass
from typing import List, Dict

type Tokens = List[str]
type TokenIds = List[int]

@dataclass
class Response:
    """A single model-generated response to a prompt, with associated reward."""

    prompt: str  # Input prompt text
    full_text: str  # prompt + generation
    prompt_token_ids: TokenIds  # Tokenized prompt
    prompt_tokens: Tokens  # Prompt tokens (for debugging)
    generated_token_ids: TokenIds  # Generated tokens only
    is_complete: bool  # Whether generation finished naturally
    reward: float  # Scalar reward (normalized during training)
    reward_components: Dict[str, float]  # Breakdown for logging


@dataclass
class Minibatch:
    """Stores a minibatch of questions passed to the model for training."""

    prompts: List[str]  # List of input prompt texts
    prompt_tokens: List[Tokens]  # Tokenized prompts (for debugging)
    prompt_token_ids: List[TokenIds]  # Tokenized prompts
    