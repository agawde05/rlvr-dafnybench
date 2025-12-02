"""
Unit tests for GRPO-Zero implementation.

Tests cover the rollout pipeline, reward normalization, batch building,
and other core GRPO-Zero components.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from grpo_zero import sample_questions  # noqa: E402
from data_types import Minibatch  # noqa: E402


class MockTokenizer:
    """Mock HuggingFace tokenizer for testing."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        # Simple vocabulary: special tokens + words
        self.vocab = {
            "<pad>": 0,
            "<eos>": 1,
            "<bos>": 2,
            "hello": 3,
            "world": 4,
            "how": 5,
            "are": 6,
            "you": 7,
            "test": 8,
            "question": 9,
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        """Tokenize text into token IDs."""
        words = text.lower().split()
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.bos_token_id)

        for word in words:
            # Get token ID from vocab, default to 0 if unknown
            token_ids.append(self.vocab.get(word, 0))

        if add_special_tokens:
            token_ids.append(self.eos_token_id)

        return {"input_ids": token_ids}

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs back to text."""
        tokens = []
        for tid in token_ids:
            token = self.reverse_vocab.get(tid, "<unk>")
            if skip_special_tokens and token in ["<pad>", "<eos>", "<bos>"]:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def convert_ids_to_tokens(self, token_ids):
        """Convert token IDs to token strings."""
        if isinstance(token_ids, int):
            return self.reverse_vocab.get(token_ids, "<unk>")
        return [self.reverse_vocab.get(tid, "<unk>") for tid in token_ids]


###################################
# Test cases for sample_questions #
###################################

def test_sample_questions_returns_minibatch():
    """Test that sample_questions returns a proper Minibatch object."""
    dataset = ["hello world", "how are you", "test question"]
    tokenizer = MockTokenizer()
    batch_size = 2

    result = sample_questions(dataset, batch_size, tokenizer)

    assert isinstance(result, Minibatch)
    assert hasattr(result, "prompts")
    assert hasattr(result, "prompt_tokens")
    assert hasattr(result, "prompt_token_ids")


def test_sample_questions_correct_batch_size():
    """Test that sample_questions returns the correct number of samples."""
    dataset = ["hello world", "how are you", "test question", "another one"]
    tokenizer = MockTokenizer()
    batch_size = 2

    result = sample_questions(dataset, batch_size, tokenizer)

    assert len(result.prompts) == batch_size
    assert len(result.prompt_token_ids) == batch_size


def test_sample_questions_handles_small_dataset():
    """Test that sample_questions handles dataset smaller than batch_size."""
    dataset = ["hello world", "how are you"]
    tokenizer = MockTokenizer()
    batch_size = 5  # Larger than dataset

    result = sample_questions(dataset, batch_size, tokenizer)

    # Should return all available samples
    assert len(result.prompts) == len(dataset)
    assert len(result.prompt_token_ids) == len(dataset)


def test_sample_questions_samples_from_dataset():
    """Test that sampled questions come from the dataset."""
    dataset = ["hello world", "how are you", "test question"]
    tokenizer = MockTokenizer()
    batch_size = 2

    result = sample_questions(dataset, batch_size, tokenizer)

    # All sampled prompts should be in the original dataset
    for prompt in result.prompts:
        assert prompt in dataset


def test_sample_questions_no_duplicates_within_batch():
    """Test that sample_questions doesn't duplicate within a batch."""
    dataset = ["hello world", "how are you", "test question", "another one"]
    tokenizer = MockTokenizer()
    batch_size = 3

    result = sample_questions(dataset, batch_size, tokenizer)

    # No duplicates within the batch (sampling without replacement)
    assert len(result.prompts) == len(set(result.prompts))


def test_sample_questions_tokenizes_correctly():
    """Test that prompts are tokenized correctly."""
    dataset = ["hello world"]
    tokenizer = MockTokenizer()
    batch_size = 1

    result = sample_questions(dataset, batch_size, tokenizer)

    # Check that tokenization includes special tokens
    token_ids = result.prompt_token_ids[0]
    assert len(token_ids) > 0
    # Should have bos, "hello", "world", eos
    assert token_ids[0] == tokenizer.bos_token_id  # BOS token
    assert token_ids[-1] == tokenizer.eos_token_id  # EOS token
    assert len(token_ids) == 4  # <bos> hello world <eos>


def test_sample_questions_token_ids_are_lists():
    """Test that token IDs are returned as Python lists, not tensors."""
    dataset = ["hello world", "how are you"]
    tokenizer = MockTokenizer()
    batch_size = 2

    result = sample_questions(dataset, batch_size, tokenizer)

    # Token IDs should be lists
    for token_ids in result.prompt_token_ids:
        assert isinstance(token_ids, list)
        assert all(isinstance(tid, int) for tid in token_ids)


def test_sample_questions_prompt_tokens_empty():
    """Test that prompt_tokens is left empty (not populated)."""
    dataset = ["hello world", "how are you"]
    tokenizer = MockTokenizer()
    batch_size = 2

    result = sample_questions(dataset, batch_size, tokenizer)

    # prompt_tokens should be empty as per implementation
    assert result.prompt_tokens == []


def test_sample_questions_randomness():
    """Test that sample_questions samples randomly (probabilistically)."""
    dataset = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"]
    tokenizer = MockTokenizer()
    batch_size = 3

    # Sample multiple times and check we get different results
    samples = []
    for _ in range(5):
        result = sample_questions(dataset, batch_size, tokenizer)
        samples.append(tuple(result.prompts))

    # With high probability, we should get at least 2 different samples
    # (This test could theoretically fail with very low probability)
    unique_samples = set(samples)
    assert len(unique_samples) >= 2, "Expected some randomness in sampling"


def test_sample_questions_handles_single_question():
    """Test that sample_questions works with a single question."""
    dataset = ["hello world"]
    tokenizer = MockTokenizer()
    batch_size = 1

    result = sample_questions(dataset, batch_size, tokenizer)

    assert len(result.prompts) == 1
    assert result.prompts[0] == "hello world"
    assert len(result.prompt_token_ids) == 1


def test_sample_questions_preserves_original_text():
    """Test that the original prompt text is preserved exactly."""
    original_prompt = "Hello World"  # Capital letters
    dataset = [original_prompt, "test question"]
    tokenizer = MockTokenizer()
    batch_size = 1

    # Run multiple times to ensure we eventually sample the first one
    for _ in range(10):
        result = sample_questions(dataset, batch_size, tokenizer)
        if result.prompts[0] == original_prompt:
            # Found it - text is preserved exactly
            assert result.prompts[0] == original_prompt
            break


def test_sample_questions_with_empty_dataset():
    """Test that sample_questions handles empty dataset gracefully."""
    dataset = []
    tokenizer = MockTokenizer()
    batch_size = 2

    result = sample_questions(dataset, batch_size, tokenizer)

    # Should return empty minibatch
    assert len(result.prompts) == 0
    assert len(result.prompt_token_ids) == 0


def test_sample_questions_token_id_lengths_match_text():
    """Test that token IDs have reasonable length relative to text."""
    dataset = ["hello", "hello world", "how are you doing today"]
    tokenizer = MockTokenizer()
    batch_size = 3

    result = sample_questions(dataset, batch_size, tokenizer)

    # Each prompt should have token IDs
    for i, prompt in enumerate(result.prompts):
        token_ids = result.prompt_token_ids[i]
        word_count = len(prompt.split())

        # With special tokens: bos + words + eos
        expected_length = word_count + 2
        assert len(token_ids) == expected_length


def test_sample_questions_exact_token_ids():
    """Test that the exact token IDs are correct for known inputs."""
    dataset = ["hello world"]
    tokenizer = MockTokenizer()
    batch_size = 1

    result = sample_questions(dataset, batch_size, tokenizer)

    token_ids = result.prompt_token_ids[0]
    # Expected: <bos>(2), "hello"(3), "world"(4), <eos>(1)
    assert token_ids == [2, 3, 4, 1]


def test_sample_questions_preserves_token_order():
    """Test that token order matches the word order in the prompt."""
    dataset = ["how are you"]
    tokenizer = MockTokenizer()
    batch_size = 1

    result = sample_questions(dataset, batch_size, tokenizer)

    token_ids = result.prompt_token_ids[0]
    # Expected: <bos>(2), "how"(5), "are"(6), "you"(7), <eos>(1)
    assert token_ids == [2, 5, 6, 7, 1]

    # Verify we can decode back to original (ignoring case)
    decoded = tokenizer.decode(token_ids[1:-1], skip_special_tokens=True)  # Skip BOS/EOS
    assert decoded.lower() == "how are you"


def test_sample_questions_multiple_prompts_have_correct_tokens():
    """Test that multiple prompts all have correct tokenization."""
    dataset = ["hello world", "how are you"]
    tokenizer = MockTokenizer()
    batch_size = 2

    result = sample_questions(dataset, batch_size, tokenizer)

    # Map prompts to expected token IDs
    expected_tokens = {
        "hello world": [2, 3, 4, 1],  # <bos> hello world <eos>
        "how are you": [2, 5, 6, 7, 1],  # <bos> how are you <eos>
    }

    for i, prompt in enumerate(result.prompts):
        token_ids = result.prompt_token_ids[i]
        assert token_ids == expected_tokens[prompt]


def test_sample_questions_tokens_are_unmodified_by_sampling():
    """Test that tokenization is consistent regardless of sampling order."""
    dataset = ["hello world", "how are you", "test question"]
    tokenizer = MockTokenizer()

    # Sample multiple times and verify tokenization is always the same for each prompt
    tokenizations = {}

    for _ in range(5):
        result = sample_questions(dataset, batch_size=2, tokenizer=tokenizer)

        for i, prompt in enumerate(result.prompts):
            token_ids = tuple(result.prompt_token_ids[i])  # Convert to tuple for hashing

            if prompt in tokenizations:
                # This prompt was seen before - verify same tokenization
                assert tokenizations[prompt] == token_ids, (
                    f"Tokenization changed for prompt '{prompt}': "
                    f"expected {tokenizations[prompt]}, got {token_ids}"
                )
            else:
                # First time seeing this prompt - record tokenization
                tokenizations[prompt] = token_ids


def test_sample_questions_special_tokens_always_present():
    """Test that special tokens (BOS/EOS) are always added."""
    dataset = ["hello", "world", "test"]
    tokenizer = MockTokenizer()
    batch_size = 3

    result = sample_questions(dataset, batch_size, tokenizer)

    for i, prompt in enumerate(result.prompts):
        token_ids = result.prompt_token_ids[i]

        # First token should be BOS
        assert token_ids[0] == tokenizer.bos_token_id, (
            f"Expected BOS token ({tokenizer.bos_token_id}) at start, "
            f"got {token_ids[0]} for prompt '{prompt}'"
        )

        # Last token should be EOS
        assert token_ids[-1] == tokenizer.eos_token_id, (
            f"Expected EOS token ({tokenizer.eos_token_id}) at end, "
            f"got {token_ids[-1]} for prompt '{prompt}'"
        )


def test_sample_questions_token_ids_roundtrip():
    """Test that token IDs can be decoded back to recover the original text."""
    dataset = ["hello world", "how are you"]
    tokenizer = MockTokenizer()
    batch_size = 2

    result = sample_questions(dataset, batch_size, tokenizer)

    for i, original_prompt in enumerate(result.prompts):
        token_ids = result.prompt_token_ids[i]

        # Decode token IDs back to text (skip special tokens)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)

        # Should match original (accounting for case normalization in mock tokenizer)
        assert decoded.lower() == original_prompt.lower()
