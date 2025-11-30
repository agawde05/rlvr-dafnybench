from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import torch


def load_model_and_tokenizer(
    model_name: str, dtype: torch.dtype, device: torch.device
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads a pre-trained model and its tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    torch.set_default_device(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device
    )
    return model, tokenizer
