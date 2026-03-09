from collections.abc import Callable
from typing import Any

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def convert_text(
    text: str | list[str],
    tokenizer: Callable | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    encoder: Callable | None = None,
    device: str | torch.device | None = None,
) -> str | list[str] | torch.Tensor:
    """
    Tokenize and otherwise transform text.

    Args:
        text (str): Text to convert.
        tokenizer (Callable): Tokenizer to use to tokenize objects.
        tokenizer_kwargs (dict): Additional kwargs to pass when calling tokenizer.
        encoder (Callable): Encode text using model.
        device (str or torch.device): Device to use.
    """
    if not tokenizer:
        return text

    if isinstance(text, str):
        text = [text]

    if tokenizer_kwargs:
        res = tokenizer(text, **tokenizer_kwargs)
    else:
        res = tokenizer(text)

    tokens = res.input_ids if isinstance(tokenizer, PreTrainedTokenizerBase) else res
    tokens = torch.as_tensor(tokens).clone().detach()
    if device:
        tokens = tokens.to(device)

    if not encoder:
        return tokens

    return encoder(tokens)
