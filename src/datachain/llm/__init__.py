from typing import Any

from datachain.llm.engine import LLMError
from datachain.llm.spec import LLMConfigError, LLMSpec


def complete(
    col: str,
    prompt: str | None = None,
    *,
    schema: Any = None,
    context: str | None = None,
    llm: str | None = None,
    retries: int = 1,
    fallback: str | list[str] | None = None,
    **params: Any,
) -> LLMSpec:
    """General generation or structured extraction over a column.

    Output is ``str`` when no ``schema`` is given, the Pydantic ``schema`` model
    when it is, and a 1:N stream meant for ``.gen()`` when ``schema`` is
    ``list[Model]``.

    Args:
        col (str): Input column. Encoding follows its type: text files and
            strings as text, images/frames as vision input, PDFs as documents;
            other binary files error. See the Inputs section.
        prompt (str | None): Instruction text added before the input.
        schema (type | None): Pydantic model (or ``list[Model]``) for structured
            output. When omitted, the output is plain ``str``.
        context (str | None): Column whose value is serialized into the prompt.
        llm (str | None): Per-call model override, taking precedence over
            ``settings(llm=...)``.
        retries (int): Transient and schema-validation retry budget.
        fallback (str | list[str] | None): Model string(s) tried if the primary
            model fails.
        params (Any): Extra arguments forwarded to the underlying model call.

    Returns:
        LLMSpec: A spec used inside ``.map()`` (1:1) or ``.gen()`` (1:N).

    Example:
        ```py
        .map(scene=llm.complete("file", schema=Scene))
        .gen(chunk=llm.complete("file", schema=list[Chunk], prompt="split"))
        ```
    """
    return LLMSpec(
        kind="complete",
        col=col,
        prompt=prompt,
        schema=schema,
        context_col=context,
        llm=llm,
        retries=retries,
        fallback=fallback,
        params=params,
    )


def extract(
    col: str,
    schema: Any,
    prompt: str | None = None,
    *,
    context: str | None = None,
    llm: str | None = None,
    retries: int = 1,
    fallback: str | list[str] | None = None,
    **params: Any,
) -> LLMSpec:
    """Schema-driven extraction. Alias for ``complete`` with a required ``schema``.

    Args:
        col (str): Input column passed to the model.
        schema (type): Pydantic model describing the fields to extract.
        prompt (str | None): Optional extra instruction text.
        context (str | None): Column whose value is serialized into the prompt.
        llm (str | None): Per-call model override.
        retries (int): Transient and schema-validation retry budget.
        fallback (str | list[str] | None): Model string(s) tried on failure.
        params (Any): Extra arguments forwarded to the underlying model call.

    Returns:
        LLMSpec: A spec whose output type is ``schema``.
    """
    return complete(
        col,
        prompt,
        schema=schema,
        context=context,
        llm=llm,
        retries=retries,
        fallback=fallback,
        **params,
    )


def classify(
    col: str,
    into: list[str],
    prompt: str | None = None,
    *,
    context: str | None = None,
    llm: str | None = None,
    retries: int = 1,
    fallback: str | list[str] | None = None,
    **params: Any,
) -> LLMSpec:
    """Categorize a column into exactly one of the given labels.

    Args:
        col (str): Input column passed to the model.
        into (list[str]): Allowed categories; the output is constrained to one.
        prompt (str | None): Optional extra guidance added to the instruction.
        context (str | None): Column whose value is serialized into the prompt.
        llm (str | None): Per-call model override.
        retries (int): Transient and schema-validation retry budget.
        fallback (str | list[str] | None): Model string(s) tried on failure.
        params (Any): Extra arguments forwarded to the underlying model call.

    Returns:
        LLMSpec: A spec whose output type is ``str``.
    """
    return LLMSpec(
        kind="classify",
        col=col,
        prompt=prompt,
        into=list(into),
        context_col=context,
        llm=llm,
        retries=retries,
        fallback=fallback,
        params=params,
    )


def score(
    col: str,
    prompt: str | None = None,
    *,
    context: str | None = None,
    llm: str | None = None,
    retries: int = 1,
    fallback: str | list[str] | None = None,
    **params: Any,
) -> LLMSpec:
    """Numeric scoring of a column against a prompt.

    Args:
        col (str): Input column passed to the model.
        prompt (str | None): The scoring criterion (e.g. ``"accident risk 0..1"``).
        context (str | None): Column whose value is serialized into the prompt.
        llm (str | None): Per-call model override.
        retries (int): Transient and schema-validation retry budget.
        fallback (str | list[str] | None): Model string(s) tried on failure.
        params (Any): Extra arguments forwarded to the underlying model call.

    Returns:
        LLMSpec: A spec whose output type is ``float``.
    """
    return LLMSpec(
        kind="score",
        col=col,
        prompt=prompt,
        context_col=context,
        llm=llm,
        retries=retries,
        fallback=fallback,
        params=params,
    )


def embed(
    col: str,
    *,
    llm: str | None = None,
    retries: int = 1,
    fallback: str | list[str] | None = None,
    **params: Any,
) -> LLMSpec:
    """Embed a column into a vector.

    Args:
        col (str): Input column to embed.
        llm (str | None): Per-call embedding-model override.
        retries (int): Transient retry budget.
        fallback (str | list[str] | None): Model string(s) tried on failure.
        params (Any): Extra arguments forwarded to the underlying model call.

    Returns:
        LLMSpec: A spec whose output type is ``list[float]``.
    """
    return LLMSpec(
        kind="embed",
        col=col,
        llm=llm,
        retries=retries,
        fallback=fallback,
        params=params,
    )


__all__ = [
    "LLMConfigError",
    "LLMError",
    "LLMSpec",
    "classify",
    "complete",
    "embed",
    "extract",
    "score",
]
