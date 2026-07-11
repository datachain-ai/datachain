from collections.abc import Sequence
from typing import Any

from datachain.llm.content import Media
from datachain.llm.engine import LLMError
from datachain.llm.spec import LLMConfigError, LLMSpec
from datachain.llm.types import Usage


def complete(
    col: str,
    prompt: str | None = None,
    *,
    schema: Any = None,
    context: str | None = None,
    type: Media | None = None,
    llm: str | None = None,
    retries: int = 1,
    fallback: str | list[str] | None = None,
    include_usage: bool = False,
    **params: Any,
) -> LLMSpec:
    """Generate text or extract a structured object from a column.

    Output is ``str`` when no ``schema`` is given and the Pydantic ``schema`` model
    when it is. A ``list[Model]`` schema returns many items: use ``.gen()`` to fan
    them out (one row each) or ``.map()`` to keep them as one ``list[Model]`` column.

    Args:
        col (str): Input column. Its type decides the encoding (text files and
            strings as text, images/frames as vision input); for raw ``bytes`` or
            an untyped ``File`` set ``type``. See the Inputs section.
        prompt (str | None): Instruction text added before the input.
        schema (type | None): Pydantic model (or ``list[Model]``) for structured
            output. When omitted, the output is plain ``str``.
        context (str | None): Column whose value is serialized into the prompt.
        type ("text" | "image" | "document" | None): Override how ``col`` is
            encoded. Needed for raw ``bytes`` or an untyped ``File``: e.g.
            ``type="image"`` for an image-bytes column, ``type="document"`` for
            a PDF (sent to a document-capable model).
        llm (str | None): Per-call model override, taking precedence over
            ``settings(llm=...)``.
        retries (int): Transient and schema-validation retry budget.
        fallback (str | list[str] | None): Model string(s) tried if the primary
            model fails.
        include_usage (bool): When True, the call returns a tuple
            ``(value, dc.llm.Usage)``, used with the multi-output form that names
            both columns:
            ``.map(llm.complete(...), output={"res": Scene, "tok": dc.llm.Usage})``.
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
        type=type,
        llm=llm,
        retries=retries,
        fallback=fallback,
        include_usage=include_usage,
        params=params,
    )


def classify(
    col: str,
    into: Sequence[str],
    prompt: str | None = None,
    *,
    context: str | None = None,
    type: Media | None = None,
    llm: str | None = None,
    retries: int = 1,
    fallback: str | list[str] | None = None,
    include_usage: bool = False,
    **params: Any,
) -> LLMSpec:
    """Categorize a column into exactly one of the given labels.

    Args:
        col (str): Input column passed to the model.
        into (list[str]): Allowed categories; the output is constrained to one.
        prompt (str | None): Optional extra guidance added to the instruction.
        context (str | None): Column whose value is serialized into the prompt.
        type ("text" | "image" | "document" | None): Override how ``col`` is encoded
            (see ``complete``).
        llm (str | None): Per-call model override.
        retries (int): Transient and schema-validation retry budget.
        fallback (str | list[str] | None): Model string(s) tried on failure.
        include_usage (bool): Also emit a ``dc.llm.Usage`` column (see ``complete``).
        params (Any): Extra arguments forwarded to the underlying model call.

    Returns:
        LLMSpec: A spec whose output type is ``str``.
    """
    if isinstance(into, str):
        raise ValueError(  # noqa: TRY004 - a config value error, not a type guard
            "llm.classify(into=...) expects a list of categories, not a single string"
        )
    return LLMSpec(
        kind="classify",
        col=col,
        prompt=prompt,
        into=list(into),
        context_col=context,
        type=type,
        llm=llm,
        retries=retries,
        fallback=fallback,
        include_usage=include_usage,
        params=params,
    )


def score(
    col: str,
    prompt: str | None = None,
    *,
    context: str | None = None,
    type: Media | None = None,
    llm: str | None = None,
    retries: int = 1,
    fallback: str | list[str] | None = None,
    include_usage: bool = False,
    **params: Any,
) -> LLMSpec:
    """Numeric scoring of a column against a prompt.

    Args:
        col (str): Input column passed to the model.
        prompt (str | None): The scoring criterion (e.g. ``"accident risk 0..1"``).
        context (str | None): Column whose value is serialized into the prompt.
        type ("text" | "image" | "document" | None): Override how ``col`` is encoded
            (see ``complete``).
        llm (str | None): Per-call model override.
        retries (int): Transient and schema-validation retry budget.
        fallback (str | list[str] | None): Model string(s) tried on failure.
        include_usage (bool): Also emit a ``dc.llm.Usage`` column (see ``complete``).
        params (Any): Extra arguments forwarded to the underlying model call.

    Returns:
        LLMSpec: A spec whose output type is ``float``.
    """
    return LLMSpec(
        kind="score",
        col=col,
        prompt=prompt,
        context_col=context,
        type=type,
        llm=llm,
        retries=retries,
        fallback=fallback,
        include_usage=include_usage,
        params=params,
    )


def embed(
    col: str,
    *,
    llm: str | None = None,
    retries: int = 1,
    fallback: str | list[str] | None = None,
    include_usage: bool = False,
    **params: Any,
) -> LLMSpec:
    """Embed a column into a vector.

    Args:
        col (str): Input column to embed.
        llm (str | None): Per-call embedding-model override.
        retries (int): Transient retry budget.
        fallback (str | list[str] | None): Model string(s) tried on failure.
        include_usage (bool): Also emit a ``dc.llm.Usage`` column (see ``complete``).
            Embeddings report no output tokens, so ``output_tokens`` stays 0.
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
        include_usage=include_usage,
        params=params,
    )


__all__ = [
    "LLMConfigError",
    "LLMError",
    "LLMSpec",
    "Usage",
    "classify",
    "complete",
    "embed",
    "score",
]
