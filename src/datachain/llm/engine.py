from functools import cache
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, create_model

from datachain.lib.utils import DataChainError
from datachain.llm.types import Response

T = TypeVar("T", bound=BaseModel)


class LLMError(DataChainError):
    """Raised when a `datachain.llm` operation cannot produce a valid result."""


# Keys datachain.llm sets on the underlying call; users cannot pass them as params.
RESERVED_PARAMS = frozenset(
    {"model", "messages", "input", "num_retries", "fallbacks", "response_format"}
)


def _litellm():
    try:
        import litellm
    except ImportError as exc:  # pragma: no cover - exercised via tests with stub
        raise ImportError(
            "datachain.llm requires the 'litellm' package. "
            "Install it with: pip install 'datachain[llm]'"
        ) from exc
    return litellm


def _fallbacks(fallback: str | list[str] | None) -> list[str] | None:
    if not fallback:  # None, "", or []
        return None
    return [fallback] if isinstance(fallback, str) else list(fallback)


def _has_document(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(m.get("content"), list)
        and any(p.get("type") == "file" for p in m["content"])
        for m in messages
    )


def _check_document_support(
    model: str, fallback: str | list[str] | None, messages: list[dict[str, Any]]
) -> None:
    if not _has_document(messages):
        return
    supports = getattr(_litellm(), "supports_pdf_input", None)
    if supports is None:
        return
    for m in [model, *(_fallbacks(fallback) or [])]:
        try:
            ok = supports(model=m)
        except Exception:  # noqa: BLE001, S112 - a probe must not block the call
            continue
        if not ok:
            raise LLMError(
                f"model '{m}' does not accept document input; use "
                "document-capable models or extract the text first"
            )


def _completion_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    _check_document_support(model, fallback, messages)
    # `params` first so the keys datachain.llm owns always win (also guarded by
    # RESERVED_PARAMS validation upstream). LiteLLM owns retries/backoff/rate limits.
    kwargs: dict[str, Any] = {
        **params,
        "model": model,
        "messages": messages,
        "num_retries": max(retries, 0),
    }
    if (fallbacks := _fallbacks(fallback)) is not None:
        kwargs["fallbacks"] = fallbacks
    return kwargs


def _content(response: Any) -> str:
    if not response.choices:
        raise LLMError("model returned no choices")
    return response.choices[0].message.content or ""


def _finish_reason(response: Any) -> str:
    if not response.choices:
        return ""
    return getattr(response.choices[0], "finish_reason", "") or ""


def _strip_fences(text: str) -> str:
    """Best-effort unwrap of a ```...``` markdown code fence around JSON."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = t[3:]
    newline = t.find("\n")
    if newline != -1 and t[:newline].strip().isalpha():  # drop a ```json language tag
        t = t[newline + 1 :]
    if t.rstrip().endswith("```"):
        t = t.rstrip()[:-3]
    return t.strip()


def _truncated_error(schema_name: str) -> LLMError:
    return LLMError(
        f"model output for '{schema_name}' was truncated "
        "(finish_reason=length); increase max_tokens"
    )


def parse_one(schema: type[T], content: str) -> T:
    """Validate stored text against a model offline (no model call)."""
    last_error: ValidationError | None = None
    for candidate in (content, _strip_fences(content)):
        try:
            return schema.model_validate_json(candidate)
        except ValidationError as exc:
            last_error = exc
    raise LLMError(
        f"stored output could not be parsed as '{schema.__name__}'"
    ) from last_error


def parse_list(item_type: type, content: str) -> list:
    container = _list_container(item_type)
    adapter = _list_adapter(item_type)
    last_error: ValidationError | None = None
    for candidate in (content, _strip_fences(content)):
        try:
            return container.model_validate_json(candidate).items  # type: ignore[attr-defined]
        except ValidationError:
            try:
                return adapter.validate_json(candidate)  # bare top-level array
            except ValidationError as exc:
                last_error = exc
    raise LLMError(
        f"stored output could not be parsed as list[{item_type.__name__}]"
    ) from last_error


def complete_text(
    model: str,
    messages: list[dict[str, Any]],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> str:
    litellm = _litellm()
    kwargs = _completion_kwargs(model, messages, retries, fallback, params)
    return _content(litellm.completion(**kwargs))


def complete_raw(
    model: str,
    messages: list[dict[str, Any]],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> Response:
    litellm = _litellm()
    kwargs = _completion_kwargs(model, messages, retries, fallback, params)
    return Response.from_litellm(litellm.completion(**kwargs))


def complete_structured(
    model: str,
    messages: list[dict[str, Any]],
    schema: type[T],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> T:
    litellm = _litellm()
    kwargs = _completion_kwargs(model, messages, retries, fallback, params)
    kwargs["response_format"] = schema

    # Re-validate schema mismatches here; transient errors are LiteLLM's job above.
    attempts = max(retries, 0) + 1
    last_error: ValidationError | None = None
    for _ in range(attempts):
        resp = litellm.completion(**kwargs)
        content = _content(resp)
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            if _finish_reason(resp) == "length":
                raise _truncated_error(schema.__name__) from exc
            try:
                return schema.model_validate_json(_strip_fences(content))
            except ValidationError:
                last_error = exc
    raise LLMError(
        f"model output did not match schema '{schema.__name__}' "
        f"after {attempts} attempt(s)"
    ) from last_error


@cache
def _list_container(item_type: type) -> type[BaseModel]:
    return create_model("LLMListOutput", items=(list[item_type], ...))  # type: ignore[valid-type]


@cache
def _list_adapter(item_type: type) -> "TypeAdapter[list[Any]]":
    return TypeAdapter(list[item_type])  # type: ignore[valid-type]


@cache
def _classification_model(categories: tuple[str, ...]) -> type[BaseModel]:
    return create_model("LLMClassification", category=(Literal[categories], ...))


@cache
def _score_model() -> type[BaseModel]:
    return create_model("LLMScore", score=(float, Field(allow_inf_nan=False)))


def classify(
    model: str,
    messages: list[dict[str, Any]],
    categories: tuple[str, ...],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> str:
    schema = _classification_model(categories)
    result: Any = complete_structured(
        model, messages, schema, retries, fallback, params
    )
    return result.category


def score(
    model: str,
    messages: list[dict[str, Any]],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> float:
    schema = _score_model()
    result: Any = complete_structured(
        model, messages, schema, retries, fallback, params
    )
    return result.score


def complete_structured_list(
    model: str,
    messages: list[dict[str, Any]],
    item_type: type,
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> list:
    litellm = _litellm()
    container = _list_container(item_type)
    kwargs = _completion_kwargs(model, messages, retries, fallback, params)
    kwargs["response_format"] = container

    attempts = max(retries, 0) + 1
    last_error: ValidationError | None = None
    for _ in range(attempts):
        resp = litellm.completion(**kwargs)
        content = _content(resp)
        try:
            wrapped: Any = container.model_validate_json(content)
            return wrapped.items
        except ValidationError as exc:
            if _finish_reason(resp) == "length":
                raise _truncated_error(f"list[{item_type.__name__}]") from exc
            try:
                return parse_list(item_type, content)  # bare array / fenced fallback
            except LLMError:
                last_error = exc
    raise LLMError(
        f"model output did not match list[{item_type.__name__}] "
        f"after {attempts} attempt(s)"
    ) from last_error


def embed(
    model: str,
    text: str,
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> list[float]:
    litellm = _litellm()
    kwargs: dict[str, Any] = {
        **params,
        "model": model,
        "input": [text],
        "num_retries": max(retries, 0),
    }
    if (fallbacks := _fallbacks(fallback)) is not None:
        kwargs["fallbacks"] = fallbacks
    data = litellm.embedding(**kwargs).data
    if not data:
        raise LLMError("embedding response contained no data")
    item = data[0]
    return list(item["embedding"] if isinstance(item, dict) else item.embedding)
