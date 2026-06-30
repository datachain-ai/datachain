from functools import cache
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, create_model

from datachain.lib.utils import DataChainError

T = TypeVar("T", bound=BaseModel)


class LLMError(DataChainError):
    """Raised when a `datachain.llm` operation cannot produce a valid result."""


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
    if fallback is None:
        return None
    return [fallback] if isinstance(fallback, str) else list(fallback)


def _completion_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    # LiteLLM owns transient retries, backoff, and rate-limit handling.
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "num_retries": max(retries, 0),
        **params,
    }
    if (fallbacks := _fallbacks(fallback)) is not None:
        kwargs["fallbacks"] = fallbacks
    return kwargs


def _content(response: Any) -> str:
    if not response.choices:
        raise LLMError("model returned no choices")
    return response.choices[0].message.content or ""


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
        try:
            return schema.model_validate_json(_content(litellm.completion(**kwargs)))
        except ValidationError as exc:
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
    adapter = _list_adapter(item_type)
    kwargs = _completion_kwargs(model, messages, retries, fallback, params)
    kwargs["response_format"] = container

    attempts = max(retries, 0) + 1
    last_error: ValidationError | None = None
    for _ in range(attempts):
        content = _content(litellm.completion(**kwargs))
        try:
            wrapped: Any = container.model_validate_json(content)
            return wrapped.items
        except ValidationError as exc:
            try:
                return adapter.validate_json(content)  # tolerate a bare top-level array
            except ValidationError:
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
        "model": model,
        "input": [text],
        "num_retries": max(retries, 0),
        **params,
    }
    if (fallbacks := _fallbacks(fallback)) is not None:
        kwargs["fallbacks"] = fallbacks
    item = litellm.embedding(**kwargs).data[0]
    return list(item["embedding"] if isinstance(item, dict) else item.embedding)
