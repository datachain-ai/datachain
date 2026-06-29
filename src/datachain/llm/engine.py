from functools import cache
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError, create_model

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
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "num_retries": retries,
        **params,
    }
    if (fallbacks := _fallbacks(fallback)) is not None:
        kwargs["fallbacks"] = fallbacks
    return kwargs


def complete_text(
    model: str,
    messages: list[dict[str, Any]],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> str:
    litellm = _litellm()
    kwargs = _completion_kwargs(model, messages, retries, fallback, params)
    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content
    return content if content is not None else ""


def complete_structured(
    model: str,
    messages: list[dict[str, Any]],
    schema: type[T],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> T:
    """Call the model with a Pydantic `response_format` and validate the result.

    Pydantic validation doubles as the retry hook: providers vary in how strictly
    they honor the schema, so an unparseable response is retried before failing.
    """
    litellm = _litellm()
    kwargs = _completion_kwargs(model, messages, retries, fallback, params)
    kwargs["response_format"] = schema

    last_error: Exception | None = None
    for _ in range(retries + 1):
        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content or ""
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            last_error = exc
    raise LLMError(
        f"model '{model}' did not return output matching schema "
        f"'{schema.__name__}' after {retries + 1} attempt(s)"
    ) from last_error


@cache
def _list_container(item_type: type) -> type[BaseModel]:
    return create_model("LLMListOutput", items=(list[item_type], ...))  # type: ignore[valid-type]


@cache
def _classification_model(categories: tuple[str, ...]) -> type[BaseModel]:
    return create_model("LLMClassification", category=(Literal[categories], ...))


@cache
def _score_model() -> type[BaseModel]:
    return create_model("LLMScore", score=(float, ...))


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
    container = _list_container(item_type)
    result: Any = complete_structured(
        model, messages, container, retries, fallback, params
    )
    return result.items


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
        "num_retries": retries,
        **params,
    }
    if (fallbacks := _fallbacks(fallback)) is not None:
        kwargs["fallbacks"] = fallbacks
    response = litellm.embedding(**kwargs)
    return list(response.data[0]["embedding"])
