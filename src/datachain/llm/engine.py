from functools import cache
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, create_model

from datachain.lib.utils import DataChainError
from datachain.llm.types import Usage

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T", bound=BaseModel)


class LLMError(DataChainError):
    """Raised when a `datachain.llm` operation cannot produce a valid result."""


RESERVED_PARAMS = frozenset(
    {"model", "messages", "input", "num_retries", "fallbacks", "response_format"}
)


def _litellm():
    # Imported lazily (it is slow to import) so `import datachain` stays fast.
    import litellm

    return litellm


def _fallbacks(fallback: str | list[str] | None) -> list[str] | None:
    if not fallback:  # None, "", or []
        return None
    return [fallback] if isinstance(fallback, str) else list(fallback)


def _is_transient(exc: Exception) -> bool:
    """Worth retrying: timeout (408), rate limit (429), and server errors. Client
    errors (bad auth/request) and errors with no HTTP status (e.g. a ValueError
    from a bug) are fatal and not retried."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        return False
    return status in (408, 429) or status >= 500


def _base_kwargs(
    model: str,
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    # `params` first so datachain.llm's own keys always win.
    kwargs: dict[str, Any] = {**params, "model": model, "num_retries": max(retries, 0)}
    if (fallbacks := _fallbacks(fallback)) is not None:
        kwargs["fallbacks"] = fallbacks
    return kwargs


def _completion_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    kwargs = _base_kwargs(model, retries, fallback, params)
    kwargs["messages"] = messages
    return kwargs


def _content(response: Any) -> str:
    if not response.choices:
        raise LLMError("model returned no choices")
    message = getattr(response.choices[0], "message", None)
    if message is None:
        raise LLMError("model returned no message")
    return message.content or ""


def _finish_reason(response: Any) -> str:
    if not response.choices:
        return ""
    return getattr(response.choices[0], "finish_reason", "") or ""


def _tokens(response: Any) -> tuple[int, int]:
    u = getattr(response, "usage", None)
    return (
        getattr(u, "prompt_tokens", 0) or 0,
        getattr(u, "completion_tokens", 0) or 0,
    )


def _usage(response: Any) -> Usage:
    input_tokens, output_tokens = _tokens(response)
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, retries=0)


def _content_or_empty(response: Any) -> str:
    try:
        return _content(response)
    except LLMError:
        return ""


def _reask_prompt(name: str, error: Exception) -> str:
    detail = str(error.__cause__ or error)[:500]
    return (
        f"Your previous response could not be parsed as {name}: {detail}. "
        "Return only valid JSON matching the schema, with no extra text."
    )


def _strip_fences(text: str) -> str:
    """Best-effort unwrap of a ```...``` markdown code fence around JSON."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = t[3:]
    newline = t.find("\n")
    if newline != -1 and t[:newline].strip().isidentifier():  # drop a ```json tag
        t = t[newline + 1 :]
    if t.rstrip().endswith("```"):
        t = t.rstrip()[:-3]
    return t.strip()


def _truncated_error(name: str) -> LLMError:
    return LLMError(
        f"model output for {name} was truncated "
        "(finish_reason=length); increase max_tokens"
    )


def parse_one(schema: type[T], content: str) -> T:
    last_error: ValidationError | None = None
    for candidate in (content, _strip_fences(content)):
        try:
            return schema.model_validate_json(candidate)
        except ValidationError as exc:
            last_error = exc
    raise LLMError(
        f"model output could not be parsed as '{schema.__name__}'"
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
        f"model output could not be parsed as list[{item_type.__name__}]"
    ) from last_error


def complete_text(
    model: str,
    messages: list[dict[str, Any]],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> tuple[str, Usage]:
    litellm = _litellm()
    kwargs = _completion_kwargs(model, messages, retries, fallback, params)
    resp = litellm.completion(**kwargs)
    return _content(resp), _usage(resp)


def _parse_with_retries(
    kwargs: dict[str, Any],
    retries: int,
    parse: "Callable[[str], Any]",
    name: str,
) -> tuple[Any, Usage]:
    # This loop owns the retry budget (num_retries=0) so it does not multiply with
    # LiteLLM's. A schema mismatch reasks (feeds the failed output back); a `length`
    # finish aborts; tokens accumulate across attempts.
    litellm = _litellm()
    messages: list[dict[str, Any]] = list(kwargs["messages"])
    kwargs = {**kwargs, "messages": messages, "num_retries": 0}
    attempts = max(retries, 0) + 1
    total_in = total_out = 0
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = litellm.completion(**kwargs)
        except Exception as exc:
            if not _is_transient(exc):
                raise
            last_error = exc
            continue
        tokens_in, tokens_out = _tokens(resp)
        total_in += tokens_in
        total_out += tokens_out
        try:
            value = parse(_content(resp))
            usage = Usage(
                input_tokens=total_in, output_tokens=total_out, retries=attempt
            )
            return value, usage
        except LLMError as exc:
            if _finish_reason(resp) == "length":
                raise _truncated_error(name) from exc
            last_error = exc
            if bad := _content_or_empty(resp):
                messages.append({"role": "assistant", "content": bad})
            messages.append({"role": "user", "content": _reask_prompt(name, exc)})
    if isinstance(last_error, LLMError):
        raise LLMError(
            f"model output did not match {name} after {attempts} attempt(s)"
        ) from last_error
    assert last_error is not None  # attempts >= 1, so a failure was recorded
    raise last_error


def complete_structured(
    model: str,
    messages: list[dict[str, Any]],
    schema: type[T],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> tuple[T, Usage]:
    kwargs = _completion_kwargs(model, messages, retries, fallback, params)
    kwargs["response_format"] = schema
    return _parse_with_retries(
        kwargs, retries, lambda c: parse_one(schema, c), f"schema '{schema.__name__}'"
    )


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
) -> tuple[str, Usage]:
    schema = _classification_model(categories)
    result, usage = complete_structured(
        model, messages, schema, retries, fallback, params
    )
    return result.category, usage  # type: ignore[attr-defined]


def score(
    model: str,
    messages: list[dict[str, Any]],
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> tuple[float, Usage]:
    schema = _score_model()
    result, usage = complete_structured(
        model, messages, schema, retries, fallback, params
    )
    return result.score, usage  # type: ignore[attr-defined]


def complete_structured_list(
    model: str,
    messages: list[dict[str, Any]],
    item_type: type,
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> tuple[list, Usage]:
    kwargs = _completion_kwargs(model, messages, retries, fallback, params)
    kwargs["response_format"] = _list_container(item_type)
    return _parse_with_retries(
        kwargs,
        retries,
        lambda c: parse_list(item_type, c),
        f"list[{item_type.__name__}]",
    )


def embed(
    model: str,
    text: str,
    retries: int,
    fallback: str | list[str] | None,
    params: dict[str, Any],
) -> tuple[list[float], Usage]:
    litellm = _litellm()
    kwargs = _base_kwargs(model, retries, fallback, params)
    kwargs["input"] = [text]
    resp = litellm.embedding(**kwargs)
    data = resp.data
    if not data:
        raise LLMError("embedding response contained no data")
    item = data[0]
    vector = list(item["embedding"] if isinstance(item, dict) else item.embedding)
    return vector, _usage(resp)
