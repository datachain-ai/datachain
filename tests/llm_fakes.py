import json
import types
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel


def _fill(annotation: Any) -> Any:  # noqa: PLR0911
    if get_origin(annotation) is Literal:
        return get_args(annotation)[0]
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        return [_fill(args[0])] if args else []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {n: _fill(f.annotation) for n, f in annotation.model_fields.items()}
    if annotation is float:
        return 0.5
    if annotation is int:
        return 1
    if annotation is bool:
        return True
    return "x"


def _structured_json(schema: type[BaseModel]) -> str:
    return json.dumps(
        {name: _fill(f.annotation) for name, f in schema.model_fields.items()}
    )


def _response(content: str, finish_reason: str = "stop"):
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=7)
    return types.SimpleNamespace(choices=[choice], usage=usage)


class FakeLiteLLM:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.embedding_calls: list[dict[str, Any]] = []
        self.text_response = "hello"
        self.embedding_response = [0.1, 0.2, 0.3]
        self.invalid_json_attempts = 0
        self.structured_overrides: dict[str, str] = {}
        self.embedding_empty = False
        self.embedding_as_object = False
        self.finish_reason = "stop"
        self.fail_models: set[str] = set()
        self.fatal_status: int | None = None

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.fatal_status is not None:
            exc = RuntimeError(f"http {self.fatal_status}")
            exc.status_code = self.fatal_status  # type: ignore[attr-defined]
            raise exc
        # LiteLLM transparently routes to a fallback; with none, the call errors.
        if kwargs.get("model") in self.fail_models and not kwargs.get("fallbacks"):
            raise RuntimeError(f"model {kwargs.get('model')} failed")
        fr = self.finish_reason
        schema = kwargs.get("response_format")
        if schema is None:
            return _response(self.text_response, fr)
        if self.invalid_json_attempts > 0:
            self.invalid_json_attempts -= 1
            return _response("not json", fr)
        if schema.__name__ in self.structured_overrides:
            return _response(self.structured_overrides[schema.__name__], fr)
        return _response(_structured_json(schema), fr)

    def embedding(self, **kwargs):
        self.embedding_calls.append(kwargs)
        if self.embedding_empty:
            return types.SimpleNamespace(data=[])
        vector = list(self.embedding_response)
        usage = types.SimpleNamespace(prompt_tokens=5)
        item = (
            types.SimpleNamespace(embedding=vector)
            if self.embedding_as_object
            else {"embedding": vector}
        )
        return types.SimpleNamespace(data=[item], usage=usage)
