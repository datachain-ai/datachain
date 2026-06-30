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


def _response(content: str):
    message = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class FakeLiteLLM:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.embedding_calls: list[dict[str, Any]] = []
        self.text_response = "hello"
        self.embedding_response = [0.1, 0.2, 0.3]
        # Number of leading completion calls that return unparsable content.
        self.invalid_json_attempts = 0
        # Optional map of schema name -> JSON string overriding the auto-fill.
        self.structured_overrides: dict[str, str] = {}
        self.pdf_supported = True
        self.no_pdf_models: set[str] = set()
        self.embedding_empty = False

    def supports_pdf_input(self, model):
        return self.pdf_supported and model not in self.no_pdf_models

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs.get("response_format")
        if schema is None:
            return _response(self.text_response)
        if self.invalid_json_attempts > 0:
            self.invalid_json_attempts -= 1
            return _response("not json")
        if schema.__name__ in self.structured_overrides:
            return _response(self.structured_overrides[schema.__name__])
        return _response(_structured_json(schema))

    def embedding(self, **kwargs):
        self.embedding_calls.append(kwargs)
        if self.embedding_empty:
            return types.SimpleNamespace(data=[])
        vector = list(self.embedding_response)
        return types.SimpleNamespace(data=[{"embedding": vector}])
