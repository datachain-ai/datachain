import inspect
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, get_args, get_origin

from datachain.llm import engine
from datachain.llm.content import build_messages, serialize_value

if TYPE_CHECKING:
    from datachain.lib.settings import Settings

MODEL_ENV_VAR = "DATACHAIN_AI_MODEL"


class LLMConfigError(engine.LLMError):
    """Raised when no model can be resolved for a `datachain.llm` operation."""


def _element_type(schema: Any) -> tuple[Any, bool]:
    """Return ``(type, is_list)`` splitting ``list[X]`` into element type ``X``."""
    if get_origin(schema) is list:
        args = get_args(schema)
        return (args[0] if args else str), True
    return schema, False


@dataclass
class LLMSpec:
    """A configured `datachain.llm` operation, used inside `.map()` / `.gen()`.

    Returned by `complete`, `classify`, `score`, and `embed`; not constructed
    directly. The chain binds it to the active settings when the verb runs.
    """

    kind: str  # "complete" | "classify" | "score" | "embed"
    col: str
    prompt: str | None = None
    schema: Any = None
    into: list[str] | None = None
    context_col: str | None = None
    llm: str | None = None
    retries: int = 1
    fallback: str | list[str] | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def output_type(self) -> Any:
        if self.kind == "embed":
            return list[float]
        if self.kind == "score":
            return float
        if self.kind == "classify":
            return str
        # complete
        if self.schema is None:
            return str
        return _element_type(self.schema)[0]

    def return_annotation(self) -> Any:
        """Annotation seen by the verb: ``Iterator[T]`` for the 1:N (list) case."""
        if self.kind == "complete" and self.schema is not None:
            elem, is_list = _element_type(self.schema)
            if is_list:
                return Iterator[elem]  # type: ignore[valid-type]
        return self.output_type()

    def identity(self, model: str) -> tuple:
        """Stable identity baked into the UDF hash so the cache invalidates only
        when the model, prompt, or schema actually changes."""
        schema_repr = None
        if self.schema is not None and hasattr(self.schema, "model_json_schema"):
            schema_repr = str(self.schema.model_json_schema())
        elif self.schema is not None:
            schema_repr = str(self.schema)
        return (
            self.kind,
            model,
            self.prompt,
            schema_repr,
            tuple(self.into) if self.into else None,
            self.context_col,
            self.retries,
            tuple(self.fallback) if isinstance(self.fallback, list) else self.fallback,
            tuple(sorted(self.params)),
        )

    def _resolve_model(self, settings: "Settings") -> str:
        model = self.llm or settings.llm or os.environ.get(MODEL_ENV_VAR)
        if not model:
            raise LLMConfigError(
                f"no model configured for llm.{self.kind}(). Set one with "
                '.settings(llm="provider/model"), a per-call llm=, '
                f"or the {MODEL_ENV_VAR} environment variable."
            )
        return model

    def _build_prompt(self) -> str | None:
        if self.kind == "classify":
            cats = ", ".join(self.into or [])
            base = f"Classify the input into exactly one of: {cats}."
            return f"{base}\n\n{self.prompt}" if self.prompt else base
        if self.kind == "score":
            base = self.prompt or "Score the input."
            return f"{base}\nReturn a single numeric score."
        return self.prompt

    def _run(
        self,
        model: str,
        llm_params: Any,
        value: Any,
        context: Any,
    ) -> Any:
        params = dict(llm_params() if callable(llm_params) else (llm_params or {}))
        params.update(self.params)

        if self.kind == "embed":
            return engine.embed(
                model, serialize_value(value), self.retries, self.fallback, params
            )

        messages = build_messages(self._build_prompt(), value, context)

        if self.kind == "classify":
            return engine.classify(
                model,
                messages,
                tuple(self.into or []),
                self.retries,
                self.fallback,
                params,
            )
        if self.kind == "score":
            return engine.score(model, messages, self.retries, self.fallback, params)
        # complete
        if self.schema is None:
            return engine.complete_text(
                model, messages, self.retries, self.fallback, params
            )
        elem, is_list = _element_type(self.schema)
        if is_list:
            return engine.complete_structured_list(
                model, messages, elem, self.retries, self.fallback, params
            )
        return engine.complete_structured(
            model, messages, self.schema, self.retries, self.fallback, params
        )

    def __datachain_bind__(self, settings: "Settings") -> Callable:
        model = self._resolve_model(settings)
        llm_params = settings.llm_params
        identity = self.identity(model)
        spec = self

        param = inspect.Parameter(self.col, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        _call: Any
        if self.context_col:

            def _call_with_context(value, context, _identity=identity):
                return spec._run(model, llm_params, value, context)

            _call = _call_with_context
            parameters = [
                param,
                inspect.Parameter(
                    self.context_col, inspect.Parameter.POSITIONAL_OR_KEYWORD
                ),
            ]
        else:

            def _call_value(value, _identity=identity):
                return spec._run(model, llm_params, value, None)

            _call = _call_value
            parameters = [param]

        _call.__signature__ = inspect.Signature(
            parameters, return_annotation=self.return_annotation()
        )
        _call.__name__ = f"llm_{self.kind}"
        _call.__qualname__ = _call.__name__
        return _call
