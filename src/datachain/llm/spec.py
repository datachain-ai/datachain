import inspect
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from pydantic import BaseModel

from datachain.llm import engine
from datachain.llm.content import MEDIA_VALUES, Media, build_messages, to_text
from datachain.llm.types import Usage

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


def _canonical(value: Any) -> Any:
    """Order-independent form of a value, so the cache key is stable across
    processes (``repr`` of a ``set`` or unsorted ``dict`` is not)."""
    if isinstance(value, dict):
        items = ((k, _canonical(v)) for k, v in value.items())
        return tuple(sorted(items, key=lambda kv: repr(kv[0])))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_canonical(v) for v in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(v) for v in value)
    return value


@dataclass
class LLMSpec:
    """A configured `datachain.llm` operation, used inside `.map()` / `.gen()`.

    Returned by `complete`, `classify`, `score`, and `embed`; not constructed
    directly. The chain binds it to the active settings when the verb runs.
    """

    kind: Literal["complete", "classify", "score", "embed"]
    col: str
    prompt: str | None = None
    schema: Any = None
    into: list[str] | None = None
    context_col: str | None = None
    media: Media | None = None
    llm: str | None = None
    retries: int = 1
    fallback: str | list[str] | None = None
    include_usage: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema is not None:
            elem = _element_type(self.schema)[0]
            if not (isinstance(elem, type) and issubclass(elem, BaseModel)):
                raise TypeError(
                    "llm schema must be a pydantic model or list[model], "
                    f"got {self.schema!r}"
                )
        if self.into is not None:
            if not self.into or not all(isinstance(c, str) for c in self.into):
                raise ValueError(
                    "llm.classify(into=...) must be a non-empty list of strings"
                )
            if len(set(self.into)) != len(self.into):
                raise ValueError("llm.classify(into=...) categories must be distinct")
        if self.media is not None and self.media not in MEDIA_VALUES:
            raise ValueError(
                f"media must be 'text', 'image', or 'document', got {self.media!r}"
            )
        if not isinstance(self.retries, int) or isinstance(self.retries, bool):
            raise ValueError(  # noqa: TRY004 - a config value error, not a type guard
                f"retries must be an int, got {self.retries!r}"
            )
        if self.context_col is not None and self.context_col == self.col:
            raise ValueError("col and context must be different columns")
        reserved = engine.RESERVED_PARAMS & set(self.params)
        if reserved:
            raise ValueError(
                "these are managed by datachain.llm and cannot be passed as params: "
                f"{sorted(reserved)}"
            )

    def _is_one_to_many(self) -> bool:
        """A `complete(schema=list[...])` yields many rows per input (needs .gen)."""
        return self.kind == "complete" and _element_type(self.schema)[1]

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
        """Annotation seen by the verb. ``Iterator[T]`` for the 1:N (list) case;
        ``include_usage`` pairs every value with a ``Usage`` (``tuple[T, Usage]``)."""
        if self._is_one_to_many():
            elem = _element_type(self.schema)[0]
            item = tuple[elem, Usage] if self.include_usage else elem  # type: ignore[valid-type]
            return Iterator[item]  # type: ignore[valid-type]
        out = self.output_type()
        return tuple[out, Usage] if self.include_usage else out  # type: ignore[valid-type]

    def identity(self, model: str, llm_params: Any = None) -> tuple:
        """Cache key baked into the UDF hash; changes iff an output-affecting
        input (model, prompt, schema, params, llm_params, ...) changes."""
        schema_repr: Any = None
        if self.schema is not None:
            elem, is_list = _element_type(self.schema)
            # Hash the model's fields (not just its name) so editing a schema
            # while keeping its class name still invalidates the cache.
            if hasattr(elem, "model_json_schema"):
                schema_repr = (str(elem.model_json_schema()), is_list)
            else:
                schema_repr = str(self.schema)
        # Only the dict form of settings(llm_params=) is output-affecting; the
        # callable form resolves per-worker credentials and is left out.
        params = self.params
        if isinstance(llm_params, dict):
            params = {**llm_params, **self.params}
        return (
            self.kind,
            model,
            self.prompt,
            schema_repr,
            tuple(self.into) if self.into else None,
            self.context_col,
            self.media,
            self.retries,
            tuple(self.fallback) if isinstance(self.fallback, list) else self.fallback,
            self.include_usage,
            _canonical(params),
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
        params: dict[str, Any],
        value: Any,
        context: Any,
    ) -> Any:
        result, usage, is_list = self._call(model, params, value, context)
        if not self.include_usage:
            return result
        if is_list:
            return [(item, usage) for item in result]
        return (result, usage)

    def _call(
        self, model: str, params: dict[str, Any], value: Any, context: Any
    ) -> tuple[Any, Usage, bool]:
        """Run the model call, returning the tuple ``(value, usage, is_list)``."""
        result: Any
        if self.kind == "embed":
            result, usage = engine.embed(
                model, to_text(value), self.retries, self.fallback, params
            )
            return result, usage, False

        messages = build_messages(self._build_prompt(), value, self.media, context)
        if self.kind == "classify":
            result, usage = engine.classify(
                model,
                messages,
                tuple(self.into or []),
                self.retries,
                self.fallback,
                params,
            )
        elif self.kind == "score":
            result, usage = engine.score(
                model, messages, self.retries, self.fallback, params
            )
        elif self.schema is None:
            result, usage = engine.complete_text(
                model, messages, self.retries, self.fallback, params
            )
        else:
            elem, is_list = _element_type(self.schema)
            if is_list:
                items, usage = engine.complete_structured_list(
                    model, messages, elem, self.retries, self.fallback, params
                )
                return items, usage, True
            result, usage = engine.complete_structured(
                model, messages, self.schema, self.retries, self.fallback, params
            )
        return result, usage, False

    def _validate_target(self, target: Any) -> None:
        """Reject cardinality mismatches: 1:N `complete` needs `.gen()`, the rest
        produce one value per row and need `.map()`."""
        if target is None:
            return
        out_batched = getattr(target, "is_output_batched", False)
        in_batched = getattr(target, "is_input_batched", False)
        one_to_many = self._is_one_to_many()
        if one_to_many:
            if not out_batched or in_batched:
                raise LLMConfigError(
                    f"llm.{self.kind}(schema=list[...]) yields many rows per input; "
                    "use .gen()"
                )
        elif out_batched:
            raise LLMConfigError(
                f"llm.{self.kind}() yields one value per row; use .map()"
            )

    def _stamp(self, fn: Any, names: tuple[str, ...]) -> Callable:
        # Input columns travel through the explicit `params` channel (see
        # DataChain._udf_to_obj), so they may be nested/dotted; the signature only
        # carries the output type.
        fn.__signature__ = inspect.Signature(
            [
                inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for n in names
            ],
            return_annotation=self.return_annotation(),
        )
        fn.__name__ = fn.__qualname__ = f"llm_{self.kind}"
        fn.__datachain_params__ = self.input_columns()
        return fn

    def __datachain_bind__(self, settings: "Settings", target: Any = None) -> Callable:
        self._validate_target(target)
        spec = self
        model = self._resolve_model(settings)
        llm_params = settings.llm_params
        resolved: list[dict[str, Any]] = []

        def params() -> dict[str, Any]:
            # Resolve credentials once per worker (the closure is pickled fresh to
            # each worker), then overlay the per-call params.
            if not resolved:
                base = llm_params() if callable(llm_params) else dict(llm_params or {})
                resolved.append({**base, **spec.params})
            return resolved[0]

        # The `_id` default arg bakes the cache key into the closure, so the UDF
        # hash (which folds in __defaults__) changes when an input does.
        _id = self.identity(model, llm_params)
        if self.context_col:

            def run_with_context(value, context, _id=_id):
                return spec._run(model, params(), value, context)

            return self._stamp(run_with_context, ("value", "context"))

        def run(value, _id=_id):
            return spec._run(model, params(), value, None)

        return self._stamp(run, ("value",))

    def input_columns(self) -> list[str]:
        return [self.col, self.context_col] if self.context_col else [self.col]
