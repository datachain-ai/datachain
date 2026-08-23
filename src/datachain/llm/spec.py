import re
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from pydantic import BaseModel

from datachain.lib.udf import BindContext, BoundSpec
from datachain.llm import engine
from datachain.llm.content import MEDIA_VALUES, Media, build_messages, to_text
from datachain.llm.types import Usage

if TYPE_CHECKING:
    from datachain.lib.settings import Settings


class LLMConfigError(engine.LLMError):
    """Raised when no model can be resolved for a `datachain.llm` operation."""


def _element_type(schema: Any) -> tuple[Any, bool]:
    """Return ``(type, is_list)`` splitting ``list[X]`` into element type ``X``."""
    if get_origin(schema) is list:
        args = get_args(schema)
        return (args[0] if args else str), True
    return schema, False


# A value with no custom repr shows its memory address (``<Cls object at 0x...>``),
# which changes each run and would make the cache key unstable.
_DEFAULT_REPR = re.compile(r" object at 0x[0-9a-fA-F]+>")


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
    if _DEFAULT_REPR.search(repr(value)):
        warnings.warn(
            f"llm param {type(value).__name__!r} has no stable repr; it breaks "
            "caching (full recompute every run).",
            stacklevel=2,
        )
    return value


def _is_secret_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    k = key.lower()
    hints = ("key", "secret", "password", "authorization", "credential")
    # `_token` as a suffix so `max_tokens` is not treated as a secret
    return any(s in k for s in hints) or k == "token" or k.endswith("_token")


def _without_secrets(value: Any) -> Any:
    """Recursively drop credential-like keys so secrets never enter the cache key."""
    if isinstance(value, dict):
        return {
            k: _without_secrets(v) for k, v in value.items() if not _is_secret_key(k)
        }
    if isinstance(value, (list, tuple)):
        return [_without_secrets(v) for v in value]
    return value


@dataclass
class LLMSpec(BoundSpec):
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
    type: Media | None = None
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
        if self.type is not None and self.type not in MEDIA_VALUES:
            raise ValueError(
                f"type must be 'text', 'image', or 'document', got {self.type!r}"
            )
        if not isinstance(self.retries, int) or isinstance(self.retries, bool):
            raise ValueError(  # noqa: TRY004 - a config value error, not a type guard
                f"retries must be an int, got {self.retries!r}"
            )
        if self.retries < 0:
            raise ValueError(f"retries must be >= 0, got {self.retries}")
        if self.fallback is not None and not (
            isinstance(self.fallback, str)
            or (
                isinstance(self.fallback, list)
                and all(isinstance(m, str) for m in self.fallback)
            )
        ):
            raise ValueError(
                "fallback must be a model string or a list of model strings, "
                f"got {self.fallback!r}"
            )
        if self.context_col is not None and self.context_col == self.col:
            raise ValueError("col and context must be different columns")
        reserved = engine.RESERVED_PARAMS & set(self.params)
        if reserved:
            raise ValueError(
                "these are managed by datachain.llm and cannot be passed as params: "
                f"{sorted(reserved)}"
            )

    def _can_fan_out(self) -> bool:
        """`complete(schema=list[...])` can produce many items from one input."""
        return self.kind == "complete" and _element_type(self.schema)[1]

    def output_type(self) -> Any:
        """The value the call yields for one input (``list[Item]`` for a list
        schema). The verb decides shape: ``.map()`` stores it, ``.gen()`` fans it
        out into the element type."""
        if self.kind == "embed":
            return list[float]
        if self.kind == "score":
            return float
        if self.kind == "classify":
            return str
        return self.schema if self.schema is not None else str

    def return_annotation(self, to_many: bool = False) -> Any:
        """Annotation seen by the verb. ``.gen()`` (``to_many``) fans a list schema
        into ``Iterator[Item]``; ``.map()`` keeps the whole value (``list[Item]`` or
        a scalar), optional since a ``None`` input yields ``None``.
        ``include_usage`` pairs each value with a ``Usage``."""
        if to_many and self._can_fan_out():
            elem = _element_type(self.schema)[0]
            item = tuple[elem, Usage] if self.include_usage else elem  # type: ignore[valid-type]
            return Iterator[item]  # type: ignore[valid-type]
        out = self.output_type() | None
        return tuple[out, Usage] if self.include_usage else out  # type: ignore[valid-type]

    def identity(self, model: str, llm_params: Any = None) -> tuple:
        """Cache key baked into the UDF hash; changes iff an output-affecting
        input (model, prompt, schema, params, llm_params, ...) changes.

        The schema is keyed by its JSON schema (fields, types, constraints, name);
        a pure validator/serializer-logic edit with unchanged fields will not
        invalidate the cache. A callable ``llm_params`` is resolved per worker at
        runtime (e.g. credentials) and is not part of the key; put output-affecting
        params in the dict form of ``llm_params`` or in per-call kwargs so they are
        captured here. Well-known credential keys are stripped so secrets never
        enter the key. ``retries`` and ``fallback`` are excluded (reliability
        policy, not part of the request) so changing them resumes from the
        checkpoint instead of re-running primary-answered rows.
        """
        schema_repr: Any = None
        if self.schema is not None:
            elem, is_list = _element_type(self.schema)
            if hasattr(elem, "model_json_schema"):
                schema_repr = (_canonical(elem.model_json_schema()), is_list)
            else:
                schema_repr = str(self.schema)
        params = self.params
        if isinstance(llm_params, dict):
            params = {**llm_params, **self.params}
        params = _without_secrets(params)
        return (
            self.kind,
            model,
            self.prompt,
            schema_repr,
            tuple(self.into) if self.into else None,
            self.col,
            self.context_col,
            self.type,
            self.include_usage,
            _canonical(params),
        )

    def _resolve_model(self, settings: "Settings") -> str:
        model = self.llm or settings.llm
        if not model:
            raise LLMConfigError(
                f"no model configured for llm.{self.kind}(); set one with "
                '.settings(llm="provider/model") or a per-call llm="provider/model".'
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
        to_many: bool,
    ) -> Any:
        if value is None:  # no input, no model call: propagate None (empty in .gen())
            if to_many:
                return []
            return (None, Usage()) if self.include_usage else None
        result, usage = self._call(model, params, value, context)
        if to_many:  # .gen(): fan the list out, one row per item
            if not self.include_usage:
                return result
            # per-call usage on the first row, zero on the rest, so a sum counts it once
            return [
                (item, usage if i == 0 else Usage()) for i, item in enumerate(result)
            ]
        return (result, usage) if self.include_usage else result

    def _call(
        self, model: str, params: dict[str, Any], value: Any, context: Any
    ) -> tuple[Any, Usage]:
        """Run the model call, returning ``(value, usage)``; ``value`` is the whole
        list for a list schema."""
        if self.kind == "embed":
            return engine.embed(
                model, to_text(value), self.retries, self.fallback, params
            )

        messages = build_messages(self._build_prompt(), value, self.type, context)
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

    def _validate_target(self, target: Any) -> None:
        """`.map()` always works (a list schema yields a list-valued column).
        `.gen()` needs a fan-out-able op (`complete(schema=list[...])`); `.agg()`
        is not supported."""
        if target is None:
            return
        out_batched = getattr(target, "is_output_batched", False)
        in_batched = getattr(target, "is_input_batched", False)
        if not out_batched:
            return  # .map(): 1:1
        if self._can_fan_out():
            if in_batched:  # aggregator
                raise LLMConfigError(
                    f"llm.{self.kind}() cannot aggregate; use .gen() (one row per "
                    "item) or .map() (one list-valued column)"
                )
        elif out_batched:
            raise LLMConfigError(
                f"llm.{self.kind}() yields one value per row; use .map()"
            )

    def _stamp(self, fn: Any, to_many: bool) -> Callable:
        # Output type is declared as a normal return annotation; inputs flow via
        # __datachain_params__ because column names may be dotted (e.g. "file.path"),
        # which can't be function parameters.
        fn.__annotations__["return"] = self.return_annotation(to_many)
        fn.__name__ = fn.__qualname__ = f"llm_{self.kind}"
        fn.__datachain_params__ = self.input_columns()
        return fn

    def bind(self, ctx: BindContext) -> Callable:
        self._validate_target(ctx.target)
        spec = self
        # .gen() fans a list schema into rows; .map() keeps the whole value.
        to_many = getattr(ctx.target, "is_output_batched", False)
        model = self._resolve_model(ctx.settings)
        llm_params = ctx.settings.llm_params
        resolved: list[dict[str, Any]] = []

        def params() -> dict[str, Any]:
            # Resolve credentials once per worker, then overlay per-call params.
            if not resolved:
                base = llm_params() if callable(llm_params) else llm_params
                if base is not None and not isinstance(base, dict):
                    raise TypeError(
                        f"llm_params must resolve to a dict, got {type(base).__name__}"
                    )
                resolved.append({**(base or {}), **spec.params})
            return resolved[0]

        # `_id` default arg bakes the cache key into the UDF hash (via __defaults__).
        _id = self.identity(model, llm_params)
        if self.context_col:

            def run_with_context(value, context, _id=_id):
                return spec._run(model, params(), value, context, to_many)

            return self._stamp(run_with_context, to_many)

        def run(value, _id=_id):
            return spec._run(model, params(), value, None, to_many)

        return self._stamp(run, to_many)

    def input_columns(self) -> list[str]:
        return [self.col, self.context_col] if self.context_col else [self.col]
