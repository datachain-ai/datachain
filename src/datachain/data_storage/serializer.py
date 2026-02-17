import base64
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from datachain import json
from datachain.plugins import ensure_plugins_loaded


class CallableRegistry:
    _registry: ClassVar[dict[str, Callable]] = {}

    @classmethod
    def register(cls, callable_obj: Callable, name: str) -> str:
        cls._registry[name] = callable_obj
        return name

    @classmethod
    def get(cls, name: str) -> Callable:
        return cls._registry[name]


class Serializable:
    @classmethod
    @abstractmethod
    def serialize_callable_name(cls) -> str:
        """Return the registered name used for this class' factory callable."""

    @abstractmethod
    def clone_params(self) -> tuple[Callable[..., Any], list[Any], dict[str, Any]]:
        """Return (callable, args, kwargs) necessary to recreate this object."""

    def _prepare(self, params: tuple) -> dict:
        callable, args, kwargs = params
        callable_name = callable.__self__.serialize_callable_name()
        return {
            "callable": callable_name,
            "args": args,
            "kwargs": {
                k: self._prepare(v) if isinstance(v, tuple) else v
                for k, v in kwargs.items()
            },
        }

    def serialize(self) -> str:
        """Return a base64-encoded JSON string with registered callable + params."""
        _ensure_default_callables_registered()
        data = self.clone_params()
        return base64.b64encode(json.dumps(self._prepare(data)).encode()).decode()


def deserialize(s: str) -> Serializable:
    """Deserialize from base64-encoded JSON using only registered callables.

    Nested serialized objects are instantiated automatically except for those
    passed via clone parameter tuples (keys ending with ``_clone_params``),
    which must remain as (callable, args, kwargs) for later factory usage.
    """
    ensure_plugins_loaded()
    _ensure_default_callables_registered()
    decoded = base64.b64decode(s.encode())
    data = json.loads(decoded.decode())

    def _is_serialized(obj: Any) -> bool:
        return isinstance(obj, dict) and {"callable", "args", "kwargs"}.issubset(
            obj.keys()
        )

    def _reconstruct(obj: Any, nested: bool = False) -> Any:
        if not _is_serialized(obj):
            return obj
        callable_name: str = obj["callable"]
        args: list[Any] = obj["args"]
        kwargs: dict[str, Any] = obj["kwargs"]
        # Recurse only inside kwargs because serialize() only nests through kwargs
        for k, v in list(kwargs.items()):
            if _is_serialized(v):
                kwargs[k] = _reconstruct(v, True)
        callable_obj = CallableRegistry.get(callable_name)
        if nested:
            return (callable_obj, args, kwargs)
        # Otherwise instantiate
        return callable_obj(*args, **kwargs)

    if not _is_serialized(data):
        raise ValueError("Invalid serialized data format")
    return _reconstruct(data, False)


class _DefaultsState:
    registered = False


def _ensure_default_callables_registered() -> None:
    if _DefaultsState.registered:
        return

    from datachain.data_storage.sqlite import (
        SQLiteDatabaseEngine,
        SQLiteMetastore,
        SQLiteWarehouse,
    )

    # Register (idempotent by name overwrite is fine) using class-level
    # serialization names to avoid hard-coded literals here.
    CallableRegistry.register(
        SQLiteDatabaseEngine.from_db_file,
        SQLiteDatabaseEngine.serialize_callable_name(),
    )
    CallableRegistry.register(
        SQLiteMetastore.init_after_clone,
        SQLiteMetastore.serialize_callable_name(),
    )
    CallableRegistry.register(
        SQLiteWarehouse.init_after_clone,
        SQLiteWarehouse.serialize_callable_name(),
    )

    _DefaultsState.registered = True
