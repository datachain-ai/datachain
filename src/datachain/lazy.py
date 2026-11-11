from collections.abc import Callable
from importlib import import_module
from typing import Any


def lazy_exports(
    package: str, namespace: dict[str, Any], targets: dict[str, str]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Return a package's `__getattr__` and `__dir__` (PEP 562) that import each
    name in `targets` on first access and cache it in `namespace`.

    A target is "module:attribute", or "module" when the name is the module itself.
    """

    def getattr_(name: str) -> Any:
        if (target := targets.get(name)) is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module_name, _, attr = target.partition(":")
        try:
            module = import_module(module_name)
            value = getattr(module, attr) if attr else module
        except AttributeError as exc:
            # `from package import name` turns an AttributeError into a bare
            # "cannot import name" and drops the cause.
            raise ImportError(f"cannot import {name!r} from {target!r}: {exc}") from exc
        namespace[name] = value
        return value

    def dir_() -> list[str]:
        return sorted({*namespace, *targets})

    return getattr_, dir_
