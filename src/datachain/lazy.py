from importlib import import_module
from typing import Any


def lazy_exports(namespace: dict[str, Any], targets: dict[str, str]) -> None:
    """Install a package's `__getattr__` and `__dir__` (PEP 562) that import each
    name in `targets` on first access and cache it in `namespace`.

    A target is "module:attribute", or "module" when the name is the module itself.
    """

    package = namespace["__name__"]
    for name in targets:
        namespace.pop(name, None)

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

    namespace["__getattr__"] = getattr_
    namespace["__dir__"] = dir_
