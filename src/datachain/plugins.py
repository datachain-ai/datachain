"""Plugin loader for DataChain callables.

Discovers and invokes entry points in the group "datachain.callables" once
per process. This enables external packages (e.g., Studio) to register
their callables with the serializer registry without explicit imports.
"""

from importlib import metadata as importlib_metadata

_plugins_loaded = False


def ensure_plugins_loaded() -> None:
    global _plugins_loaded  # noqa: PLW0603
    if _plugins_loaded:
        return

    # Compatible across importlib.metadata versions
    eps_obj = importlib_metadata.entry_points()
    for ep in eps_obj.select(group="datachain.callables"):
        func = ep.load()
        func()

    _plugins_loaded = True
