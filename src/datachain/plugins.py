"""Plugin loader for DataChain callables.

Discovers and invokes entry points in the group "datachain.callables" once
per process. This enables external packages (e.g., Studio) to register
their callables with the serializer registry without explicit imports.
"""

import logging
from importlib import metadata as importlib_metadata

logger = logging.getLogger("datachain")

_plugins_loaded = False


def ensure_plugins_loaded() -> None:
    global _plugins_loaded  # noqa: PLW0603
    if _plugins_loaded:
        return

    # Compatible across importlib.metadata versions
    eps_obj = importlib_metadata.entry_points()
    for ep in eps_obj.select(group="datachain.callables"):
        try:
            func = ep.load()
            func()
        except Exception as exc:  # noqa: BLE001
            distribution = getattr(getattr(ep, "dist", None), "name", "unknown")
            logger.warning(
                "Failed to initialize DataChain plugin %r from distribution %r: %s",
                ep.name,
                distribution,
                exc,
            )

    _plugins_loaded = True
