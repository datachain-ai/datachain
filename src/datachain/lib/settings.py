from typing import Any

from datachain.lib.utils import DataChainParamsError

DEFAULT_CACHE = False
DEFAULT_PREFETCH = 2


class SettingsError(DataChainParamsError):
    def __init__(self, msg: str) -> None:
        super().__init__(f"Dataset settings error: {msg}")


def _validate_bool(name: str, value: bool | None) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise SettingsError(
            f"'{name}' argument must be bool"
            f" while {value.__class__.__name__} was given"
        )
    return value


def _validate_str(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SettingsError(
            f"'{name}' argument must be str"
            f", {value.__class__.__name__} was given"
        )
    return value


def _validate_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SettingsError(
            f"'{name}' argument must be int"
            f", {value.__class__.__name__} was given"
        )
    if value <= 0:
        raise SettingsError(
            f"'{name}' argument must be positive integer"
            f", {value} was given"
        )
    return value


def _validate_prefetch(value: bool | int | None) -> int | None:
    if value is None or value is True:
        return None
    if value is False:
        return 0
    if not isinstance(value, int):
        raise SettingsError(
            "'prefetch' argument must be int or bool"
            f" while {value.__class__.__name__} was given"
        )
    if value < 0:
        raise SettingsError(
            "'prefetch' argument must be non-negative integer"
            f", {value} was given"
        )
    return value


def _validate_parallel(value: bool | int | None) -> bool | int | None:
    if value is None or value is False:
        return None
    if value is True:
        return True
    if not isinstance(value, int):
        raise SettingsError(
            "'parallel' argument must be int or bool"
            f" while {value.__class__.__name__} was given"
        )
    if value <= 0:
        raise SettingsError(
            "'parallel' argument must be positive integer"
            f", {value} was given"
        )
    return value


FIELDS = [
    "cache",
    "prefetch",
    "parallel",
    "workers",
    "namespace",
    "project",
    "min_task_size",
    "batch_size",
    "ephemeral",
]


class Settings:
    """Settings for datachain."""

    _cache: bool | None
    _prefetch: int | None
    _parallel: bool | int | None
    _workers: int | None
    _namespace: str | None
    _project: str | None
    _min_task_size: int | None
    _batch_size: int | None
    _ephemeral: bool | None

    VALIDATORS: tuple[tuple[str, str | None, Any], ...] = (
        ("_cache", "cache", _validate_bool),
        ("_prefetch", None, _validate_prefetch),
        ("_parallel", None, _validate_parallel),
        ("_workers", "workers", _validate_positive_int),
        ("_namespace", "namespace", _validate_str),
        ("_project", "project", _validate_str),
        ("_min_task_size", "min_task_size", _validate_positive_int),
        ("_batch_size", "batch_size", _validate_positive_int),
        ("_ephemeral", "ephemeral", _validate_bool),
    )

    def __init__(
        self,
        cache: bool | None = None,
        prefetch: bool | int | None = None,
        parallel: bool | int | None = None,
        workers: int | None = None,
        namespace: str | None = None,
        project: str | None = None,
        min_task_size: int | None = None,
        batch_size: int | None = None,
        ephemeral: bool | None = None,
    ) -> None:
        values = {
            "_cache": cache,
            "_prefetch": prefetch,
            "_parallel": parallel,
            "_workers": workers,
            "_namespace": namespace,
            "_project": project,
            "_min_task_size": min_task_size,
            "_batch_size": batch_size,
            "_ephemeral": ephemeral,
        }
        for attr, name, validator in self.VALIDATORS:
            val = values[attr]
            if name is None:
                setattr(self, attr, validator(val))
            else:
                setattr(self, attr, validator(name, val))

    @property
    def cache(self) -> bool:
        return self._cache if self._cache is not None else DEFAULT_CACHE

    @property
    def prefetch(self) -> int | None:
        return self._prefetch if self._prefetch is not None else DEFAULT_PREFETCH

    @property
    def parallel(self) -> bool | int | None:
        return self._parallel if self._parallel is not None else None

    @property
    def workers(self) -> int | None:
        return self._workers if self._workers is not None else None

    @property
    def namespace(self) -> str | None:
        return self._namespace if self._namespace is not None else None

    @property
    def project(self) -> str | None:
        return self._project if self._project is not None else None

    @property
    def min_task_size(self) -> int | None:
        return self._min_task_size if self._min_task_size is not None else None

    @property
    def batch_size(self) -> int | None:
        return self._batch_size if self._batch_size is not None else None

    @property
    def ephemeral(self) -> bool:
        return self._ephemeral if self._ephemeral is not None else False

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in FIELDS if getattr(self, f"_{f}") is not None}

    def add(self, settings: "Settings") -> None:
        for f in FIELDS:
            val = getattr(settings, f"_{f}")
            if val is not None:
                setattr(self, f"_{f}", val)
