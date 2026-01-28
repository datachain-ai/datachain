from typing import Any

from datachain.lib.utils import DataChainParamsError

DEFAULT_CACHE = False
DEFAULT_PREFETCH = 2


class SettingsError(DataChainParamsError):
    def __init__(self, msg: str) -> None:
        super().__init__(f"Dataset settings error: {msg}")


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

    def __init__(  # noqa: C901, PLR0912
        self,
        cache: bool | None = None,
        prefetch: bool | int | None = None,
        parallel: bool | int | None = None,
        workers: int | None = None,
        namespace: str | None = None,
        project: str | None = None,
        min_task_size: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        if cache is None:
            self._cache = None
        else:
            if not isinstance(cache, bool):
                raise SettingsError(
                    "'cache' argument must be bool"
                    f" while {cache.__class__.__name__} was given"
                )
            self._cache = cache

        if prefetch is None or prefetch is True:
            self._prefetch = None
        elif prefetch is False:
            self._prefetch = 0  # disable prefetch (False == 0)
        else:
            if not isinstance(prefetch, int):
                raise SettingsError(
                    "'prefetch' argument must be int or bool"
                    f" while {prefetch.__class__.__name__} was given"
                )
            if prefetch < 0:
                raise SettingsError(
                    "'prefetch' argument must be non-negative integer"
                    f", {prefetch} was given"
                )
            self._prefetch = prefetch

        if parallel is None or parallel is False:
            self._parallel = None
        elif parallel is True:
            self._parallel = True
        else:
            if not isinstance(parallel, int):
                raise SettingsError(
                    "'parallel' argument must be int or bool"
                    f" while {parallel.__class__.__name__} was given"
                )
            if parallel <= 0:
                raise SettingsError(
                    "'parallel' argument must be positive integer"
                    f", {parallel} was given"
                )
            self._parallel = parallel

        if workers is None:
            self._workers = None
        else:
            if not isinstance(workers, int) or isinstance(workers, bool):
                raise SettingsError(
                    "'workers' argument must be int"
                    f" while {workers.__class__.__name__} was given"
                )
            if workers <= 0:
                raise SettingsError(
                    f"'workers' argument must be positive integer, {workers} was given"
                )
            self._workers = workers

        if namespace is None:
            self._namespace = None
        else:
            if not isinstance(namespace, str):
                raise SettingsError(
                    "'namespace' argument must be str"
                    f", {namespace.__class__.__name__} was given"
                )
            self._namespace = namespace

        if project is None:
            self._project = None
        else:
            if not isinstance(project, str):
                raise SettingsError(
                    "'project' argument must be str"
                    f", {project.__class__.__name__} was given"
                )
            self._project = project

        if min_task_size is None:
            self._min_task_size = None
        else:
            if not isinstance(min_task_size, int) or isinstance(min_task_size, bool):
                raise SettingsError(
                    "'min_task_size' argument must be int"
                    f", {min_task_size.__class__.__name__} was given"
                )
            if min_task_size <= 0:
                raise SettingsError(
                    "'min_task_size' argument must be positive integer"
                    f", {min_task_size} was given"
                )
            self._min_task_size = min_task_size

        if batch_size is None:
            self._batch_size = None
        else:
            if not isinstance(batch_size, int) or isinstance(batch_size, bool):
                raise SettingsError(
                    "'batch_size' argument must be int"
                    f", {batch_size.__class__.__name__} was given"
                )
            if batch_size <= 0:
                raise SettingsError(
                    "'batch_size' argument must be positive integer"
                    f", {batch_size} was given"
                )
            self._batch_size = batch_size

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

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {}
        if self._cache is not None:
            res["cache"] = self.cache
        if self._prefetch is not None:
            res["prefetch"] = self.prefetch
        if self._parallel is not None:
            res["parallel"] = self.parallel
        if self._workers is not None:
            res["workers"] = self.workers
        if self._min_task_size is not None:
            res["min_task_size"] = self.min_task_size
        if self._namespace is not None:
            res["namespace"] = self.namespace
        if self._project is not None:
            res["project"] = self.project
        if self._batch_size is not None:
            res["batch_size"] = self.batch_size
        return res

    def add(self, settings: "Settings") -> None:
        if settings._cache is not None:
            self._cache = settings._cache
        if settings._prefetch is not None:
            self._prefetch = settings._prefetch
        if settings._parallel is not None:
            self._parallel = settings._parallel
        if settings._workers is not None:
            self._workers = settings._workers
        if settings._namespace is not None:
            self._namespace = settings._namespace
        if settings._project is not None:
            self._project = settings._project
        if settings._min_task_size is not None:
            self._min_task_size = settings._min_task_size
        if settings._batch_size is not None:
            self._batch_size = settings._batch_size
