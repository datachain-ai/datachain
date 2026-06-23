from dataclasses import dataclass
from typing import Any

from datachain.lib.utils import DataChainParamsError

DEFAULT_CACHE = False
DEFAULT_PREFETCH = 2


class SettingsError(DataChainParamsError):
    def __init__(self, msg: str) -> None:
        super().__init__(f"Dataset settings error: {msg}")


@dataclass
class ExecutionConfig:
    cache: bool | None = None
    prefetch: int | None = None
    parallel: bool | int | None = None
    workers: int | None = None
    min_task_size: int | None = None
    batch_size: int | None = None


@dataclass
class ProjectConfig:
    namespace: str | None = None
    project: str | None = None


class Settings:
    """Settings for datachain."""

    def __init__(  # noqa: C901, PLR0912, PLR0915
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
        self._execution = ExecutionConfig()
        self._project = ProjectConfig()
        self._ephemeral: bool | None = None

        if cache is not None:
            if not isinstance(cache, bool):
                raise SettingsError(
                    "'cache' argument must be bool"
                    f" while {cache.__class__.__name__} was given"
                )
            self._execution.cache = cache

        if prefetch is None or prefetch is True:
            pass
        elif prefetch is False:
            self._execution.prefetch = 0
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
            self._execution.prefetch = prefetch

        if parallel is None or parallel is False:
            pass
        elif parallel is True:
            self._execution.parallel = True
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
            self._execution.parallel = parallel

        if workers is not None:
            if not isinstance(workers, int) or isinstance(workers, bool):
                raise SettingsError(
                    "'workers' argument must be int"
                    f" while {workers.__class__.__name__} was given"
                )
            if workers <= 0:
                raise SettingsError(
                    f"'workers' argument must be positive integer, {workers} was given"
                )
            self._execution.workers = workers

        if namespace is not None:
            if not isinstance(namespace, str):
                raise SettingsError(
                    "'namespace' argument must be str"
                    f", {namespace.__class__.__name__} was given"
                )
            self._project.namespace = namespace

        if project is not None:
            if not isinstance(project, str):
                raise SettingsError(
                    "'project' argument must be str"
                    f", {project.__class__.__name__} was given"
                )
            self._project.project = project

        if min_task_size is not None:
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
            self._execution.min_task_size = min_task_size

        if batch_size is not None:
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
            self._execution.batch_size = batch_size

        if ephemeral is not None:
            if not isinstance(ephemeral, bool):
                raise SettingsError(
                    "'ephemeral' argument must be bool"
                    f" while {ephemeral.__class__.__name__} was given"
                )
            self._ephemeral = ephemeral

    @property
    def cache(self) -> bool:
        val = self._execution.cache
        return val if val is not None else DEFAULT_CACHE

    @property
    def prefetch(self) -> int | None:
        val = self._execution.prefetch
        return val if val is not None else DEFAULT_PREFETCH

    @property
    def parallel(self) -> bool | int | None:
        val = self._execution.parallel
        return val if val is not None else None

    @property
    def workers(self) -> int | None:
        val = self._execution.workers
        return val if val is not None else None

    @property
    def namespace(self) -> str | None:
        val = self._project.namespace
        return val if val is not None else None

    @property
    def project(self) -> str | None:
        val = self._project.project
        return val if val is not None else None

    @property
    def min_task_size(self) -> int | None:
        val = self._execution.min_task_size
        return val if val is not None else None

    @property
    def batch_size(self) -> int | None:
        val = self._execution.batch_size
        return val if val is not None else None

    @property
    def ephemeral(self) -> bool:
        return self._ephemeral if self._ephemeral is not None else False

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {}
        if self._execution.cache is not None:
            res["cache"] = self.cache
        if self._execution.prefetch is not None:
            res["prefetch"] = self.prefetch
        if self._execution.parallel is not None:
            res["parallel"] = self.parallel
        if self._execution.workers is not None:
            res["workers"] = self.workers
        if self._execution.min_task_size is not None:
            res["min_task_size"] = self.min_task_size
        if self._project.namespace is not None:
            res["namespace"] = self.namespace
        if self._project.project is not None:
            res["project"] = self.project
        if self._execution.batch_size is not None:
            res["batch_size"] = self.batch_size
        if self._ephemeral is not None:
            res["ephemeral"] = self.ephemeral
        return res

    def add(self, settings: "Settings") -> None:
        if settings._execution.cache is not None:
            self._execution.cache = settings._execution.cache
        if settings._execution.prefetch is not None:
            self._execution.prefetch = settings._execution.prefetch
        if settings._execution.parallel is not None:
            self._execution.parallel = settings._execution.parallel
        if settings._execution.workers is not None:
            self._execution.workers = settings._execution.workers
        if settings._project.namespace is not None:
            self._project.namespace = settings._project.namespace
        if settings._project.project is not None:
            self._project.project = settings._project.project
        if settings._execution.min_task_size is not None:
            self._execution.min_task_size = settings._execution.min_task_size
        if settings._execution.batch_size is not None:
            self._execution.batch_size = settings._execution.batch_size
        if settings._ephemeral is not None:
            self._ephemeral = settings._ephemeral
