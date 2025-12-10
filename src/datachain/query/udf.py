from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from sqlalchemy import Select, Table

    from datachain.catalog import Catalog
    from datachain.query.batch import BatchingStrategy


class UdfInfo(TypedDict):
    udf_data: bytes
    catalog_init: dict[str, Any]
    metastore_clone_params: tuple[Callable[..., Any], list[Any], dict[str, Any]]
    warehouse_clone_params: tuple[Callable[..., Any], list[Any], dict[str, Any]]
    table: "Table"
    query: "Select"
    udf_fields: list[str]
    batching: "BatchingStrategy"
    processes: int | None
    is_generator: bool
    cache: bool
    rows_total: int
    batch_size: int


class AbstractUDFDistributor(ABC):
    @abstractmethod
    def __init__(
        self,
        catalog: "Catalog",
        table: "Table",
        query: "Select",
        udf_data: bytes,
        batching: "BatchingStrategy",
        workers: bool | int,
        processes: bool | int,
        udf_fields: list[str],
        rows_total: int,
        use_cache: bool,
        is_generator: bool = False,
        min_task_size: str | int | None = None,
        batch_size: int | None = None,
    ) -> None: ...

    @abstractmethod
    def __call__(self) -> None: ...

    @staticmethod
    @abstractmethod
    def run_udf() -> int: ...
