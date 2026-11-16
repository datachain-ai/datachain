# ruff: noqa: TC004

from typing import TYPE_CHECKING

from datachain.lazy import lazy_exports

if TYPE_CHECKING:
    from .job import JobQueryType, JobStatus
    from .metastore import AbstractDBMetastore, AbstractMetastore
    from .warehouse import AbstractWarehouse

_LAZY = {
    "AbstractDBMetastore": "datachain.data_storage.metastore:AbstractDBMetastore",
    "AbstractMetastore": "datachain.data_storage.metastore:AbstractMetastore",
    "AbstractWarehouse": "datachain.data_storage.warehouse:AbstractWarehouse",
    "JobQueryType": "datachain.data_storage.job:JobQueryType",
    "JobStatus": "datachain.data_storage.job:JobStatus",
}

__all__ = [
    "AbstractDBMetastore",
    "AbstractMetastore",
    "AbstractWarehouse",
    "JobQueryType",
    "JobStatus",
]


lazy_exports(globals(), _LAZY)
