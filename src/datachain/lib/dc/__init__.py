"""Public DataChain API: the ``DataChain`` class and ``read_*`` constructors."""

# ruff: noqa: TC004

from typing import TYPE_CHECKING

from datachain import _LAZY as _ROOT
from datachain.lazy import lazy_exports

if TYPE_CHECKING:
    from .csv import read_csv
    from .database import read_database
    from .datachain import C, Column, ColumnExpr, DataChain, DataChainSchema
    from .datasets import (
        datasets,
        delete_dataset,
        describe,
        move_dataset,
        read_dataset,
    )
    from .hf import read_hf
    from .json import read_json
    from .lerobot import read_lerobot
    from .listings import listings
    from .mcap import read_mcap
    from .pandas import read_pandas
    from .parquet import read_parquet
    from .records import read_records
    from .storage import Listing, list_storage, read_storage
    from .utils import DatasetMergeError, DatasetPrepareError, Sys, is_local, is_studio
    from .values import read_values
    from .webdataset import read_webdataset
    from .zarr import read_zarr

_LAZY = {k: v for k, v in _ROOT.items() if ".lib.dc." in v} | {
    "DatasetMergeError": "datachain.lib.dc.utils:DatasetMergeError",
    "DatasetPrepareError": "datachain.lib.dc.utils:DatasetPrepareError",
}

__all__ = [
    "C",
    "Column",
    "ColumnExpr",
    "DataChain",
    "DataChainSchema",
    "DatasetMergeError",
    "DatasetPrepareError",
    "Listing",
    "Sys",
    "datasets",
    "delete_dataset",
    "describe",
    "is_local",
    "is_studio",
    "list_storage",
    "listings",
    "move_dataset",
    "read_csv",
    "read_database",
    "read_dataset",
    "read_hf",
    "read_json",
    "read_lerobot",
    "read_mcap",
    "read_pandas",
    "read_parquet",
    "read_records",
    "read_storage",
    "read_values",
    "read_webdataset",
    "read_zarr",
]

lazy_exports(globals(), _LAZY)
