import os
import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any

from datachain.plugins import ensure_plugins_loaded
from datachain.utils import get_envs_by_prefix

if TYPE_CHECKING:
    from datachain.catalog import Catalog
    from datachain.data_storage import AbstractMetastore, AbstractWarehouse
    from datachain.query.udf import AbstractUDFDistributor

METASTORE_SERIALIZED = "DATACHAIN__METASTORE"
METASTORE_IMPORT_PATH = "DATACHAIN_METASTORE"
METASTORE_ARG_PREFIX = "DATACHAIN_METASTORE_ARG_"
WAREHOUSE_SERIALIZED = "DATACHAIN__WAREHOUSE"
WAREHOUSE_IMPORT_PATH = "DATACHAIN_WAREHOUSE"
WAREHOUSE_ARG_PREFIX = "DATACHAIN_WAREHOUSE_ARG_"
DISTRIBUTED_IMPORT_PYTHONPATH = "DATACHAIN_DISTRIBUTED_PYTHONPATH"
DISTRIBUTED_IMPORT_PATH = "DATACHAIN_DISTRIBUTED"
DISTRIBUTED_DISABLED = "DATACHAIN_DISTRIBUTED_DISABLED"

IN_MEMORY_ERROR_MESSAGE = "In-memory is only supported on SQLite"


def get_metastore(in_memory: bool = False) -> "AbstractMetastore":
    ensure_plugins_loaded()

    from datachain.data_storage import AbstractMetastore
    from datachain.data_storage.serializer import deserialize

    metastore_serialized = os.environ.get(METASTORE_SERIALIZED)
    if metastore_serialized:
        metastore_obj = deserialize(metastore_serialized)
        if not isinstance(metastore_obj, AbstractMetastore):
            raise RuntimeError(
                "Deserialized Metastore is not an instance of AbstractMetastore: "
                f"{metastore_obj}"
            )
        return metastore_obj

    metastore_import_path = os.environ.get(METASTORE_IMPORT_PATH)
    metastore_arg_envs = get_envs_by_prefix(METASTORE_ARG_PREFIX)
    # Convert env variable names to keyword argument names by lowercasing them
    metastore_args: dict[str, Any] = {
        k.lower(): v for k, v in metastore_arg_envs.items()
    }

    if not metastore_import_path:
        from datachain.data_storage.sqlite import SQLiteMetastore

        metastore_args["in_memory"] = in_memory
        return SQLiteMetastore(**metastore_args)
    if in_memory:
        raise RuntimeError(IN_MEMORY_ERROR_MESSAGE)
    # Metastore paths are specified as (for example):
    # datachain.data_storage.SQLiteMetastore
    if "." not in metastore_import_path:
        raise RuntimeError(
            f"Invalid {METASTORE_IMPORT_PATH} import path: {metastore_import_path}"
        )
    module_name, _, class_name = metastore_import_path.rpartition(".")
    metastore = import_module(module_name)
    metastore_class = getattr(metastore, class_name)
    return metastore_class(**metastore_args)


def get_warehouse(in_memory: bool = False) -> "AbstractWarehouse":
    ensure_plugins_loaded()

    from datachain.data_storage import AbstractWarehouse
    from datachain.data_storage.serializer import deserialize

    warehouse_serialized = os.environ.get(WAREHOUSE_SERIALIZED)
    if warehouse_serialized:
        warehouse_obj = deserialize(warehouse_serialized)
        if not isinstance(warehouse_obj, AbstractWarehouse):
            raise RuntimeError(
                "Deserialized Warehouse is not an instance of AbstractWarehouse: "
                f"{warehouse_obj}"
            )
        return warehouse_obj

    warehouse_import_path = os.environ.get(WAREHOUSE_IMPORT_PATH)
    warehouse_arg_envs = get_envs_by_prefix(WAREHOUSE_ARG_PREFIX)
    # Convert env variable names to keyword argument names by lowercasing them
    warehouse_args: dict[str, Any] = {
        k.lower(): v for k, v in warehouse_arg_envs.items()
    }

    if not warehouse_import_path:
        from datachain.data_storage.sqlite import SQLiteWarehouse

        warehouse_args["in_memory"] = in_memory
        return SQLiteWarehouse(**warehouse_args)
    if in_memory:
        raise RuntimeError(IN_MEMORY_ERROR_MESSAGE)
    # Warehouse paths are specified as (for example):
    # datachain.data_storage.SQLiteWarehouse
    if "." not in warehouse_import_path:
        raise RuntimeError(
            f"Invalid {WAREHOUSE_IMPORT_PATH} import path: {warehouse_import_path}"
        )
    module_name, _, class_name = warehouse_import_path.rpartition(".")
    warehouse = import_module(module_name)
    warehouse_class = getattr(warehouse, class_name)
    return warehouse_class(**warehouse_args)


def get_udf_distributor_class() -> type["AbstractUDFDistributor"] | None:
    if os.environ.get(DISTRIBUTED_DISABLED) == "True":
        return None

    if not (distributed_import_path := os.environ.get(DISTRIBUTED_IMPORT_PATH)):
        return None

    # Distributed class paths are specified as (for example): module.classname
    if "." not in distributed_import_path:
        raise RuntimeError(
            f"Invalid {DISTRIBUTED_IMPORT_PATH} import path: {distributed_import_path}"
        )

    # Optional: set the Python path to look for the module
    distributed_import_pythonpath = os.environ.get(DISTRIBUTED_IMPORT_PYTHONPATH)
    if distributed_import_pythonpath and distributed_import_pythonpath not in sys.path:
        sys.path.insert(0, distributed_import_pythonpath)

    module_name, _, class_name = distributed_import_path.rpartition(".")
    distributed = import_module(module_name)
    return getattr(distributed, class_name)


def get_catalog(
    client_config: dict[str, Any] | None = None,
    in_memory: bool = False,
) -> "Catalog":
    """
    Function that creates Catalog instance with appropriate metastore
    and warehouse classes. Metastore class can be provided with env variable
    DATACHAIN_METASTORE and if not provided, default one is used. Warehouse class
    can be provided with env variable DATACHAIN_WAREHOUSE and if not provided,

    If classes expects some kwargs, they can be provided via env variables
    by adding them with prefix (DATACHAIN_METASTORE_ARG_ and DATACHAIN_WAREHOUSE_ARG_)
    and name of variable after, e.g. if it accepts team_id as kwargs
    we can provide DATACHAIN_METASTORE_ARG_TEAM_ID=12345 env variable.
    """
    from datachain.catalog import Catalog

    metastore = get_metastore(in_memory=in_memory)
    return Catalog(
        metastore=metastore,
        warehouse=get_warehouse(in_memory=in_memory),
        client_config=client_config,
        in_memory=in_memory,
    )
