import os
import pathlib
from unittest.mock import patch

import pytest

from datachain.catalog.loader import (
    get_catalog,
    get_metastore,
    get_udf_distributor_class,
    get_warehouse,
)
from datachain.data_storage.sqlite import (
    SQLiteMetastore,
    SQLiteWarehouse,
)
from datachain.dataset import StorageURI


class DistributedClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_get_metastore(sqlite_db):
    uri = StorageURI("s3://bucket")

    metastore = SQLiteMetastore(uri, sqlite_db)
    assert metastore.uri == uri
    assert metastore.db == sqlite_db

    with patch.dict(os.environ, {"DATACHAIN__METASTORE": metastore.serialize()}):
        metastore2 = get_metastore(None)
        assert metastore2
        assert isinstance(metastore2, SQLiteMetastore)
        assert metastore2.uri == uri
        assert metastore2.db.db_file == sqlite_db.db_file
        assert metastore2.clone_params() == metastore.clone_params()

    with patch.dict(os.environ, {"DATACHAIN__METASTORE": sqlite_db.serialize()}):
        with pytest.raises(RuntimeError, match="instance of AbstractMetastore"):
            get_metastore(None)


def test_get_metastore_in_memory():
    if os.environ.get("DATACHAIN_METASTORE"):
        with pytest.raises(RuntimeError):
            get_metastore(in_memory=True)
    else:
        metastore = get_metastore(in_memory=True)
        assert isinstance(metastore, SQLiteMetastore)
        assert metastore.db.db_file == ":memory:"
        metastore.close()


def test_get_warehouse(sqlite_db):
    warehouse = SQLiteWarehouse(sqlite_db)
    assert warehouse.db == sqlite_db

    with patch.dict(os.environ, {"DATACHAIN__WAREHOUSE": warehouse.serialize()}):
        warehouse2 = get_warehouse(None)
        assert warehouse2
        assert isinstance(warehouse2, SQLiteWarehouse)
        assert warehouse2.db.db_file == sqlite_db.db_file
        assert warehouse2.clone_params() == warehouse.clone_params()

    with patch.dict(os.environ, {"DATACHAIN__WAREHOUSE": sqlite_db.serialize()}):
        with pytest.raises(RuntimeError, match="instance of AbstractWarehouse"):
            get_warehouse(None)


def test_get_warehouse_in_memory():
    if os.environ.get("DATACHAIN_WAREHOUSE"):
        with pytest.raises(RuntimeError):
            get_warehouse(in_memory=True)
    else:
        warehouse = get_warehouse(in_memory=True)
        assert isinstance(warehouse, SQLiteWarehouse)
        assert warehouse.db.db_file == ":memory:"
        warehouse.close()


def test_get_distributed_class(monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED_DISABLED", raising=False)

    with patch.dict(os.environ, {"DATACHAIN_DISTRIBUTED": ""}):
        assert get_udf_distributor_class() is None

    with patch.dict(
        os.environ,
        {"DATACHAIN_DISTRIBUTED": "tests.unit.test_catalog_loader.DistributedClass"},
    ):
        distributed_class = get_udf_distributor_class()
        assert distributed_class == DistributedClass
        assert distributed_class.__name__ == "DistributedClass"
        assert distributed_class.__module__ == "tests.unit.test_catalog_loader"

    with patch.dict(
        os.environ,
        {
            "DATACHAIN_DISTRIBUTED_PYTHONPATH": str(pathlib.Path(__file__).parent),
            "DATACHAIN_DISTRIBUTED": "test_catalog_loader.DistributedClass",
        },
    ):
        distributed_class = get_udf_distributor_class()
        assert distributed_class.__name__ == "DistributedClass"
        assert distributed_class.__module__ == "test_catalog_loader"

    with patch.dict(
        os.environ,
        {"DATACHAIN_DISTRIBUTED": "tests.unit.test_catalog_loader.NonExistent"},
    ):
        with pytest.raises(AttributeError, match="has no attribute 'NonExistent'"):
            get_udf_distributor_class()

    with patch.dict(os.environ, {"DATACHAIN_DISTRIBUTED": "DistributionClass"}):
        with pytest.raises(
            RuntimeError, match="Invalid DATACHAIN_DISTRIBUTED import path"
        ):
            get_udf_distributor_class()


def test_get_catalog(sqlite_db):
    uri = StorageURI("s3://bucket")
    metastore = SQLiteMetastore(uri, sqlite_db)
    warehouse = SQLiteWarehouse(sqlite_db)
    env = {
        "DATACHAIN__METASTORE": metastore.serialize(),
        "DATACHAIN__WAREHOUSE": warehouse.serialize(),
    }

    with patch.dict(os.environ, env):
        catalog = get_catalog()
        assert catalog

        assert catalog.metastore
        assert isinstance(catalog.metastore, SQLiteMetastore)
        assert catalog.metastore.uri == uri
        assert catalog.metastore.db.db_file == sqlite_db.db_file
        assert catalog.metastore.clone_params() == metastore.clone_params()

        assert catalog.warehouse
        assert isinstance(catalog.warehouse, SQLiteWarehouse)
        assert catalog.warehouse.db.db_file == sqlite_db.db_file
        assert catalog.warehouse.clone_params() == warehouse.clone_params()
