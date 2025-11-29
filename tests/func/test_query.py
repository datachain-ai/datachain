from textwrap import dedent

import cloudpickle
import pytest

from datachain.data_storage import AbstractDBMetastore
from datachain.job import Job


@pytest.fixture
def catalog_info_filepath(cloud_test_catalog_tmpfile, tmp_path):
    catalog = cloud_test_catalog_tmpfile.catalog

    catalog_info = {
        "catalog_init_params": catalog.get_init_params(),
        "metastore_params": catalog.metastore.clone_params(),
        "warehouse_params": catalog.warehouse.clone_params(),
    }
    catalog_info_filepath = tmp_path / "catalog-info"
    with open(catalog_info_filepath, "wb") as f:
        cloudpickle.dump(catalog_info, f)

    return catalog_info_filepath


def setup_catalog(query: str, catalog_info_filepath: str) -> str:
    query_catalog_setup = f"""\
    import cloudpickle
    from datachain.catalog import Catalog
    from datachain.query.session import Session

    catalog_info_filepath = {str(catalog_info_filepath)!r}
    with open(catalog_info_filepath, "rb") as f:
        catalog_info = cloudpickle.load(f)
    (
        metastore_class,
        metastore_args,
        metastore_kwargs,
    ) = catalog_info["metastore_params"]
    metastore = metastore_class(*metastore_args, **metastore_kwargs)
    (
        warehouse_class,
        warehouse_args,
        warehouse_kwargs,
    ) = catalog_info["warehouse_params"]
    warehouse = warehouse_class(*warehouse_args, **warehouse_kwargs)
    catalog = Catalog(
        metastore=metastore,
        warehouse=warehouse,
        **catalog_info["catalog_init_params"],
    )
    session = Session("test", catalog=catalog)
    """
    return dedent(query_catalog_setup + "\n" + query)


def get_latest_job(metastore: AbstractDBMetastore) -> Job:
    j = metastore._jobs
    query = metastore._jobs_select().order_by(j.c.created_at.desc()).limit(1)
    (row,) = metastore.db.execute(query)
    return metastore._parse_job(row)
