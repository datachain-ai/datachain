import pytest

import datachain as dc
from datachain.query.session import Session


def test_listing_dataset_lifecycle(tmp_path, catalog):
    p = tmp_path / "hello.txt"
    p.write_text("hello", encoding="utf-8")

    session_name = "asd3d5"
    ds_name = "my_test_ds13"

    with pytest.raises(ValueError):
        with Session(session_name, catalog=catalog):
            ds_name = "my_test_ds13"
            dc.read_storage(str(tmp_path)).exec()
            dc.read_values(key=["a", "b", "c"]).save(ds_name)
            raise ValueError("This is a test exception")

    # Datasets persist even if session fails with exception
    dataset = catalog.get_dataset(ds_name)
    assert dataset is not None

    assert dc.listings(catalog=catalog).count() == 1

    # Cleanup for test
    catalog.remove_dataset(ds_name, force=True)


def test_catalog_inside_another_session_keeps_its_database_and_cleanup(
    catalog, tmp_path, metastore_tmpfile
):
    folder = tmp_path / "input"
    folder.mkdir()
    source = folder / "data.txt"
    source.write_text("contents")
    other = catalog.copy()
    other.metastore = metastore_tmpfile
    for owner in ["first", "second"]:
        with Session(owner, catalog=other) as parent:
            assert parent.catalog.metastore is not catalog.metastore
            rows = [
                row
                for _, result in catalog.ls(
                    [str(folder) + "/"], fields=["name"], update=True
                )
                for row in result
            ]
            assert rows == [("data.txt",)]
            assert catalog.session.catalog is catalog
            assert catalog.session.name == parent.name
            assert list(
                catalog.metastore.list_datasets_by_prefix(parent.get_temp_prefix())
            )
        assert not list(
            catalog.metastore.list_datasets_by_prefix(parent.get_temp_prefix())
        )
    assert list(catalog.listings())
