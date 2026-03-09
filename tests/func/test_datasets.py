import posixpath
import uuid
from unittest.mock import ANY

import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.data_storage.schema import DataTable
from datachain.dataset import DatasetDependencyType, DatasetStatus
from datachain.error import (
    DatasetInvalidVersionError,
    DatasetNotFoundError,
)
from datachain.lib.file import File
from datachain.lib.listing import parse_listing_uri
from datachain.query.dataset import DatasetQuery
from datachain.sql.types import Float32, Int, Int64
from tests.utils import assert_row_names, dataset_dependency_asdict, table_row_count

FILE_SCHEMA = {
    f"file__{name}": _type if _type != Int else Int64
    for name, _type in File._datachain_column_types.items()
}


def add_column(engine, table_name, column, catalog):
    # Simple method that adds new column to a table, with default value if specified
    column_name = column.compile(dialect=engine.dialect)
    column_type = column.type.compile(engine.dialect)
    query_str = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
    if column.default:
        query_str += f" DEFAULT {column.default.arg}"
    catalog.warehouse.db.execute_str(query_str)


@pytest.mark.parametrize("create_rows", [True, False])
def test_create_dataset_no_version_specified(cloud_test_catalog, project, create_rows):
    catalog = cloud_test_catalog.catalog

    name = uuid.uuid4().hex
    dataset = catalog.create_dataset(
        name,
        project,
        query_script="script",
        columns=[sa.Column("similarity", Float32)],
        create_rows=create_rows,
    )

    assert [v.version for v in dataset.versions] == ["1.0.0"]

    dataset_version = dataset.get_version("1.0.0")

    assert dataset.name == name
    assert dataset.project == project
    assert dataset_version.query_script == "script"
    assert dataset.schema["similarity"] == Float32
    assert dataset_version.schema["similarity"] == Float32
    assert dataset_version.status == DatasetStatus.CREATED
    assert dataset_version.uuid
    assert dataset.status == DatasetStatus.CREATED  # dataset status is deprecated
    if create_rows:
        assert dataset_version.num_objects == 0
    else:
        assert dataset_version.num_objects is None


@pytest.mark.parametrize("create_rows", [True, False])
def test_create_dataset_with_explicit_version(cloud_test_catalog, project, create_rows):
    catalog = cloud_test_catalog.catalog

    name = uuid.uuid4().hex
    dataset = catalog.create_dataset(
        name,
        project,
        version="1.0.0",
        query_script="script",
        columns=[sa.Column("similarity", Float32)],
        create_rows=create_rows,
    )

    dataset_version = dataset.get_version("1.0.0")

    assert dataset.name == name
    assert dataset.project == project
    assert dataset_version.query_script == "script"
    assert dataset.schema["similarity"] == Float32
    assert dataset_version.schema["similarity"] == Float32
    assert dataset_version.status == DatasetStatus.CREATED
    assert dataset_version.uuid
    assert dataset.status == DatasetStatus.CREATED
    if create_rows:
        assert dataset_version.num_objects == 0
    else:
        assert dataset_version.num_objects is None


def test_create_dataset_already_exist_but_in_different_project(
    cloud_test_catalog, dogs_dataset, project
):
    catalog = cloud_test_catalog.catalog
    dataset = catalog.create_dataset(
        dogs_dataset.name,
        project,
        query_script="script",
        columns=[sa.Column("similarity", Float32)],
    )

    assert dataset.id != dogs_dataset.id
    assert dataset.name == dogs_dataset.name
    assert dataset.project == project
    assert dataset.project != dogs_dataset.project


@pytest.mark.parametrize("create_rows", [True, False])
def test_create_dataset_already_exist(cloud_test_catalog, dogs_dataset, create_rows):
    catalog = cloud_test_catalog.catalog

    dataset = catalog.create_dataset(
        dogs_dataset.name,
        dogs_dataset.project,
        query_script="script",
        columns=[sa.Column("similarity", Float32)],
        create_rows=create_rows,
    )

    assert sorted([v.version for v in dataset.versions]) == sorted(["1.0.0", "1.0.1"])

    dataset_version = dataset.get_version("1.0.1")

    assert dataset.name == dogs_dataset.name
    assert dataset.project == dogs_dataset.project
    assert dataset_version.query_script == "script"
    assert dataset_version.schema["similarity"] == Float32
    assert dataset_version.status == DatasetStatus.CREATED
    assert dataset.status == DatasetStatus.COMPLETE
    if create_rows:
        assert dataset_version.num_objects == 0
    else:
        assert dataset_version.num_objects is None


@pytest.mark.parametrize("create_rows", [True, False])
def test_create_dataset_already_exist_wrong_version(
    cloud_test_catalog, dogs_dataset, create_rows
):
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DatasetInvalidVersionError) as exc_info:
        catalog.create_dataset(
            dogs_dataset.name,
            dogs_dataset.project,
            version="1.0.0",
            columns=[sa.Column(name, typ) for name, typ in dogs_dataset.schema.items()],
            create_rows=create_rows,
        )
    assert str(exc_info.value) == (
        f"Version 1.0.0 already exists in dataset {dogs_dataset.name}"
    )


def test_get_dataset(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    dataset = catalog.get_dataset(dogs_dataset.name)
    assert dataset.name == dogs_dataset.name

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("wrong name")


def test_create_dataset_from_sources(listed_bucket, cloud_test_catalog, project):
    dataset_name = uuid.uuid4().hex
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

    dataset = catalog.create_dataset_from_sources(
        dataset_name, [f"{src_uri}/dogs/"], project, recursive=True
    )

    dataset_version = dataset.get_version(dataset.latest_version)

    assert dataset.name == dataset_name
    assert dataset.description is None
    assert [v.version for v in dataset.versions] == ["1.0.0"]
    assert dataset.attrs == []
    assert dataset.status == DatasetStatus.COMPLETE
    assert dataset.project == project

    assert dataset_version.status == DatasetStatus.COMPLETE
    assert dataset_version.created_at
    assert dataset_version.finished_at
    assert dataset_version.error_message == ""
    assert dataset_version.error_stack == ""
    assert dataset_version.script_output == ""
    assert dataset_version.sources == f"{src_uri}/dogs/"
    assert dataset_version.uuid

    dr = catalog.warehouse.schema.dataset_row_cls
    sys_schema = {c.name: type(c.type) for c in dr.sys_columns()}
    default_dataset_schema = FILE_SCHEMA | sys_schema
    assert dataset.schema == default_dataset_schema
    assert dataset.query_script == ""

    assert dataset_version.schema == default_dataset_schema
    assert dataset_version.query_script == ""
    assert dataset_version.num_objects
    assert dataset_version.preview


def test_create_dataset_from_sources_dataset(cloud_test_catalog, dogs_dataset, project):
    dataset_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    ds_uri = f"ds://{dogs_dataset.full_name}"

    dataset = catalog.create_dataset_from_sources(
        dataset_name, [ds_uri], project, recursive=True
    )

    dataset_version = dataset.get_version(dataset.latest_version)

    assert dataset.name == dataset_name
    assert dataset.description is None
    assert [v.version for v in dataset.versions] == ["1.0.0"]
    assert dataset.attrs == []
    assert dataset.status == DatasetStatus.COMPLETE

    assert dataset_version.status == DatasetStatus.COMPLETE
    assert dataset_version.created_at
    assert dataset_version.finished_at
    assert dataset_version.error_message == ""
    assert dataset_version.error_stack == ""
    assert dataset_version.script_output == ""
    assert dataset_version.sources == ds_uri
    assert dataset_version.uuid

    dr = catalog.warehouse.schema.dataset_row_cls
    sys_schema = {c.name: type(c.type) for c in dr.sys_columns()}
    default_dataset_schema = FILE_SCHEMA | sys_schema
    assert dataset.schema == default_dataset_schema
    assert dataset.query_script == ""

    assert dataset_version.schema == default_dataset_schema
    assert dataset_version.query_script == ""
    assert dataset_version.num_objects
    assert dataset_version.preview


def test_create_dataset_from_sources_empty_sources(cloud_test_catalog, project):
    dataset_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog

    with pytest.raises(ValueError) as exc_info:
        catalog.create_dataset_from_sources(dataset_name, [], project, recursive=True)

    assert str(exc_info.value) == "Sources needs to be non empty list"


def test_create_dataset_from_sources_failed(
    listed_bucket, cloud_test_catalog, project, mocker
):
    dataset_name = uuid.uuid4().hex
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog
    # Mocks are automatically undone at the end of a test.
    mocker.patch.object(
        catalog.__class__,
        "listings",
        side_effect=RuntimeError("Error"),
    )
    with pytest.raises(RuntimeError):
        catalog.create_dataset_from_sources(
            dataset_name, [f"{src_uri}/dogs/"], project, recursive=True
        )

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(dataset_name)


def test_create_dataset_whole_bucket(listed_bucket, cloud_test_catalog, project):
    ctc = cloud_test_catalog
    src_uri = ctc.src_uri
    catalog = ctc.catalog

    dataset_name_1 = uuid.uuid4().hex
    dataset_name_2 = uuid.uuid4().hex

    ds1 = dc.read_storage(
        f"{src_uri}",
        session=ctc.session,
    ).save(dataset_name_1)
    ds2 = dc.read_storage(
        f"{src_uri}/",
        session=ctc.session,
    ).save(dataset_name_2)

    expected_rows = {
        "description",
        "cat1",
        "cat2",
        "dog1",
        "dog2",
        "dog3",
        "dog4",
    }

    assert_row_names(catalog, ds1.dataset, ds1.dataset.latest_version, expected_rows)
    assert_row_names(catalog, ds2.dataset, ds2.dataset.latest_version, expected_rows)


def test_remove_dataset(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    dataset_version = dogs_dataset.get_version("1.0.0")
    assert dataset_version.num_objects

    catalog.remove_dataset(dogs_dataset.name, dogs_dataset.project, force=True)
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(dogs_dataset.name)

    dataset_table_name = catalog.warehouse.dataset_table_name(dogs_dataset, "1.0.0")
    assert table_row_count(catalog.warehouse.db, dataset_table_name) is None

    assert (
        catalog.metastore.get_direct_dataset_dependencies(dogs_dataset, "1.0.0") == []
    )


def test_remove_dataset_with_multiple_versions(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    columns = tuple(sa.Column(name, typ) for name, typ in dogs_dataset.schema.items())
    updated_dogs_dataset = catalog.create_dataset_version(
        dogs_dataset, "2.0.0", columns=columns
    )
    assert updated_dogs_dataset.has_version("2.0.0")
    assert updated_dogs_dataset.has_version("1.0.0")

    catalog.remove_dataset(updated_dogs_dataset.name, dogs_dataset.project, force=True)
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(updated_dogs_dataset.name)

    assert (
        catalog.metastore.get_direct_dataset_dependencies(updated_dogs_dataset, "1.0.0")
        == []
    )


def test_remove_dataset_dataset_not_found(cloud_test_catalog, project):
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DatasetNotFoundError):
        catalog.remove_dataset("wrong_name", project, force=True)


def test_remove_dataset_wrong_version(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DatasetInvalidVersionError):
        catalog.remove_dataset(
            dogs_dataset.name, dogs_dataset.project, version="100.0.0"
        )


def test_edit_dataset(cloud_test_catalog, dogs_dataset):
    dataset_new_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog

    catalog.edit_dataset(
        dogs_dataset.name,
        dogs_dataset.project,
        new_name=dataset_new_name,
        description="new description",
        attrs=["cats", "birds"],
    )

    dataset = catalog.get_dataset(dataset_new_name)
    assert dataset.name == dataset_new_name
    assert dataset.description == "new description"
    assert dataset.attrs == ["cats", "birds"]

    # check if dataset tables are renamed correctly
    old_dataset_table_name = catalog.warehouse.dataset_table_name(dogs_dataset, "1.0.0")
    new_dataset_table_name = catalog.warehouse.dataset_table_name(dataset, "1.0.0")

    assert table_row_count(catalog.warehouse.db, old_dataset_table_name) is None
    expected_table_row_count = table_row_count(
        catalog.warehouse.db, new_dataset_table_name
    )
    assert expected_table_row_count
    assert dataset.get_version("1.0.0").num_objects == expected_table_row_count


@pytest.mark.parametrize(
    "old_name,new_name",
    [
        ("old.old.numbers", "new.new.numbers"),
        ("old.old.numbers", "new.new.numbers_new"),
        ("old.old.numbers", "old.new.numbers"),
        ("old.old.numbers", "old.old.numbers"),
        ("numbers", "numbers2"),
        ("numbers", "numbers"),
    ],
)
def test_move_dataset(
    test_session,
    old_name,
    new_name,
):
    catalog = test_session.catalog

    # create 2 versions of dataset in old project
    for _ in range(2):
        (dc.read_values(num=[1, 2, 3], session=test_session).save(old_name))

    dataset = dc.read_dataset(old_name).dataset

    dc.move_dataset(old_name, new_name, session=test_session)

    if old_name != new_name:
        # check that old dataset doesn't exist any more
        with pytest.raises(DatasetNotFoundError):
            dc.read_dataset(old_name).save("wrong")

    dataset_updated = dc.read_dataset(new_name).dataset

    # check if dataset tables are renamed correctly as well
    for version in [v.version for v in dataset.versions]:
        old_table_name = catalog.warehouse.dataset_table_name(dataset, version)
        new_table_name = catalog.warehouse.dataset_table_name(dataset_updated, version)
        if old_name == new_name:
            assert old_table_name == new_table_name
        else:
            assert table_row_count(catalog.warehouse.db, old_table_name) is None

        assert table_row_count(catalog.warehouse.db, new_table_name) == 3


def test_move_dataset_then_save_into(test_session):
    old_name = "old.old.numbers"
    new_name = "new.new.numbers"

    # create 2 versions of dataset in old project
    for _ in range(2):
        dc.read_values(num=[1, 2, 3], session=test_session).save(old_name)

    dc.move_dataset(old_name, new_name, session=test_session)
    dc.read_values(num=[1, 2, 3], session=test_session).save(new_name)

    ds = dc.datasets(column="dataset", session=test_session)
    datasets = [
        d
        for d in ds.to_values("dataset")
        if d.name == "numbers" and d.project == "new" and d.namespace == "new"
    ]

    assert len(datasets) == 3


def test_move_dataset_wrong_old_project(test_session, project):
    dc.read_values(num=[1, 2, 3], session=test_session).save("old.old.numbers")

    with pytest.raises(DatasetNotFoundError):
        dc.move_dataset("wrong.wrong.numbers", "new.new.numbers", session=test_session)


def test_move_dataset_error_in_session_moved_dataset_persisted(catalog):
    from datachain.query.session import Session

    old_name = "old.old.numbers"
    new_name = "new.new.numbers"

    with pytest.raises(DatasetNotFoundError):
        with Session("new", catalog=catalog) as test_session:
            dc.read_values(num=[1, 2, 3], session=test_session).save(old_name)
            dc.move_dataset(old_name, new_name, session=test_session)

            # throws DatasetNotFoundError
            dc.read_dataset("wrong", session=test_session)

    ds = dc.datasets(column="dataset", session=Session.get(catalog=catalog))
    datasets = [d for d in ds.to_values("dataset")]  # noqa: C416
    assert len(datasets) == 1
    assert datasets[0].name == "numbers"
    assert datasets[0].project == "new"
    assert datasets[0].namespace == "new"


def test_edit_dataset_same_name(cloud_test_catalog, dogs_dataset):
    dataset_new_name = dogs_dataset.name
    catalog = cloud_test_catalog.catalog

    catalog.edit_dataset(
        dogs_dataset.name, dogs_dataset.project, new_name=dataset_new_name
    )

    dataset = catalog.get_dataset(dataset_new_name)
    assert dataset.name == dataset_new_name

    # check if dataset tables are renamed correctly
    old_dataset_table_name = catalog.warehouse.dataset_table_name(dogs_dataset, "1.0.0")
    new_dataset_table_name = catalog.warehouse.dataset_table_name(dataset, "1.0.0")

    expected_table_row_count = table_row_count(
        catalog.warehouse.db, old_dataset_table_name
    )
    assert expected_table_row_count
    assert dataset.get_version("1.0.0").num_objects == expected_table_row_count
    assert expected_table_row_count == table_row_count(
        catalog.warehouse.db, new_dataset_table_name
    )


def test_edit_dataset_remove_attrs_and_description(cloud_test_catalog, dogs_dataset):
    dataset_new_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog

    catalog.edit_dataset(
        dogs_dataset.name,
        dogs_dataset.project,
        new_name=dataset_new_name,
        description="",
        attrs=[],
    )

    dataset = catalog.get_dataset(dataset_new_name)
    assert [v.version for v in dataset.versions] == ["1.0.0"]
    assert dataset.name == dataset_new_name
    assert dataset.description == ""
    assert dataset.attrs == []


def test_ls_dataset_rows(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    assert {
        posixpath.basename(r["file__path"])
        for r in catalog.ls_dataset_rows(dogs_dataset, "1.0.0")
    } == {
        "dog1",
        "dog2",
        "dog3",
        "dog4",
    }


def test_ls_dataset_rows_with_limit_offset(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    # these should be sorted by id already
    all_rows = list(catalog.ls_dataset_rows(dogs_dataset, "1.0.0"))

    assert {
        r["file__path"]
        for r in catalog.ls_dataset_rows(dogs_dataset, "1.0.0", offset=2, limit=1)
    } == {
        all_rows[2]["file__path"],
    }


def test_ls_dataset_rows_with_custom_columns(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    int_example = 25

    def test_types():
        return (
            5,
            5,
            5,
            0.5,
            0.5,
            0.5,
            [0.5],
            [[0.5], [0.5]],
            [0.5],
            [0.5],
            "s",
            True,
            {"a": 1},
            int_example.to_bytes(2, "big"),
        )

    chain = (
        dc.read_storage(cloud_test_catalog.src_uri, session=cloud_test_catalog.session)
        .map(
            test_types,
            params=[],
            output={
                "int_col": int,
                "int_col_32": int,
                "int_col_64": int,
                "float_col": float,
                "float_col_32": float,
                "float_col_64": float,
                "array_col": list[float],
                "array_col_nested": list[list[float]],
                "array_col_32": list[float],
                "array_col_64": list[float],
                "string_col": str,
                "bool_col": bool,
                "json_col": dict,
                "binary_col": bytes,
            },
        )
        .save("dogs_custom_columns")
    )

    for r in catalog.ls_dataset_rows(chain.dataset, "1.0.0"):
        assert r["int_col"] == 5
        assert r["int_col_32"] == 5
        assert r["int_col_64"] == 5
        assert r["float_col"] == 0.5
        assert r["float_col_32"] == 0.5
        assert r["float_col_64"] == 0.5
        assert r["array_col"] == [0.5]
        assert r["array_col_nested"] == [[0.5], [0.5]]
        assert r["array_col_32"] == [0.5]
        assert r["array_col_64"] == [0.5]
        assert r["string_col"] == "s"
        assert r["bool_col"]
        assert r["json_col"] == {"a": 1}
        assert r["binary_col"] == int_example.to_bytes(2, "big")


def test_dataset_preview_custom_columns(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    int_example = 25

    def test_types():
        return (
            5,
            5,
            5,
            0.5,
            0.5,
            0.5,
            [0.5],
            [[0.5], [0.5]],
            [0.5],
            [0.5],
            "s",
            True,
            {"a": 1},
            int_example.to_bytes(2, "big"),
        )

    (
        dc.read_storage(cloud_test_catalog.src_uri, session=cloud_test_catalog.session)
        .map(
            test_types,
            params=[],
            output={
                "int_col": int,
                "int_col_32": int,
                "int_col_64": int,
                "float_col": float,
                "float_col_32": float,
                "float_col_64": float,
                "array_col": list[float],
                "array_col_nested": list[list[float]],
                "array_col_32": list[float],
                "array_col_64": list[float],
                "string_col": str,
                "bool_col": bool,
                "json_col": dict,
                "binary_col": bytes,
            },
        )
        .save("dogs_custom_columns")
    )

    for r in catalog.get_dataset("dogs_custom_columns").get_version("1.0.0").preview:
        assert r["int_col"] == 5
        assert r["int_col_32"] == 5
        assert r["int_col_64"] == 5
        assert r["float_col"] == 0.5
        assert r["float_col_32"] == 0.5
        assert r["float_col_64"] == 0.5
        assert r["array_col"] == [0.5]
        assert r["array_col_nested"] == [[0.5], [0.5]]
        assert r["array_col_32"] == [0.5]
        assert r["array_col_64"] == [0.5]
        assert r["string_col"] == "s"
        assert r["bool_col"]
        assert r["json_col"] == {"a": 1}
        assert r["binary_col"] == [0, 25]


def test_dataset_preview_order(test_session):
    ids = list(range(10000))
    order = ids[::-1]
    catalog = test_session.catalog
    dataset_name = "test"

    dc.read_values(id=ids, order=order, session=test_session).order_by("order").save(
        dataset_name
    )

    preview_values = []

    for r in catalog.get_dataset(dataset_name).get_version("1.0.0").preview:
        id = ids.pop()
        o = order.pop()
        entry = (id, o)
        preview_values.append((id, o))
        assert (r["id"], r["order"]) == entry

    dc.read_dataset(dataset_name, session=test_session).save(dataset_name)

    for r in catalog.get_dataset(dataset_name).get_version("1.0.1").preview:
        assert (r["id"], r["order"]) == preview_values.pop(0)

    dc.read_dataset(dataset_name, version="1.0.1", session=test_session).order_by(
        "id"
    ).save(dataset_name)

    for r in catalog.get_dataset(dataset_name).get_version("1.0.2").preview:
        assert r["id"] == ids.pop(0)
        assert r["order"] == order.pop(0)


def test_dataset_preview_last_modified(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    project = dogs_dataset.project

    DatasetQuery(
        name=dogs_dataset.name,
        namespace_name=project.namespace.name,
        project_name=project.name,
        catalog=catalog,
    ).save("dogs_custom_columns", project=project)

    for r in catalog.get_dataset("dogs_custom_columns").get_version("1.0.0").preview:
        assert isinstance(r.get("file__last_modified"), str)


@pytest.mark.parametrize("tree", [{str(i): str(i) for i in range(50)}], indirect=True)
def test_row_random(cloud_test_catalog):
    # Note: this is technically a probabilistic test, but the probability
    # of accidental failure is < 1e-10
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    chain = dc.read_storage(ctc.src_uri, session=ctc.session).save("test")
    random_values = [
        row["sys__rand"] for row in catalog.ls_dataset_rows(chain.dataset, "1.0.0")
    ]

    # Random values are unique
    assert len(set(random_values)) == len(random_values)

    RAND_MAX = DataTable.MAX_RANDOM  # noqa: N806

    # Values are drawn uniformly from range(2**63)
    assert 0 <= min(random_values) < 0.4 * RAND_MAX
    assert 0.6 * RAND_MAX < max(random_values) < RAND_MAX

    # Creating a new dataset preserves random values
    chain = dc.read_storage(ctc.src_uri, session=ctc.session).save("test2")
    random_values2 = {
        row["sys__rand"] for row in catalog.ls_dataset_rows(chain.dataset, "1.0.0")
    }
    assert random_values2 == set(random_values)


def test_dataset_stats_registered_ds(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    dataset = catalog.get_dataset(dogs_dataset.name).get_version("1.0.0")
    assert dataset.num_objects == 4
    assert dataset.size == 15
    rows_count = catalog.warehouse.dataset_rows_count(dogs_dataset, "1.0.0")
    assert rows_count == 4


@pytest.mark.parametrize("indirect", [True, False])
def test_dataset_storage_dependencies(cloud_test_catalog, cloud_type, indirect):
    ctc = cloud_test_catalog
    session = ctc.session
    catalog = session.catalog
    uri = cloud_test_catalog.src_uri
    dep_name, _, _ = parse_listing_uri(ctc.src_uri)

    ds_name = "some_ds"
    dc.read_storage(uri, session=ctc.session).save(ds_name)

    lst_ds_name, _, _ = parse_listing_uri(uri)
    lst_dataset = catalog.get_dataset(lst_ds_name)

    assert [
        dataset_dependency_asdict(d)
        for d in catalog.get_dataset_dependencies(ds_name, "1.0.0", indirect=indirect)
    ] == [
        {
            "id": ANY,
            "type": DatasetDependencyType.STORAGE,
            "name": dep_name,
            "namespace": catalog.metastore.system_namespace_name,
            "project": catalog.metastore.listing_project_name,
            "version": "1.0.0",
            "created_at": lst_dataset.get_version("1.0.0").created_at,
            "dependencies": [],
        }
    ]
