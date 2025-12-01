import io
import posixpath
import uuid
from unittest.mock import ANY

import pytest
import sqlalchemy

from datachain.dataset import DatasetDependencyType, DatasetStatus
from datachain.error import DatasetNotFoundError
from datachain.lib.file import File
from datachain.lib.listing import parse_listing_uri
from datachain.lib.signal_schema import SignalSchema
from datachain.query import C, DatasetQuery, Object, Stream
from datachain.sql.functions import path as pathfunc
from datachain.sql.types import String
from tests.utils import assert_row_names, dataset_dependency_asdict


def from_result_row(col_names, row):
    return dict(zip(col_names, row, strict=False))


def create_dataset_query_mutate_schema(**mutations):
    schema_values = {"file": File}
    schema_values.update(mutations)
    return SignalSchema(schema_values)


@pytest.fixture
def dogs_cats_dataset(listed_bucket, cloud_test_catalog, dogs_dataset, cats_dataset):
    dataset_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    (
        DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
        .union(DatasetQuery(name=cats_dataset.name, version="1.0.0", catalog=catalog))
        .save(dataset_name)
    )
    return catalog.get_dataset(dataset_name)


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_save_dataset_version_already_exists(cloud_test_catalog, cats_dataset):
    catalog = cloud_test_catalog.catalog
    DatasetQuery(cats_dataset.name, catalog=catalog).save("cats", version="1.0.0")
    with pytest.raises(RuntimeError) as exc_info:
        DatasetQuery(cats_dataset.name, catalog=catalog).save("cats", version="1.0.0")

    assert str(exc_info.value) == "Dataset cats already has version 1.0.0"


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_save_multiple_versions(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    # ensure we can select a subset of a bucket properly
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    ds_name = "animals_cats"
    q = ds
    q.save(ds_name)

    q = q.filter(C("file.path").glob("cats*") | (C("file.size") < 4))
    q.save(ds_name)
    q.save(ds_name)

    dataset_record = catalog.get_dataset(ds_name)
    assert dataset_record.status == DatasetStatus.COMPLETE
    assert DatasetQuery(name=ds_name, version="1.0.0", catalog=catalog).count() == 7
    assert DatasetQuery(name=ds_name, version="1.0.1", catalog=catalog).count() == 3
    assert DatasetQuery(name=ds_name, version="1.0.2", catalog=catalog).count() == 3

    with pytest.raises(DatasetNotFoundError):
        DatasetQuery(name=ds_name, version="4.0.0", catalog=catalog).count()


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_filter(cloud_test_catalog, save, cats_dataset):
    catalog = cloud_test_catalog.catalog
    # ensure we can select a subset of a bucket properly
    ds = DatasetQuery(cats_dataset.name, catalog=catalog)
    q = (
        ds.filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .filter(C("file.path").regexp("^cats/cat[0-9]$"))
    )
    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        q = DatasetQuery(name=ds_name, catalog=catalog)
        dataset_record = catalog.get_dataset(ds_name)
        assert dataset_record.status == DatasetStatus.COMPLETE
    result = q.db_results()
    count = q.count()
    assert len(result) == 2
    assert count == 2


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_instance_returned_after_save(cloud_test_catalog, dogs_cats_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(name=dogs_cats_dataset.name, version="1.0.0", catalog=catalog)

    ds2 = ds.save("dogs_cats_2")
    assert isinstance(ds2, DatasetQuery)
    expected_names = {"cat1", "cat2", "dog1", "dog2", "dog3", "dog4"}
    assert_row_names(catalog, dogs_cats_dataset, "1.0.0", expected_names)
    assert_row_names(
        catalog, catalog.get_dataset("dogs_cats_2"), "1.0.0", expected_names
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_query_specific_dataset_set_proper_dataset_name_version(
    cloud_test_catalog, dogs_cats_dataset
):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(name=dogs_cats_dataset.name, version="1.0.0", catalog=catalog)
    assert ds.name == dogs_cats_dataset.name
    assert ds.version == "1.0.0"


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_save_set_proper_dataset_name_version(cloud_test_catalog, dogs_cats_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(name=dogs_cats_dataset.name, version="1.0.0", catalog=catalog)
    ds = ds.filter(C("file.path").glob("*dog*"))
    ds2 = ds.save("dogs_small")

    assert ds2.name == "dogs_small"
    assert ds2.version == "1.0.0"
    assert len(ds2.steps) == 0

    # old dataset query remains detached
    assert ds.name is None
    assert ds.version is None


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_reset_dataset_name_version_after_filter(cloud_test_catalog, dogs_cats_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(name=dogs_cats_dataset.name, version="1.0.0", catalog=catalog)
    ds2 = ds.save("dogs_small")
    assert ds2.name == "dogs_small"
    assert ds2.version == "1.0.0"

    ds3 = ds2.filter(C("file.path").glob("*dog1"))
    assert ds3.name is None
    assert ds3.version is None

    # old ds2 remains attached
    assert ds2.name == "dogs_small"
    assert ds2.version == "1.0.0"


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_chain_after_save(cloud_test_catalog, dogs_cats_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(name=dogs_cats_dataset.name, version="1.0.0", catalog=catalog)
    ds.filter(C("file.path").glob("*dog*")).save("ds1").filter(C("file.size") < 4).save(
        "ds2"
    )

    assert_row_names(
        catalog, catalog.get_dataset("ds1"), "1.0.0", {"dog1", "dog2", "dog3", "dog4"}
    )
    assert_row_names(catalog, catalog.get_dataset("ds2"), "1.0.0", {"dog2"})


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_select(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    q = (
        ds.order_by(C("file.size").desc())
        .limit(6)
        .select(
            C("file.size"), size10x=C("file.size") * 10, size100x=C("file.size") * 100
        )
    )
    result = q.db_results()
    assert result == [
        (13, 130, 1300),
        (4, 40, 400),
        (4, 40, 400),
        (4, 40, 400),
        (4, 40, 400),
        (4, 40, 400),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_select_missing_column(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    ds1 = ds.select(C.missing_column_name)
    ds2 = ds.select("missing_column_name")
    # The exception type varies by database backend
    exc1 = pytest.raises(Exception, ds1.db_results)  # noqa: B017
    assert "missing_column_name" in str(exc1.value)
    exc2 = pytest.raises(KeyError, ds2.db_results)
    assert "missing_column_name" in str(exc2.value)


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_select_except(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    q = (
        ds.order_by(C("file.size").desc())
        .limit(6)
        .select(
            C("file.path"),
            C("file.size"),
            size10x=C("file.size") * 10,
            size100x=C("file.size") * 100,
        )
        .select_except(C("file.path"), C.size10x)
    )
    result = q.db_results()
    assert result == [
        (13, 1300),
        (4, 400),
        (4, 400),
        (4, 400),
        (4, 400),
        (4, 400),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_distinct(cloud_test_catalog, animal_dataset):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)

    q = (
        ds.select(pathfunc.name(C("file.path")), C("file.size"))
        .order_by(pathfunc.name(C("file.path")))
        .distinct(C("file.size"))
    )
    result = q.db_results()

    assert result == [("cat1", 4), ("description", 13), ("dog2", 3)]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_distinct_count(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)

    assert ds.distinct(C("file.size")).count() == 3
    assert ds.distinct(C("file.path")).count() == 7
    assert ds.distinct().count() == 7


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_mutate(cloud_test_catalog, save, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    schema = create_dataset_query_mutate_schema(size10x=int, size1000x=int)
    q = (
        ds.mutate(new_schema=schema, size10x=C("file.size") * 10)
        .mutate(new_schema=schema, size1000x=C.size10x * 100)
        .mutate(
            ("s2", C("file.size") * 2),
            ("s3", C("file.size") * 3),
            new_schema=schema,
            s4=C("file.size") * 4,
        )
        .filter((C.size10x < 40) | (C.size10x > 100) | C("file.path").glob("cat*"))
        .order_by(C.size10x.desc(), C("file.path"))
    )
    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        new_query = DatasetQuery(name=ds_name, catalog=catalog).order_by(
            C.size10x.desc(), C("file.path")
        )
        result = new_query.db_results(
            row_factory=lambda c, v: dict(zip(c, v, strict=False))
        )
        dataset_record = catalog.get_dataset(ds_name)
        assert dataset_record.status == DatasetStatus.COMPLETE
    else:
        result = q.db_results(row_factory=lambda c, v: dict(zip(c, v, strict=False)))
    assert len(result) == 4
    assert len(result[0]) == 15
    cols = {"size10x", "size1000x", "s2", "s3", "s4"}
    new_data = [[v for k, v in r.items() if k in cols] for r in result]
    assert new_data == [
        [130, 13000, 26, 39, 52],
        [40, 4000, 8, 12, 16],
        [40, 4000, 8, 12, 16],
        [30, 3000, 6, 9, 12],
    ]


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_order_by_after_mutate(cloud_test_catalog, save, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    schema = create_dataset_query_mutate_schema(size10x=int)
    q = (
        ds.mutate(new_schema=schema, size10x=C("file.size") * 10)
        .filter((C.size10x < 40) | (C.size10x > 100) | C("file.path").glob("cat*"))
        .order_by(C.size10x.desc())
    )

    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        result = (
            DatasetQuery(name=ds_name, catalog=catalog)
            .order_by(C.size10x.desc(), pathfunc.name(C("file.path")))
            .db_results(row_factory=lambda c, v: dict(zip(c, v, strict=False)))
        )
    else:
        result = q.db_results(row_factory=lambda c, v: dict(zip(c, v, strict=False)))

    assert [r["size10x"] for r in result] == [130, 40, 40, 30]


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_order_by_limit(cloud_test_catalog, save, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    q = ds.order_by(pathfunc.name(C("file.path")).desc()).limit(5)
    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        new_query = DatasetQuery(name=ds_name, catalog=catalog).order_by(
            pathfunc.name(C("file.path")).desc()
        )
        result = new_query.db_results()
        dataset_record = catalog.get_dataset(ds_name)
        assert dataset_record.status == DatasetStatus.COMPLETE
    else:
        result = q.db_results()

    assert [posixpath.basename(r[3]) for r in result] == [
        "dog4",
        "dog3",
        "dog2",
        "dog1",
        "description",
    ]


@pytest.mark.parametrize("save", [True, False])
def test_limit(cloud_test_catalog, save, animal_dataset):
    catalog = cloud_test_catalog.catalog
    q = (
        DatasetQuery(animal_dataset.name, catalog=catalog)
        .order_by(C("file.path"))
        .limit(2)
    )
    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        result = DatasetQuery(name=ds_name, catalog=catalog).db_results()
        dataset_record = catalog.get_dataset(ds_name)
        assert dataset_record.status == DatasetStatus.COMPLETE
    else:
        result = q.db_results()

    assert len(result) == 2
    assert [posixpath.basename(r[3]) for r in result] == ["cat1", "cat2"]


@pytest.mark.parametrize("save", [True, False])
def test_offset_limit(cloud_test_catalog, save, animal_dataset):
    catalog = cloud_test_catalog.catalog
    q = (
        DatasetQuery(animal_dataset.name, catalog=catalog)
        .order_by(C("file.path"))
        .offset(3)
        .limit(2)
    )
    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        result = DatasetQuery(name=ds_name, catalog=catalog).db_results()
        dataset_record = catalog.get_dataset(ds_name)
        assert dataset_record.status == DatasetStatus.COMPLETE
    else:
        result = q.db_results()

    assert len(result) == 2
    assert [posixpath.basename(r[3]) for r in result] == ["dog1", "dog2"]


@pytest.mark.parametrize("save", [True, False])
def test_mutate_offset_limit(cloud_test_catalog, save, animal_dataset):
    catalog = cloud_test_catalog.catalog
    base_query = DatasetQuery(animal_dataset.name, catalog=catalog).order_by(
        C("file.path")
    )
    schema = create_dataset_query_mutate_schema(size10x=int)
    q = (
        base_query.mutate(new_schema=schema, size10x=C("file.size") * 10)
        .offset(3)
        .limit(2)
    )
    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        result = DatasetQuery(name=ds_name, catalog=catalog).db_results()
        dataset_record = catalog.get_dataset(ds_name)
        assert dataset_record.status == DatasetStatus.COMPLETE
    else:
        result = q.db_results()

    assert len(result) == 2
    assert [posixpath.basename(r[3]) for r in result] == ["dog1", "dog2"]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_number_without_explicit_order_by(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds_name = uuid.uuid4().hex

    DatasetQuery(animal_dataset.name, catalog=catalog).filter(C("file.size") > 0).save(
        ds_name
    )

    results = DatasetQuery(name=ds_name, catalog=catalog).to_db_records()
    assert len(results) == 7  # unordered, just checking num of results


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_number_with_order_by_name_descending(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds_name = uuid.uuid4().hex

    DatasetQuery(animal_dataset.name, catalog=catalog).order_by(
        pathfunc.name(C("file.path")).desc()
    ).save(ds_name)

    results = DatasetQuery(name=ds_name, catalog=catalog).to_db_records()
    results_name_id = [
        {k: v for k, v in r.items() if k in ["sys__id", "file__path"]} for r in results
    ]
    assert sorted(results_name_id, key=lambda k: k["sys__id"]) == [
        {"sys__id": 1, "file__path": "dogs/others/dog4"},
        {"sys__id": 2, "file__path": "dogs/dog3"},
        {"sys__id": 3, "file__path": "dogs/dog2"},
        {"sys__id": 4, "file__path": "dogs/dog1"},
        {"sys__id": 5, "file__path": "description"},
        {"sys__id": 6, "file__path": "cats/cat2"},
        {"sys__id": 7, "file__path": "cats/cat1"},
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_number_with_order_by_name_ascending(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds_name = uuid.uuid4().hex

    DatasetQuery(animal_dataset.name, catalog=catalog).order_by(
        pathfunc.name(C("file.path")).asc()
    ).save(ds_name)

    results = DatasetQuery(name=ds_name, catalog=catalog).to_db_records()
    results_name_id = [
        {k: v for k, v in r.items() if k in ["sys__id", "file__path"]} for r in results
    ]
    assert sorted(results_name_id, key=lambda k: k["sys__id"]) == [
        {"sys__id": 1, "file__path": "cats/cat1"},
        {"sys__id": 2, "file__path": "cats/cat2"},
        {"sys__id": 3, "file__path": "description"},
        {"sys__id": 4, "file__path": "dogs/dog1"},
        {"sys__id": 5, "file__path": "dogs/dog2"},
        {"sys__id": 6, "file__path": "dogs/dog3"},
        {"sys__id": 7, "file__path": "dogs/others/dog4"},
    ]


def to_str(buf) -> str:
    return io.TextIOWrapper(buf, encoding="utf8").read()


@pytest.mark.parametrize("use_cache", [False, True])
def test_extract(cloud_test_catalog, dogs_dataset, use_cache):
    catalog = cloud_test_catalog.catalog
    q = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    results = set()
    for path, stream in q.extract("file__path", Stream(), cache=use_cache):
        with stream:
            value = stream.read().decode("utf-8")
        results.add((posixpath.basename(path), value))
    assert results == {
        ("dog1", "woof"),
        ("dog2", "arf"),
        ("dog3", "bark"),
        ("dog4", "ruff"),
    }


def test_extract_object(cloud_test_catalog, dogs_dataset):
    ctc = cloud_test_catalog
    ds = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=ctc.catalog)
    data = ds.extract(Object(to_str), "file__path")
    assert {(value, posixpath.basename(path)) for value, path in data} == {
        ("woof", "dog1"),
        ("arf", "dog2"),
        ("bark", "dog3"),
        ("ruff", "dog4"),
    }


def test_extract_chunked(cloud_test_catalog, dogs_dataset):
    ctc = cloud_test_catalog
    n = 5
    all_data = []
    ds = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=ctc.catalog)
    for i in range(n):
        data = ds.chunk(i, n).extract(Object(to_str), "file__path")
        all_data.extend(data)

    assert {(value, posixpath.basename(path)) for value, path in all_data} == {
        ("woof", "dog1"),
        ("arf", "dog2"),
        ("bark", "dog3"),
        ("ruff", "dog4"),
    }


def test_extract_chunked_limit(cloud_test_catalog, dogs_dataset):
    ctc = cloud_test_catalog
    chunks = 5
    limit = 1
    all_data = []
    q = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=ctc.catalog)
    # Add sufficient rows to ensure each chunk has rows
    for _ in range(5):
        q = q.union(q)
    for i in range(chunks):
        data = q.limit(limit).chunk(i, chunks).extract(Object(to_str), "file__path")
        all_data.extend(data)

    assert len(all_data) == limit


@pytest.mark.parametrize(
    "cloud_type, version_aware",
    [("file", False)],
    indirect=True,
)
def test_extract_limit(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    q = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    results = list(q.limit(2).extract("file__path"))
    assert len(results) == 2


@pytest.mark.parametrize(
    "cloud_type, version_aware",
    [("file", False)],
    indirect=True,
)
def test_extract_order_by(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    q = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    results = list(q.order_by("sys__rand").extract("file__path"))
    pairs = list(q.extract("sys__rand", "file__path"))
    assert results == [(p[1],) for p in sorted(pairs)]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_union(cloud_test_catalog, cats_dataset, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version="1.0.0", catalog=catalog)

    (dogs | cats).save("dogs_cats")

    q = DatasetQuery(name="dogs_cats", version="1.0.0", catalog=catalog)
    result = q.db_results()
    count = q.count()
    assert len(result) == 6
    assert count == 6


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
@pytest.mark.parametrize("n_columns", [1, 2])
def test_join_with_binary_expression(
    cloud_test_catalog, dogs_dataset, dogs_cats_dataset, inner, n_columns
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    dogs_cats = DatasetQuery(
        name=dogs_cats_dataset.name, version="1.0.0", catalog=catalog
    )

    if n_columns == 1:
        predicate = dogs_cats.c("file__path") == dogs.c("file__path")
    else:
        predicate = (dogs_cats.c("file__path") == dogs.c("file__path")) & (
            dogs_cats.c("file__size") == dogs.c("file__size")
        )

    res = dogs_cats.join(
        dogs,
        predicate,
        inner=inner,
    ).to_db_records()

    if inner:
        expected = [
            ("dogs/dog1", "dogs/dog1"),
            ("dogs/dog2", "dogs/dog2"),
            ("dogs/dog3", "dogs/dog3"),
            ("dogs/others/dog4", "dogs/others/dog4"),
        ]
    else:
        string_default = String.default_value(catalog.warehouse.db.dialect)
        expected = [
            ("cats/cat1", string_default),
            ("cats/cat2", string_default),
            ("dogs/dog1", "dogs/dog1"),
            ("dogs/dog2", "dogs/dog2"),
            ("dogs/dog3", "dogs/dog3"),
            ("dogs/others/dog4", "dogs/others/dog4"),
        ]

    assert (
        sorted(
            ((r["file__path"], r["right_file__path"]) for r in res), key=lambda x: x[0]
        )
        == expected
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
@pytest.mark.parametrize("column_predicate", ["file__path", C("file.path")])
def test_join_with_combination_binary_expression_and_column_predicates(
    cloud_test_catalog,
    dogs_dataset,
    dogs_cats_dataset,
    inner,
    column_predicate,
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    dogs_cats = DatasetQuery(
        name=dogs_cats_dataset.name, version="1.0.0", catalog=catalog
    )

    res = dogs_cats.join(
        dogs,
        [column_predicate, dogs_cats.c("file__size") == dogs.c("file__size")],
        inner=inner,
    ).to_db_records()

    if inner:
        expected = [
            ("dogs/dog1", "dogs/dog1"),
            ("dogs/dog2", "dogs/dog2"),
            ("dogs/dog3", "dogs/dog3"),
            ("dogs/others/dog4", "dogs/others/dog4"),
        ]
    else:
        string_default = String.default_value(catalog.warehouse.db.dialect)
        expected = [
            ("cats/cat1", string_default),
            ("cats/cat2", string_default),
            ("dogs/dog1", "dogs/dog1"),
            ("dogs/dog2", "dogs/dog2"),
            ("dogs/dog3", "dogs/dog3"),
            ("dogs/others/dog4", "dogs/others/dog4"),
        ]

    assert (
        sorted(
            ((r["file__path"], r["right_file__path"]) for r in res), key=lambda x: x[0]
        )
        == expected
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
def test_join_with_binary_expression_with_arithmetics(
    cloud_test_catalog,
    dogs_dataset,
    cats_dataset,
    inner,
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version="1.0.0", catalog=catalog)

    res = cats.join(
        dogs, cats.c("file__size") == dogs.c("file__size") + 1, inner=inner
    ).to_db_records()

    assert sorted(
        ((r["file__path"], r["right_file__path"]) for r in res), key=lambda x: x[0]
    ) == [
        ("cats/cat1", "dogs/dog2"),
        ("cats/cat2", "dogs/dog2"),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_join_with_wrong_predicates(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    dogs1 = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    dogs2 = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, []).to_db_records()
    assert str(excinfo.value) == "Missing predicates"

    with pytest.raises(TypeError) as excinfo:
        dogs1.join(dogs2, [[]]).to_db_records()
    assert str(excinfo.value) == "Unsupported predicate [] for join expression"


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_join_with_missing_columns_in_expression(
    cloud_test_catalog, dogs_dataset, cats_dataset
):
    catalog = cloud_test_catalog.catalog

    dogs1 = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    dogs2 = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version="1.0.0", catalog=catalog)

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, dogs1.c("wrong") == dogs2.c("file__path")).to_db_records()
    assert str(excinfo.value) == "Column wrong was not found in left part of the join"

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, dogs1.c("file__path") == dogs2.c("wrong")).to_db_records()
    assert str(excinfo.value) == "Column wrong was not found in right part of the join"

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, dogs1.c("file__path") == cats.c("file__path")).to_db_records()
    assert str(excinfo.value) == (
        "Column file__path was not found in left or right part of the join"
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
def test_join_with_using_functions_in_expression(
    cloud_test_catalog, dogs_dataset, dogs_cats_dataset, inner
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    dogs_cats = DatasetQuery(
        name=dogs_cats_dataset.name, version="1.0.0", catalog=catalog
    )

    res = dogs_cats.join(
        dogs,
        (
            sqlalchemy.func.upper(dogs_cats.c("file__path"))
            == sqlalchemy.func.upper(dogs.c("file__path"))
        ),
        inner=inner,
    ).to_db_records()

    if inner:
        expected = [
            ("dogs/dog1", "dogs/dog1"),
            ("dogs/dog2", "dogs/dog2"),
            ("dogs/dog3", "dogs/dog3"),
            ("dogs/others/dog4", "dogs/others/dog4"),
        ]
    else:
        string_default = String.default_value(catalog.warehouse.db.dialect)
        expected = [
            ("cats/cat1", string_default),
            ("cats/cat2", string_default),
            ("dogs/dog1", "dogs/dog1"),
            ("dogs/dog2", "dogs/dog2"),
            ("dogs/dog3", "dogs/dog3"),
            ("dogs/others/dog4", "dogs/others/dog4"),
        ]

    assert (
        sorted(
            ((r["file__path"], r["right_file__path"]) for r in res), key=lambda x: x[0]
        )
        == expected
    )


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
def test_simple_dataset_query(cloud_test_catalog):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    metastore = catalog.metastore
    warehouse = catalog.warehouse
    catalog.create_dataset_from_sources("ds1", [ctc.src_uri], recursive=True)
    DatasetQuery(name="ds1", version="1.0.0", catalog=catalog).save("ds2")

    ds_queries = []
    for ds_name in ("ds1", "ds2"):
        ds = metastore.get_dataset(ds_name)
        dr = warehouse.dataset_rows(ds)
        dq = dr.select().order_by(dr.c("path"))
        ds_queries.append(dq)

    ds1, ds2 = (
        [
            {
                k.name: v
                for k, v in zip(q.selected_columns, r, strict=False)
                if k.name != "sys__id"
            }
            for r in warehouse.db.execute(q)
        ]
        for q in ds_queries
    )

    # everything except the id field should match
    assert ds1 == ds2
    assert [r["file__path"] for r in ds1] == [
        ("cats/cat1"),
        ("cats/cat2"),
        ("description"),
        ("dogs/dog1"),
        ("dogs/dog2"),
        ("dogs/dog3"),
        ("dogs/others/dog4"),
    ]


def test_aggregate(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    q = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    assert q.count() == 4
    assert q.sum(C("file.size")) == 15
    assert q.avg(C("file.size")) == 15 / 4
    assert q.min(C("file.size")) == 3
    assert q.max(C("file.size")) == 4


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_to_db_records(cloud_test_catalog, cats_dataset):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    ds = (
        DatasetQuery(cats_dataset.name, catalog=catalog)
        .select(C("file__path"), C("file__size"))
        .order_by(C("file__path"))
    )

    assert ds.to_db_records() == [
        {"file__path": "cats/cat1", "file__size": 4},
        {"file__path": "cats/cat2", "file__size": 4},
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True), ("file", False)],
    indirect=True,
)
@pytest.mark.parametrize("indirect", [True, False])
def test_dataset_dependencies_one_storage_as_dependency(
    cloud_test_catalog, listed_bucket, indirect, cats_dataset
):
    ds_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    listing = catalog.listings()[0]
    dep_name, _, _ = parse_listing_uri(cloud_test_catalog.src_uri)

    DatasetQuery(cats_dataset.name, catalog=catalog).save(ds_name)

    assert [
        dataset_dependency_asdict(d)
        for d in catalog.get_dataset_dependencies(
            cats_dataset.name, "1.0.0", indirect=indirect
        )
    ] == [
        {
            "id": ANY,
            "type": DatasetDependencyType.STORAGE,
            "name": dep_name,
            "namespace": catalog.metastore.system_namespace_name,
            "project": catalog.metastore.listing_project.name,
            "version": "1.0.0",
            "created_at": listing.created_at,
            "dependencies": [],
        }
    ]


@pytest.mark.parametrize("indirect", [False])
def test_dataset_dependencies_one_dataset_as_dependency(
    cloud_test_catalog, dogs_dataset, indirect
):
    ds_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    listing = catalog.listings()[0]

    dep_name, _, _ = parse_listing_uri(cloud_test_catalog.src_uri)

    DatasetQuery(name=dogs_dataset.name, catalog=catalog).save(ds_name)

    expected = [
        {
            "id": ANY,
            "type": DatasetDependencyType.DATASET,
            "name": dogs_dataset.name,
            "namespace": catalog.metastore.default_namespace_name,
            "project": catalog.metastore.default_project.name,
            "version": "1.0.0",
            "created_at": dogs_dataset.get_version("1.0.0").created_at,
            "dependencies": [],
        }
    ]

    if indirect:
        expected[0]["dependencies"] = [
            {
                "id": ANY,
                "type": DatasetDependencyType.STORAGE,
                "name": dep_name,
                "namespace": catalog.metastore.default_namespace_name,
                "project": catalog.metastore.default_project.name,
                "version": "1.0.0",
                "created_at": listing.created_at,
                "dependencies": [],
            }
        ]

    assert [
        dataset_dependency_asdict(d)
        for d in catalog.get_dataset_dependencies(ds_name, "1.0.0", indirect=indirect)
    ] == expected

    catalog.remove_dataset(dogs_dataset.name, force=True)
    # None means dependency was there but was removed in the meantime
    assert catalog.get_dataset_dependencies(ds_name, "1.0.0") == [None]


@pytest.mark.parametrize("method", ["union", "join"])
def test_dataset_dependencies_multiple_direct_dataset_dependencies(
    cloud_test_catalog, dogs_dataset, cats_dataset, method
):
    # multiple direct dataset dependencies can be achieved with methods that are
    # combining multiple DatasetQuery instances into new one like union or join
    ds_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    listing = catalog.listings()[0]
    dep_name, _, _ = parse_listing_uri(cloud_test_catalog.src_uri)

    dogs = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version="1.0.0", catalog=catalog)

    if method == "union":
        dogs.union(cats).save(ds_name)
    else:
        dogs.join(cats, "file__path").save(ds_name)

    storage_depenedncy = {
        "id": ANY,
        "type": DatasetDependencyType.STORAGE,
        "name": dep_name,
        "namespace": catalog.metastore.system_namespace_name,
        "project": catalog.metastore.listing_project_name,
        "version": "1.0.0",
        "created_at": listing.created_at,
        "dependencies": [],
    }

    expected = [
        {
            "id": ANY,
            "type": DatasetDependencyType.DATASET,
            "name": dogs_dataset.name,
            "namespace": catalog.metastore.default_namespace_name,
            "project": catalog.metastore.default_project.name,
            "version": "1.0.0",
            "created_at": dogs_dataset.get_version("1.0.0").created_at,
            "dependencies": [storage_depenedncy],
        },
        {
            "id": ANY,
            "type": DatasetDependencyType.DATASET,
            "name": cats_dataset.name,
            "namespace": catalog.metastore.default_namespace_name,
            "project": catalog.metastore.default_project.name,
            "version": "1.0.0",
            "created_at": cats_dataset.get_version("1.0.0").created_at,
            "dependencies": [storage_depenedncy],
        },
    ]

    assert sorted(
        (
            dataset_dependency_asdict(d)
            for d in catalog.get_dataset_dependencies(ds_name, "1.0.0", indirect=True)
        ),
        key=lambda d: d["name"],
    ) == sorted(expected, key=lambda d: d["name"])

    # check when removing one dependency
    catalog.remove_dataset(dogs_dataset.name, force=True)
    expected[0] = None
    expected[1]["dependencies"] = []

    assert sorted(
        (
            dataset_dependency_asdict(d)
            for d in catalog.get_dataset_dependencies(ds_name, "1.0.0")
        ),
        key=lambda d: d["name"] if d else "",
    ) == sorted(expected, key=lambda d: d["name"] if d else "")

    # check when removing the other dependency
    catalog.remove_dataset(cats_dataset.name, force=True)
    assert catalog.get_dataset_dependencies(ds_name, "1.0.0") == [None, None]


def test_dataset_dependencies_multiple_union(
    cloud_test_catalog, dogs_dataset, cats_dataset
):
    ds_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    listing = catalog.listings()[0]
    dep_name, _, _ = parse_listing_uri(cloud_test_catalog.src_uri)

    dogs = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version="1.0.0", catalog=catalog)
    dogs_other = DatasetQuery(name=dogs_dataset.name, version="1.0.0", catalog=catalog)

    dogs.union(cats).union(dogs_other).save(ds_name)

    storage_depenedncy = {
        "id": ANY,
        "type": DatasetDependencyType.STORAGE,
        "name": dep_name,
        "namespace": catalog.metastore.system_namespace_name,
        "project": catalog.metastore.listing_project.name,
        "version": "1.0.0",
        "created_at": listing.created_at,
        "dependencies": [],
    }

    expected = [
        {
            "id": ANY,
            "type": DatasetDependencyType.DATASET,
            "name": dogs_dataset.name,
            "namespace": catalog.metastore.default_namespace_name,
            "project": catalog.metastore.default_project.name,
            "version": "1.0.0",
            "created_at": dogs_dataset.get_version("1.0.0").created_at,
            "dependencies": [storage_depenedncy],
        },
        {
            "id": ANY,
            "type": DatasetDependencyType.DATASET,
            "name": cats_dataset.name,
            "namespace": catalog.metastore.default_namespace_name,
            "project": catalog.metastore.default_project.name,
            "version": "1.0.0",
            "created_at": cats_dataset.get_version("1.0.0").created_at,
            "dependencies": [storage_depenedncy],
        },
    ]

    assert sorted(
        (
            dataset_dependency_asdict(d)
            for d in catalog.get_dataset_dependencies(ds_name, "1.0.0", indirect=True)
        ),
        key=lambda d: d["name"],
    ) == sorted(expected, key=lambda d: d["name"])


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_save_subset_of_columns(cloud_test_catalog, cats_dataset):
    catalog = cloud_test_catalog.catalog
    DatasetQuery(cats_dataset.name, catalog=catalog).select(C("file.path")).save(
        "cats", version="1.0.0"
    )

    dataset = catalog.get_dataset("cats")
    assert dataset.schema == {"file__path": String}
