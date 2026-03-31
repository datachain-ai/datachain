import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect
from sqlalchemy.schema import CreateTable

from datachain.data_storage.schema import DataTable
from datachain.dataset import (
    DatasetDependency,
    DatasetDependencyType,
    DatasetListRecord,
    DatasetRecord,
    DatasetVersion,
    parse_dataset_name,
    parse_dataset_uri,
    parse_schema,
)
from datachain.error import (
    DatasetStateNotLoadedError,
    DatasetVersionNotFoundError,
    InvalidDatasetNameError,
)
from datachain.sql.types import (
    JSON,
    Array,
    Boolean,
    Float,
    Float32,
    Int,
    Int64,
    String,
)
from datachain.utils import DatasetIdentifier


def test_dataset_table_compilation():
    table = DataTable.new_table(
        "ds-1",
        columns=[
            Column("dir_type", Int, index=True),
            Column("path", String, nullable=False, index=True),
            Column("etag", String),
            Column("version", String),
            Column("is_latest", Boolean),
            Column("last_modified", DateTime(timezone=True)),
            Column("size", Int64, nullable=False, index=True),
            Column("location", JSON),
            Column("source", String, nullable=False),
            Column("score", Float, nullable=False),
            Column("meta_info", JSON),
        ],
    )
    result = CreateTable(table, if_not_exists=True).compile(dialect=sqlite_dialect())

    assert result.string == (
        "\n"
        'CREATE TABLE IF NOT EXISTS "ds-1" (\n'
        "\tsys__id INTEGER NOT NULL, \n"
        "\tsys__rand INTEGER DEFAULT (abs(random())) NOT NULL, \n"
        "\tdir_type INTEGER, \n"
        "\tpath VARCHAR NOT NULL, \n"
        "\tetag VARCHAR, \n"
        "\tversion VARCHAR, \n"
        "\tis_latest BOOLEAN, \n"
        "\tlast_modified DATETIME, \n"
        "\tsize INTEGER NOT NULL, \n"
        "\tlocation JSON, \n"
        "\tsource VARCHAR NOT NULL, \n"
        "\tscore FLOAT NOT NULL, \n"
        "\tmeta_info JSON, \n"
        "\tPRIMARY KEY (sys__id)\n"
        ")\n"
        "\n"
    )


@pytest.mark.parametrize(
    "dep_name,dep_type,expected",
    [
        ("dogs_dataset", DatasetDependencyType.DATASET, "dogs_dataset"),
        (
            "s3://dogs_dataset/dogs",
            DatasetDependencyType.STORAGE,
            "lst__s3://dogs_dataset/dogs/",
        ),
    ],
)
def test_dataset_dependency_dataset_name(dep_name, dep_type, expected):
    dep = DatasetDependency(
        id=1,
        name=dep_name,
        version="1.0.0",
        type=dep_type,
        created_at=datetime.now(timezone.utc),
        namespace="dev",
        project="animals",
        dependencies=[],
    )

    assert dep.dataset_name == expected


@pytest.mark.parametrize(
    "use_string",
    [True, False],
)
def test_dataset_version_from_dict(use_string):
    # use_string=True covers the double-encoded JSON string from the metastore
    preview = [{"id": 1, "thing": "a"}, {"id": 2, "thing": "b"}]

    preview_data = json.dumps(preview) if use_string else preview

    data = {
        "id": 1,
        "uuid": "98928be4-b6e8-4b7b-a7c5-2ce3b33130d8",
        "dataset_id": 40,
        "version": "2.0.0",
        "status": 1,
        "feature_schema": {},
        "created_at": datetime.fromisoformat("2023-10-01T12:00:00"),
        "finished_at": None,
        "error_message": "",
        "error_stack": "",
        "script_output": "",
        "schema": {},
        "num_objects": 100,
        "size": 1000000,
        "preview": preview_data,
    }

    dataset_version = DatasetVersion.from_dict(data)
    assert dataset_version.preview == preview


@pytest.mark.parametrize(
    "name,ok",
    [
        ("dogs", True),
        ("test.parquet", False),
        ("simple.test.parquet", False),
    ],
)
def test_validate_name(name, ok):
    if ok:
        DatasetRecord.validate_name(name)
    else:
        with pytest.raises(InvalidDatasetNameError):
            DatasetRecord.validate_name(name)


@pytest.mark.parametrize(
    "full_name,namespace,project,name",
    [
        ("dogs", None, None, "dogs"),
        ("animals.dogs", None, "animals", "dogs"),
        ("dev.animals.dogs", "dev", "animals", "dogs"),
    ],
)
def test_parse_dataset_name(full_name, namespace, project, name):
    assert parse_dataset_name(full_name) == (namespace, project, name)


def test_parse_dataset_name_empty_name():
    with pytest.raises(InvalidDatasetNameError):
        assert parse_dataset_name(None)


@pytest.mark.parametrize(
    "uri,namespace,project,name,version",
    [
        ("ds://result", None, None, "result", None),
        ("ds://result@v1.0.5", None, None, "result", "1.0.5"),
        ("ds://dev.result", None, "dev", "result", None),
        ("ds://dev.result@v1.0.5", None, "dev", "result", "1.0.5"),
        ("ds://global.dev.result", "global", "dev", "result", None),
        ("ds://global.dev.result@v1.0.5", "global", "dev", "result", "1.0.5"),
        ("ds://@ilongin.dev.result", "@ilongin", "dev", "result", None),
        ("ds://@ilongin.dev.result@v1.0.4", "@ilongin", "dev", "result", "1.0.4"),
        ("ds://@vlad.dev.result", "@vlad", "dev", "result", None),
        ("ds://@vlad.dev.result@v1.0.5", "@vlad", "dev", "result", "1.0.5"),
        ("ds://@vlad.@vlad.result@v1.0.5", "@vlad", "@vlad", "result", "1.0.5"),
        ("ds://@vlad.@vlad.@vlad@v1.0.5", "@vlad", "@vlad", "@vlad", "1.0.5"),
    ],
)
def test_parse_dataset_uri(uri, namespace, project, name, version):
    assert parse_dataset_uri(uri) == (namespace, project, name, version)


@pytest.mark.parametrize(
    "uri",
    [
        "http://result",
        "s3://result",
        "",
        "result",
        "ds://",
    ],
)
def test_parse_dataset_uri_invalid(uri):
    with pytest.raises(ValueError):
        parse_dataset_uri(uri)


@pytest.mark.parametrize(
    "dataset_name,expected_namespace,expected_project,expected_name,expected_version",
    [
        ("@namespace.project.dataset", "@namespace", "project", "dataset", None),
        (
            "@namespace.project.dataset@1.0.0",
            "@namespace",
            "project",
            "dataset",
            "1.0.0",
        ),
        ("dataset@1.0.0", None, None, "dataset", "1.0.0"),
        ("dataset", None, None, "dataset", None),
    ],
)
def test_catalog_parse_dataset_name(
    catalog,
    dataset_name,
    expected_namespace,
    expected_project,
    expected_name,
    expected_version,
):
    result = catalog.parse_dataset_name(dataset_name)

    # Get default namespace and project from catalog if not specified
    if expected_namespace is None:
        expected_namespace = catalog.metastore.default_namespace_name
    if expected_project is None:
        expected_project = catalog.metastore.default_project_name

    assert isinstance(result, DatasetIdentifier)
    assert result.namespace == expected_namespace
    assert result.project == expected_project
    assert result.name == expected_name
    assert result.version == expected_version


def test_parse_dataset_schema():
    schema_dict = {
        "id": {"type": "Int"},
        "name": {"type": "String"},
        "scores": {"type": "Array", "item_type": {"type": "Float32"}},
    }

    parsed_schema = parse_schema(schema_dict)
    assert list(parsed_schema.keys()) == ["id", "name", "scores"]
    assert parsed_schema["id"] == Int
    assert parsed_schema["name"] == String
    assert isinstance(parsed_schema["scores"], Array)
    assert isinstance(parsed_schema["scores"].item_type, Float32)


def test_parse_empty_dataset_schema():
    parsed_schema = parse_schema({})
    assert parsed_schema == {}


@pytest.mark.parametrize(
    "schema_dict,exc,match_error",
    [
        (
            "String",
            TypeError,
            "Schema definition must be a dictionary",
        ),
        ({"id": "Int64"}, TypeError, "Schema column 'id' type must be a dictionary"),
        ({"id": {}}, ValueError, "Schema column 'id' type is not defined"),
        (
            {"id": {"type": "UnknownType"}},
            ValueError,
            "Schema column 'id' type 'UnknownType' is not supported",
        ),
        (
            {"values": {"type": "Array"}},
            ValueError,
            (
                "Schema column 'values' type 'Array' parsing error:"
                " Array type must have 'item_type' field"
            ),
        ),
        (
            {"values": {"type": "Array", "item_type": "Foo"}},
            ValueError,
            (
                "Schema column 'values' type 'Array' parsing error:"
                " Array 'item_type' field must be a dictionary"
            ),
        ),
        (
            {"values": {"type": "Array", "item_type": {"foo": "Bar"}}},
            ValueError,
            (
                "Schema column 'values' type 'Array' parsing error:"
                " Array 'item_type' must have 'type' field"
            ),
        ),
        (
            {"values": {"type": "Array", "item_type": {"type": "UnknownType"}}},
            ValueError,
            (
                "Schema column 'values' type 'Array' parsing error:"
                " Array item type 'UnknownType' is not supported"
            ),
        ),
        (
            {"data": {"type": "Array", "item_type": {"type": "Array"}}},
            ValueError,
            (
                "Schema column 'data' type 'Array' parsing error:"
                " Array type must have 'item_type' field"
            ),
        ),
    ],
)
def test_parse_invalid_dataset_schema(schema_dict, exc, match_error):
    with pytest.raises(exc, match=match_error):
        parse_schema(schema_dict)


def test_dataset_record_roundtrip_versions_loaded_true(dataset_record):
    # True is omitted from the dict to avoid crashing old clients
    d = dataset_record.to_dict()
    assert "_versions_loaded" not in d
    restored = DatasetRecord.from_dict(d)
    assert restored._versions_loaded is True
    assert restored.versions[0].version == "1.0.0"


def test_dataset_record_roundtrip_versions_loaded_false(dataset_record):
    record = replace(dataset_record, _versions=[], _versions_loaded=False)
    d = record.to_dict()
    assert d["_versions_loaded"] is False
    restored = DatasetRecord.from_dict(d)
    assert restored._versions_loaded is False


def test_dataset_version_roundtrip_preview_loaded_true(dataset_record):
    # True is omitted from the dict to avoid crashing old clients
    version = replace(
        dataset_record.versions[0], _preview_data=[{"a": 1}], _preview_loaded=True
    )
    d = version.to_dict()
    assert "_preview_loaded" not in d
    restored = DatasetVersion.from_dict(d)
    assert restored._preview_loaded is True
    assert restored.preview == [{"a": 1}]


def test_dataset_version_roundtrip_preview_loaded_false(dataset_record):
    version = replace(dataset_record.versions[0], _preview_loaded=False)
    d = version.to_dict()
    assert d["_preview_loaded"] is False
    restored = DatasetVersion.from_dict(d)
    assert restored._preview_loaded is False


def test_dataset_version_from_dict_rejects_internal_preview_key(dataset_record):
    version = replace(dataset_record.versions[0], _preview_data=[{"a": 1}])
    d = version.to_dict()
    d["_preview_data"] = d.pop("preview")

    with pytest.raises(ValueError, match="'preview'"):
        DatasetVersion.from_dict(d)


def test_dataset_record_from_dict_rejects_internal_versions_key(dataset_record):
    d = dataset_record.to_dict()
    d["_versions"] = d.pop("versions")

    with pytest.raises(ValueError, match="'versions'"):
        DatasetRecord.from_dict(d)


def test_versions_raises_when_not_loaded(dataset_record):
    record = replace(dataset_record, _versions_loaded=False)
    with pytest.raises(DatasetStateNotLoadedError):
        _ = record.versions


def test_preview_raises_when_not_loaded(dataset_record):
    version = replace(dataset_record.versions[0], _preview_loaded=False)
    with pytest.raises(DatasetStateNotLoadedError):
        _ = version.preview


def test_latest_version_empty_raises(dataset_record):
    record = replace(dataset_record, _versions=[], _versions_loaded=True)
    with pytest.raises(DatasetVersionNotFoundError, match="has no versions"):
        _ = record.latest_version


def test_dataset_list_record_to_dict(dataset_list_record):
    d = dataset_list_record.to_dict()

    assert d["id"] == dataset_list_record.id
    assert d["name"] == dataset_list_record.name
    assert d["project"]["name"] == dataset_list_record.project.name
    assert (
        d["project"]["namespace"]["name"] == dataset_list_record.project.namespace.name
    )
    assert len(d["versions"]) == 2
    assert d["versions"][0]["version"] == "1.0.0"
    assert d["versions"][1]["version"] == "2.0.0"


def test_dataset_list_record_roundtrip(dataset_list_record):
    d = dataset_list_record.to_dict()
    restored = DatasetListRecord.from_dict(d)

    assert restored.id == dataset_list_record.id
    assert restored.name == dataset_list_record.name
    assert restored.description == dataset_list_record.description
    assert restored.attrs == dataset_list_record.attrs
    assert restored.project.name == dataset_list_record.project.name
    assert restored.project.namespace.name == dataset_list_record.project.namespace.name
    assert len(restored.versions) == len(dataset_list_record.versions)
    for orig, rest in zip(dataset_list_record.versions, restored.versions, strict=True):
        assert rest.id == orig.id
        assert rest.version == orig.version
        assert rest.uuid == orig.uuid
        assert rest.status == orig.status


def test_dataset_list_record_has_version_with_uuid(dataset_list_record):
    assert (
        dataset_list_record.has_version_with_uuid(dataset_list_record.versions[0].uuid)
        is True
    )
    assert dataset_list_record.has_version_with_uuid("nonexistent") is False

    record = replace(dataset_list_record, versions=[])
    assert record.has_version_with_uuid("anything") is False
