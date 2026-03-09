import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect
from sqlalchemy.schema import CreateTable

from datachain.data_storage.schema import DataTable
from datachain.dataset import (
    DatasetDependency,
    DatasetDependencyType,
    DatasetRecord,
    DatasetVersion,
    parse_dataset_name,
    parse_schema,
)
from datachain.error import InvalidDatasetNameError
from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    Float,
    Float32,
    Float64,
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


def test_schema_serialization(dataset_record):
    dataset_record.schema = {"int_col": Int}
    assert dataset_record.serialized_schema == {"int_col": {"type": "Int"}}

    dataset_record.schema = {
        "binary_col": Binary,
        "float_32_col": Float32,
    }
    assert dataset_record.serialized_schema == {
        "binary_col": {"type": "Binary"},
        "float_32_col": {"type": "Float32"},
    }

    dataset_record.schema = {"nested_col": Array(Array(Float64))}
    assert dataset_record.serialized_schema == {
        "nested_col": {
            "type": "Array",
            "item_type": {"type": "Array", "item_type": {"type": "Float64"}},
        }
    }


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
