import math
from datetime import datetime, timezone
from typing import Any

import pytest
import sqlalchemy

from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    DateTime,
    Float,
    Float32,
    Float64,
    Int,
    Int32,
    Int64,
    String,
    UInt32,
    UInt64,
)
from tests.utils import (
    DEFAULT_TREE,
    TARRED_TREE,
)

COMPLEX_TREE: dict[str, Any] = {
    **TARRED_TREE,
    **DEFAULT_TREE,
    "nested": {"dir": {"path": {"abc.txt": "abc"}}},
}


@pytest.mark.parametrize(
    "col_type,default_value",
    [
        [String(), ""],
        [Boolean(), False],
        [Int(), 0],
        [Int32(), 0],
        [UInt32(), 0],
        [Int64(), 0],
        [UInt64(), 0],
        [Float(), lambda val: math.isnan(val)],  # noqa: PLW0108
        [Float32(), lambda val: math.isnan(val)],  # noqa: PLW0108
        [Float64(), lambda val: math.isnan(val)],  # noqa: PLW0108
        [Array(Int), []],
        [JSON(), {}],
        [DateTime(), datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)],
        [Binary(), b""],
    ],
)
def test_db_defaults(col_type, default_value, catalog):
    warehouse = catalog.warehouse

    table_col = sqlalchemy.Column(
        "val",
        col_type,
        nullable=False,
        server_default=col_type.db_default_value(warehouse.db.dialect),
    )
    table = warehouse.create_udf_table([table_col])
    warehouse.insert_rows(table, [{"sys__id": 1}])
    warehouse.insert_rows_done(table)

    query = sqlalchemy.Select(table_col).select_from(table)

    values = [row[0] for row in warehouse.dataset_rows_select(query)]
    assert len(values) == 1
    if callable(default_value):
        assert default_value(values[0])
    else:
        assert values[0] == default_value

    warehouse.db.drop_table(table)


def test_get_table_missing(catalog):
    from datachain.error import TableMissingError

    with pytest.raises(TableMissingError, match="not found"):
        catalog.warehouse.db.get_table("nonexistent_table_12345")


def test_list_tables(catalog):
    db = catalog.warehouse.db
    tables = db.list_tables()
    assert isinstance(tables, list)

    # Create a test table
    table = catalog.warehouse.create_udf_table([], name="test_list_tables_abc")
    try:
        tables_after = db.list_tables()
        assert "test_list_tables_abc" in tables_after

        # Test with prefix filter
        filtered = db.list_tables(prefix="test_list_tables")
        assert "test_list_tables_abc" in filtered
    finally:
        db.drop_table(table)
