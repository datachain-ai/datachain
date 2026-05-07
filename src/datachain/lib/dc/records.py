import hashlib
import json as _json
from collections.abc import Iterable
from typing import TYPE_CHECKING

import sqlalchemy
from pydantic import BaseModel

from datachain.lib.convert.flatten import flatten
from datachain.lib.data_model import DataType
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalSchema
from datachain.query import Session

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


def _flatten_record(record: dict, signal_schema: SignalSchema) -> dict:
    """Converts nested DataModel objects like {"person": Person(...)} into flattened
    dictionaries like {"person__name": "Alice", "person__age": 30, ...}.
    """
    flattened = {}

    for key, value in record.items():
        if isinstance(value, BaseModel) and ModelStore.is_pydantic(type(value)):
            db_columns = signal_schema.db_signals(name=key)
            flat_values = flatten(value)
            flattened.update(dict(zip(db_columns, flat_values, strict=True)))
        else:
            flattened[key] = value

    return flattened


def _content_hash(
    records: list[dict] | tuple[dict, ...] | dict,
    signal_schema: SignalSchema,
) -> str:
    """Compute a deterministic content hash for a concrete batch of records.

    Stored as the temp-dataset version's `content_hash` so chains starting
    from materialized records (read_records, single-file read_storage)
    produce the same chain hash on identical inputs across runs — a
    precondition for checkpoint reuse.
    """
    items: Iterable[dict] = [records] if isinstance(records, dict) else records
    # XOR-combine per-record digests so order doesn't matter and we can
    # stream records without buffering or sorting.
    combined = 0
    for rec in items:
        payload = _json.dumps(
            _flatten_record(rec, signal_schema), sort_keys=True, default=str
        )
        digest = hashlib.sha256(payload.encode()).digest()
        combined ^= int.from_bytes(digest, "big")
    h = hashlib.sha256()
    h.update(signal_schema.hash().encode())
    h.update(b"\0")
    h.update(combined.to_bytes(32, "big"))
    return h.hexdigest()


def _read_records_with_hash(
    to_insert: dict | Iterable[dict] | None,
    schema: dict[str, DataType],
    content_hash: str | None,
    session: Session | None = None,
    settings: dict | None = None,
    in_memory: bool = False,
) -> "DataChain":
    """Internal: temp-dataset creation with caller-supplied content_hash.

    Pass ``content_hash=None`` to skip content-derived hashing for this temp
    dataset (the version's UUID will be the only identity anchor).
    """
    from datachain.query.dataset import adjust_outputs, get_col_types
    from datachain.sql.types import SQLType

    from .datasets import read_dataset

    session = Session.get(session, in_memory=in_memory)
    catalog = session.catalog

    name = session.generate_temp_dataset_name()
    signal_schema = SignalSchema(schema)
    columns = [
        sqlalchemy.Column(c.name, c.type)  # type: ignore[union-attr]
        for c in signal_schema.db_signals(as_columns=True)
    ]

    if isinstance(to_insert, dict):
        to_insert = [to_insert]
    elif not to_insert:
        to_insert = []

    dsr = catalog.create_dataset(
        name,
        catalog.metastore.default_project,
        columns=columns,
        feature_schema=signal_schema.clone_without_sys_signals().serialize(),
        content_hash=content_hash,
    )

    warehouse = catalog.warehouse

    # Create the rows table (create_dataset only creates metadata).
    table_name = warehouse.dataset_table_name(dsr, dsr.latest_version)
    warehouse.create_dataset_rows_table(table_name, columns=columns)

    dr = warehouse.dataset_rows(dsr)
    table = dr.get_table()

    # Optimization: Compute row types once, rather than for every row.
    col_types = get_col_types(
        warehouse,
        {c.name: c.type for c in columns if isinstance(c.type, SQLType)},
    )

    flattened_records = (_flatten_record(record, signal_schema) for record in to_insert)
    records = (
        adjust_outputs(warehouse, record, col_types, signal_schema)
        for record in flattened_records
    )
    warehouse.insert_rows(table, records)
    warehouse.insert_rows_done(table)

    # Finalize warehouse-derived metadata before marking the version COMPLETE.
    catalog.complete_dataset_version(dsr, dsr.latest_version)

    return read_dataset(name=dsr.full_name, session=session, settings=settings)


def read_records(
    to_insert: dict | Iterable[dict] | None,
    schema: dict[str, DataType],
    session: Session | None = None,
    settings: dict | None = None,
    in_memory: bool = False,
) -> "DataChain":
    """Create a DataChain from the provided records. This is a low-level function
    that directly inserts records into the database. Unlike convenience functions
    like `read_values()` or `read_csv()`, you have to provide the schema and records
    explicitly.

    Compare it with `read_values()` which infers schema automatically and is using
    higher-level abstractions which makes it less efficient. E.g. `read_values()` cannot
    handle large datasets efficiently since it needs to load all data into memory.

    Parameters:
        to_insert: records to insert (empty list / None to create an empty chain). Can
                    be a list, iterator, or generator. Iterators are processed lazily
                    without loading all records into memory at once.

                    Each record must be a dictionary with keys matching the schema.
                    Dictionary values can be:
                    - Primitive types (str, int, etc.)
                    - DataModel objects (automatically flattened to match schema)
                    - Raw flattened data (e.g., {"person__name": "Alice", ...})
        schema: describes chain signals and their corresponding types.

    Example:
        ```py
        import datachain as dc
        from datachain import DataModel

        # Simple records with primitive types
        records = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        chain = dc.read_records(records, schema={"name": str, "age": int})

        # Complex records with DataModel objects (automatically flattened)
        class Person(DataModel):
            name: str
            age: int
            city: str

        people = [
            Person(name="Alice", age=30, city="NYC"),
            Person(name="Bob", age=25, city="LA"),
        ]
        records = [{"person": p} for p in people]
        chain = dc.read_records(records, schema={"person": Person})

        # Raw pre-flattened data (also works)
        records = [
            {"person__name": "Alice", "person__age": 30, "person__city": "NYC"},
            {"person__name": "Bob", "person__age": 25, "person__city": "LA"},
        ]
        chain = dc.read_records(records, schema={"person": Person})

        # Using an iterator/generator for memory efficiency
        def generate_records():
            for i in range(1000000):
                yield {"id": i, "value": i * 2}

        chain = dc.read_records(generate_records(), schema={"id": int, "value": int})
        ```

    Notes:
        This call blocks until all records are inserted, but iterators are processed
        in batches to avoid loading all data into memory at once.
    """
    if isinstance(to_insert, dict):
        normalized: dict | Iterable[dict] | None = [to_insert]
    elif not to_insert:
        normalized = []
    else:
        normalized = to_insert

    signal_schema = SignalSchema(schema)
    content_hash = (
        _content_hash(normalized, signal_schema)
        if isinstance(normalized, (list, tuple))
        else None
    )

    return _read_records_with_hash(
        normalized,
        schema,
        content_hash,
        session=session,
        settings=settings,
        in_memory=in_memory,
    )
