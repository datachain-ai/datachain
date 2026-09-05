from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest
from pydantic import field_serializer, field_validator, model_serializer

import datachain as dc
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.utils import DataChainParamsError

DT_0 = datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
DT_1 = datetime(2024, 2, 3, 4, 5, 6, 654321, tzinfo=timezone.utc)
DT_2 = datetime(2024, 3, 4, 5, 6, 7, tzinfo=timezone.utc)


class CodecChild(dc.DataModel):
    label: str
    observed_at: datetime
    payload: bytes

    @model_serializer(mode="plain")
    def export_child(self) -> dict[str, str]:
        # This one-way export shape is not the durable field-state representation.
        return {"export_label": self.label}


class CodecHolder(dc.DataModel):
    ints: list[int | None]
    floats: list[float | None]
    windows: list[list[datetime] | None]
    children: list[CodecChild | None]
    attributes: dict[str, bytes | None]
    observed_at: datetime
    payload: bytes


class SnapshotHolder(dc.DataModel):
    value: int

    @field_validator("value")
    @classmethod
    def normalize_value(cls, value: int) -> int:
        # A storage read must not run validation over already validated state.
        return value + 10

    @field_serializer("value")
    def export_value(self, value: int) -> str:
        # Pydantic export behavior is deliberately not DataChain's storage format.
        return f"export:{value}"


CHILD_0 = CodecChild(label="zero", observed_at=DT_0, payload=b"\x00zero\xff")
CHILD_1 = CodecChild(label="one", observed_at=DT_1, payload=b"one")


@pytest.fixture(autouse=True)
def enable_experimental_typed_codec(monkeypatch):
    monkeypatch.setenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", "1")


def make_codec_holder(row_id: int) -> CodecHolder:
    if row_id == 0:
        return CodecHolder(
            ints=[None, 1, None],
            floats=[None, 1.25, None],
            # The annotation permits None, but this value intentionally has none.
            windows=[[DT_0, DT_1], [DT_2]],
            children=[CHILD_0, None, CHILD_1],
            attributes={"raw": b"\x00\xff", "missing": None},
            observed_at=DT_0,
            payload=b"\x00holder-zero\xff",
        )
    return CodecHolder(
        ints=[None, None],
        floats=[None, None],
        windows=[None, [DT_2], None],
        children=[None, CHILD_1, None],
        attributes={},
        observed_at=DT_1,
        payload=b"holder-one",
    )


def make_snapshot_holder(row_id: int) -> SnapshotHolder:
    return SnapshotHolder(value=row_id)


def summarize_holder(holder: CodecHolder) -> str:
    child = next(child for child in holder.children if child is not None)
    assert isinstance(holder.observed_at, datetime)
    assert isinstance(holder.payload, bytes)
    assert isinstance(child, CodecChild)
    assert isinstance(child.observed_at, datetime)
    assert isinstance(child.payload, bytes)
    return f"{child.label}:{holder.observed_at.isoformat()}:{holder.payload.hex()}"


def sum_window_years(windows: list[list[datetime] | None]) -> int:
    return sum(value.year for group in windows if group for value in group)


def _collection_chain(test_session, writer: str, annotation: Any, values: list[Any]):
    row_ids = list(range(len(values)))

    if writer == "read_records":
        return dc.read_records(
            [
                {"row_id": row_id, "value": value}
                for row_id, value in zip(row_ids, values, strict=True)
            ],
            schema={"row_id": int, "value": annotation},
            session=test_session,
        )

    if writer == "read_values":
        return dc.read_values(
            row_id=row_ids,
            value=values,
            output={"row_id": int, "value": annotation},
            session=test_session,
        )

    if writer == "map":
        by_id = dict(zip(row_ids, values, strict=True))

        def emit(row_id: int):
            return by_id[row_id]

        return dc.read_values(row_id=row_ids, session=test_session).map(
            value=emit, output=annotation
        )

    raise AssertionError(f"unknown writer: {writer}")


def _save_and_reload(chain, test_session, prefix: str):
    name = f"{prefix}_{uuid4().hex}"
    chain.save(name)
    return name, dc.read_dataset(name, session=test_session)


def _without_physical_codec(value):
    if isinstance(value, dict):
        return {
            key: _without_physical_codec(item)
            for key, item in value.items()
            if key != "dc_codec"
        }
    if isinstance(value, list):
        return [_without_physical_codec(item) for item in value]
    return value


@pytest.mark.parametrize(
    ("annotation", "values"),
    [
        pytest.param(
            list[int | None],
            [[], [None], [None, 1], [1, None], [None, None], [1, 2]],
            id="int",
        ),
        pytest.param(
            list[float | None],
            [
                [],
                [None],
                [None, 1.25],
                [1.25, None],
                [None, None],
                [1.25, 2.5],
            ],
            id="float",
        ),
        pytest.param(
            tuple[int | None, ...],
            [(), (None,), (None, 1), (1, None), (None, None), (1, 2)],
            id="variadic-tuple-int",
        ),
    ],
)
def test_nullable_numeric_array_write_path_parity(test_session, annotation, values):
    """Declared element nullability, not a value scan, controls every writer."""
    for writer in ("read_records", "read_values", "map"):
        chain = _collection_chain(test_session, writer, annotation, values)
        assert chain.order_by("row_id").to_values("value") == values

        name, reloaded = _save_and_reload(
            chain, test_session, f"codec_nullable_{writer}"
        )
        assert reloaded.order_by("row_id").to_values("value") == values

        dataset = test_session.catalog.get_dataset(name, versions=None)
        version = dataset.get_version("1.0.0")
        feature_schema = version.feature_schema
        assert feature_schema is not None
        assert feature_schema["_storage_codecs"]["value"] == "typed-v1"
        physical_type = version.schema["value"]
        physical_type = (
            physical_type() if isinstance(physical_type, type) else physical_type
        )
        assert physical_type.dc_codec == "typed-v1"


@pytest.mark.parametrize("writer", ["read_records", "read_values", "map"])
def test_nested_nullable_datetime_round_trip(writer, test_session):
    """A nullable nested annotation must not turn datetimes into strings.

    The first row is important: it has a nullable nested type, but its value contains
    no None. The codec must therefore be chosen from the annotation, not the data.
    """
    values = [
        [[DT_0, DT_1], [DT_2]],
        [None, [DT_2], None],
        [None, None],
        [],
    ]
    chain = _collection_chain(test_session, writer, list[list[datetime] | None], values)
    _, reloaded = _save_and_reload(chain, test_session, f"codec_datetime_{writer}")

    actual = reloaded.order_by("row_id").to_values("value")
    assert actual == values
    assert isinstance(actual[0][0][0], datetime)
    assert isinstance(actual[1][1][0], datetime)


@pytest.mark.parametrize("writer", ["read_records", "read_values", "map"])
def test_optional_collection_distinguishes_none_from_empty(writer, test_session):
    values = [None, [], [1, 2]]
    chain = _collection_chain(test_session, writer, list[int] | None, values)
    _, reloaded = _save_and_reload(chain, test_session, f"codec_optional_{writer}")

    assert reloaded.order_by("row_id").to_values("value") == values


@pytest.mark.parametrize(
    ("annotation", "value"),
    [
        pytest.param(list[int | None], [None, 1, None], id="nullable-list"),
        pytest.param(
            tuple[int | None, ...], (None, 1, None), id="nullable-variadic-tuple"
        ),
        pytest.param(
            list[list[datetime] | None],
            [None, [DT_0, DT_1], None],
            id="nested-datetime",
        ),
        pytest.param(
            list[CodecChild | None],
            [CHILD_0, None, CHILD_1],
            id="optional-model",
        ),
    ],
)
def test_read_values_single_signal_preserves_collection_as_one_value(
    annotation, value, test_session
):
    """A tuple/list value is one output, not a multi-output row wrapper."""
    chain = dc.read_values(
        value=[value], output={"value": annotation}, session=test_session
    )
    assert chain.to_values("value") == [value]

    _, reloaded = _save_and_reload(chain, test_session, "codec_single_signal")
    assert reloaded.to_values("value") == [value]


@pytest.mark.parametrize("writer", ["read_records", "read_values", "map"])
def test_nullable_model_array_write_path_parity(writer, test_session):
    values = [[CHILD_0, None, CHILD_1], [None, None], []]
    assert CHILD_0.model_dump() == {"export_label": "zero"}
    chain = _collection_chain(test_session, writer, list[CodecChild | None], values)
    _, reloaded = _save_and_reload(chain, test_session, f"codec_models_{writer}")

    actual = reloaded.order_by("row_id").to_values("value")
    assert actual == values
    assert isinstance(actual[0][0], CodecChild)
    assert actual[0][0].observed_at == DT_0
    assert actual[0][0].payload == b"\x00zero\xff"
    assert actual[0][1] is None


def test_model_array_validates_mapping_input_before_snapshot(test_session):
    raw_child = {
        "label": "mapping",
        "observed_at": DT_2,
        "payload": b"mapping-payload",
    }
    expected = CodecChild(**raw_child)
    chain = dc.read_values(
        value=[[raw_child, None]],
        output={"value": list[CodecChild | None]},
        session=test_session,
    )
    _, reloaded = _save_and_reload(chain, test_session, "codec_mapping_model")

    assert reloaded.to_values("value") == [[expected, None]]


def test_udf_model_full_and_leaf_reads_use_the_same_codec(test_session):
    chain = dc.read_values(row_id=[0, 1], session=test_session).map(
        holder=make_codec_holder
    )
    _, reloaded = _save_and_reload(chain, test_session, "codec_holder")
    ordered = reloaded.order_by("row_id")

    holders = ordered.to_values("holder")
    assert holders == [make_codec_holder(0), make_codec_holder(1)]

    for field_name in CodecHolder.model_fields:
        leaf_values = ordered.to_values(f"holder.{field_name}")
        assert leaf_values == [getattr(holder, field_name) for holder in holders]

    assert isinstance(holders[0].observed_at, datetime)
    assert isinstance(holders[0].payload, bytes)
    assert isinstance(holders[0].windows[0][0], datetime)
    assert isinstance(holders[0].children[0], CodecChild)
    assert isinstance(holders[0].children[0].observed_at, datetime)
    assert isinstance(holders[0].children[0].payload, bytes)
    assert isinstance(holders[0].attributes["raw"], bytes)


def test_typed_dataset_decodes_full_and_leaf_udf_parameters(test_session):
    chain = dc.read_values(row_id=[0, 1], session=test_session).map(
        holder=make_codec_holder
    )
    _, reloaded = _save_and_reload(chain, test_session, "codec_udf_params")
    ordered = reloaded.order_by("row_id")

    summaries = ordered.map(summary=summarize_holder).order_by("row_id")
    assert summaries.to_values("summary") == [
        summarize_holder(make_codec_holder(0)),
        summarize_holder(make_codec_holder(1)),
    ]

    leaf_params = ordered.map(
        year_sum=sum_window_years,
        params=["holder.windows"],
        output=int,
    ).order_by("row_id")
    assert leaf_params.to_values("year_sum") == [
        DT_0.year + DT_1.year + DT_2.year,
        DT_2.year,
    ]


def test_model_storage_snapshots_validated_fields_not_export_form(test_session):
    source = make_snapshot_holder(1)
    assert source.value == 11
    assert source.model_dump() == {"value": "export:11"}

    chain = dc.read_values(row_id=[1], session=test_session).map(
        holder=make_snapshot_holder
    )
    _, reloaded = _save_and_reload(chain, test_session, "codec_snapshot")

    full = reloaded.to_values("holder")[0]
    leaf = reloaded.to_values("holder.value")[0]
    assert isinstance(full, SnapshotHolder)
    assert full.value == 11
    assert leaf == 11


def test_storage_codec_metadata_round_trip_and_legacy_default():
    schema = SignalSchema({"plain": int, "value": list[int | None]})
    stored = schema.serialize()

    assert schema.storage_codecs == {"value": "typed-v1"}
    assert stored["_storage_codecs"] == {"value": "typed-v1"}
    restored = SignalSchema.deserialize(deepcopy(stored))
    assert restored.storage_codecs == {"value": "typed-v1"}
    assert restored.column_codec("value") is not None

    legacy_stored = deepcopy(stored)
    legacy_stored.pop("_storage_codecs")
    legacy = SignalSchema.deserialize(legacy_stored)
    assert legacy.storage_codecs == {}
    assert legacy.column_codec("value") is None


def test_saved_dataset_without_codec_metadata_uses_legacy_reader(test_session):
    name, _ = _save_and_reload(
        dc.read_values(
            row_id=[0, 1],
            value=[[1, 2], []],
            output={"row_id": int, "value": list[int]},
            session=test_session,
        ),
        test_session,
        "codec_legacy",
    )

    metastore = test_session.catalog.metastore
    dataset = metastore.get_dataset(name, versions=None)
    version = dataset.get_version("1.0.0")

    legacy_feature_schema = deepcopy(version.feature_schema)
    legacy_feature_schema.pop("_storage_codecs")
    legacy_physical_schema = {
        column: _without_physical_codec(
            (sql_type() if isinstance(sql_type, type) else sql_type).to_dict()
        )
        for column, sql_type in version.schema.items()
    }
    metastore.update_dataset_version(
        dataset,
        "1.0.0",
        feature_schema=legacy_feature_schema,
        schema=legacy_physical_schema,
    )

    legacy = dc.read_dataset(name, session=test_session).order_by("row_id")
    assert legacy.signals_schema.storage_codecs == {}
    assert legacy.to_values("value") == [[1, 2], []]


def test_saved_codec_metadata_controls_reads_when_flag_is_off(
    test_session, monkeypatch
):
    value = [None, [DT_0, DT_1], None]
    name, _ = _save_and_reload(
        dc.read_values(
            value=[value],
            output={"value": list[list[datetime] | None]},
            session=test_session,
        ),
        test_session,
        "codec_flag_off_read",
    )

    monkeypatch.delenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC")
    assert SignalSchema({"new_value": list[int]}).storage_codecs == {}
    restored = dc.read_dataset(name, session=test_session)

    assert restored.signals_schema.storage_codecs == {"value": "typed-v1"}
    assert restored.to_values("value") == [value]
    assert isinstance(restored.to_values("value")[0][1][0], datetime)


def test_union_refuses_mixed_storage_codec_versions(test_session, monkeypatch):
    typed = dc.read_values(
        value=[[1, 2]],
        output={"value": list[int]},
        session=test_session,
    )
    assert typed.signals_schema.storage_codecs == {"value": "typed-v1"}

    monkeypatch.delenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC")
    legacy = dc.read_values(
        value=[[1, 2]],
        output={"value": list[int]},
        session=test_session,
    )
    assert legacy.signals_schema.storage_codecs == {}

    with pytest.raises(DataChainParamsError, match="different storage codecs"):
        typed.union(legacy)
