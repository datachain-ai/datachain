from datetime import datetime

import pytest
from pydantic import BaseModel
from sqlalchemy.dialects.sqlite import dialect

import datachain as dc
from datachain.lib.convert.column_codec import CODEC_VERSION, compile_codec
from datachain.lib.signal_schema import SignalSchema, SignalSchemaError
from datachain.lib.udf import UDFAdapter, UDFBase
from datachain.lib.utils import DataChainParamsError
from datachain.query.schema import Column
from datachain.sql.types import JSON, Array, Int64, SQLType


class CodecSchemaHolder(BaseModel):
    values: list[dict]
    label: str


class FixedHashUDF(UDFBase):
    def hash(self, include_body: bool = True) -> str:
        return ("a" if include_body else "b") * 64


def test_experimental_codec_is_opt_in(monkeypatch):
    monkeypatch.delenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", raising=False)
    assert SignalSchema({"value": list[int]}).storage_codecs == {}

    monkeypatch.setenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", "1")
    assert SignalSchema({"value": list[int]}).storage_codecs == {"value": CODEC_VERSION}


def test_legacy_schema_deserialization_does_not_enable_codec(monkeypatch):
    monkeypatch.setenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", "1")

    schema = SignalSchema.deserialize({"value": "list[dict]"})

    assert schema.storage_codecs == {}
    assert schema.column_codec("value") is None
    assert "_storage_codecs" not in schema.serialize()


def test_storage_metadata_roundtrip_without_experimental_flag(monkeypatch):
    monkeypatch.delenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", raising=False)
    schema = SignalSchema(
        {"typed": list[dict], "legacy": list[dict]},
        storage_codecs={"typed": CODEC_VERSION},
    )

    restored = SignalSchema.deserialize(schema.serialize())

    assert restored.storage_codecs == {"typed": CODEC_VERSION}
    assert restored.column_codec("typed") is not None
    assert restored.column_codec("legacy") is None
    assert restored.hash() == schema.hash()


def test_unknown_codec_version_is_refused():
    with pytest.raises(SignalSchemaError, match="Unknown storage codecs"):
        SignalSchema.deserialize(
            {"value": "list[int]", "_storage_codecs": {"value": "typed-future"}}
        )


def test_projection_keeps_typed_and_legacy_provenance():
    schema = SignalSchema(
        {"holder": CodecSchemaHolder, "legacy": list[dict]},
        storage_codecs={"holder": CODEC_VERSION},
    )

    projected = schema.resolve("holder.values", "legacy")

    assert projected.storage_codecs == {"holder.values": CODEC_VERSION}
    assert projected.column_codec("holder__values") is not None
    assert projected.column_codec("legacy") is None
    partial = schema.to_partial("holder.values")
    assert partial.storage_codecs == {"holder": CODEC_VERSION}
    assert partial.column_codec("holder__values") is not None


def test_rename_and_merge_preserve_per_signal_provenance():
    left = SignalSchema({"items": list[dict]}, storage_codecs={"items": CODEC_VERSION})
    right = SignalSchema({"items": list[dict]}, storage_codecs={})

    renamed = left.mutate({"renamed": Column("items")})
    merged = left.merge(right, "right_", right_nullable=False)

    assert renamed.storage_codecs == {"renamed": CODEC_VERSION}
    assert merged.storage_codecs == {"items": CODEC_VERSION}
    assert merged.column_codec("items") is not None
    assert merged.column_codec("right_items") is None


def test_append_and_schema_union_keep_left_duplicate_provenance():
    left = SignalSchema({"items": list[dict]}, storage_codecs={})
    right = SignalSchema(
        {"items": list[dict], "added": list[dict]},
        storage_codecs={"items": CODEC_VERSION, "added": CODEC_VERSION},
    )

    for combined in (left.append(right), left | right):
        assert combined.storage_codecs == {"added": CODEC_VERSION}
        assert combined.column_codec("items") is None


def test_legacy_column_replacement_clears_target_codec():
    schema = SignalSchema(
        {"typed": list[dict], "legacy": list[dict]},
        storage_codecs={"typed": CODEC_VERSION},
    )

    replaced = schema.mutate({"typed": Column("legacy")})

    assert replaced.storage_codecs == {}
    assert replaced.column_codec("typed") is None


def test_typed_array_physical_metadata_and_read_are_structural():
    codec = compile_codec(list[dict | None])
    assert codec is not None
    sql_type = Array.from_dict(codec.sql_type.to_dict())

    assert sql_type.dc_codec == CODEC_VERSION
    assert sql_type.item_type.dc_nullable
    assert sql_type.to_dict() == codec.sql_type.to_dict()
    assert sql_type.on_read_convert('[null,{"k":1}]', dialect()) == [None, {"k": 1}]
    # JSON-looking strings are still strings, not evidence of an old encoding.
    assert sql_type.on_read_convert('["null","{}"]', dialect()) == ["null", "{}"]


def test_legacy_array_physical_read_remains_unchanged():
    sql_type = Array.from_dict({"type": "Array", "item_type": {"type": "JSON"}})

    assert sql_type.dc_codec is None
    assert sql_type.on_read_convert('["null","{}"]', dialect()) == [None, {}]


def test_nullable_scalar_physical_metadata_roundtrip():
    sql_type = SQLType.as_nullable(Int64())
    sql_type.dc_codec = CODEC_VERSION

    restored = Int64.from_dict(sql_type.to_dict())

    assert isinstance(restored, Int64)
    assert restored.to_dict() == sql_type.to_dict()


@pytest.mark.parametrize("include_body", [True, False])
def test_adapter_fingerprints_codec_and_physical_type_with_custom_hash(include_body):
    udf = FixedHashUDF()
    udf.output = SignalSchema({"value": list[int]}, storage_codecs={})
    legacy = UDFAdapter(udf, {"value": Array(Int64())})
    legacy_hash = legacy.hash(include_body=include_body)
    assert legacy_hash == udf.hash(include_body=include_body)

    udf.output = SignalSchema(
        {"value": list[int]}, storage_codecs={"value": CODEC_VERSION}
    )
    typed = UDFAdapter(udf, {"value": Array(Int64())})
    changed_physical = UDFAdapter(udf, {"value": Array(JSON())})

    assert typed.hash(include_body=include_body) != legacy_hash
    assert typed.hash(include_body=include_body) != changed_physical.hash(
        include_body=include_body
    )


def test_adapter_fingerprint_preserves_physical_column_order():
    udf = FixedHashUDF()
    udf.output = SignalSchema(
        {"one": list[int], "two": list[int]},
        storage_codecs={"one": CODEC_VERSION, "two": CODEC_VERSION},
    )
    first = UDFAdapter(udf, {"one": Array(Int64()), "two": Array(Int64())})
    swapped = UDFAdapter(udf, {"two": Array(Int64()), "one": Array(Int64())})

    assert first.hash() != swapped.hash()


def test_union_refuses_mixed_codec_columns(test_session, monkeypatch):
    monkeypatch.delenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", raising=False)
    legacy = dc.read_records(
        [{"value": [{"k": 1}]}], schema={"value": list[dict]}, session=test_session
    )
    monkeypatch.setenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", "1")
    typed = dc.read_records(
        [{"value": [{"k": 1}]}], schema={"value": list[dict]}, session=test_session
    )

    for left, right in ((legacy, typed), (typed, legacy)):
        with pytest.raises(DataChainParamsError, match="different storage codecs"):
            left.union(right)


@pytest.mark.parametrize("typed", [False, True])
def test_union_promotion_preserves_codec_when_flag_changes(
    test_session, monkeypatch, typed
):
    monkeypatch.setenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", "1" if typed else "0")
    plain = dc.read_records(
        [{"value": [1]}], schema={"value": list[int]}, session=test_session
    )
    optional = dc.read_records(
        [{"value": None}], schema={"value": list[int] | None}, session=test_session
    )
    monkeypatch.setenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", "0" if typed else "1")

    for left, right in ((plain, optional), (optional, plain)):
        combined = left.union(right)
        assert combined.signals_schema.storage_codecs == (
            {"value": CODEC_VERSION} if typed else {}
        )


def test_chain_column_uses_typed_physical_type(test_session, monkeypatch):
    monkeypatch.setenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", "1")
    chain = dc.read_records(
        [{"value": [None]}],
        schema={"value": list[list[datetime] | None]},
        session=test_session,
    )

    sql_type = chain.c("value").type

    assert isinstance(sql_type, Array)
    assert isinstance(sql_type.item_type, JSON)
    assert sql_type.item_type.dc_nullable
    assert sql_type.dc_codec == CODEC_VERSION


def test_mixed_codec_merge_checks_collection_keys_not_scalar_keys(
    test_session, monkeypatch
):
    records = [{"id": 1, "value": [{"k": 1}]}]
    schema = {"id": int, "value": list[dict]}
    monkeypatch.delenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", raising=False)
    legacy = dc.read_records(records, schema=schema, session=test_session)
    monkeypatch.setenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", "1")
    typed = dc.read_records(records, schema=schema, session=test_session)

    with pytest.raises(DataChainParamsError, match="collection keys"):
        legacy.merge(typed, on="value", inner=True)

    combined = legacy.merge(typed, on="id", inner=True)
    assert combined.signals_schema.storage_codecs == {"right_value": CODEC_VERSION}


def test_typed_codec_merge_refuses_unchecked_sql_expression_keys(
    test_session, monkeypatch
):
    monkeypatch.setenv("DATACHAIN_EXPERIMENTAL_TYPED_CODEC", "1")
    chain = dc.read_records(
        [{"id": 1, "value": [1]}],
        schema={"id": int, "value": list[int]},
        session=test_session,
    )

    with pytest.raises(DataChainParamsError, match="direct column keys only"):
        chain.merge(chain, on=Column("id") + 1, right_on="id", inner=True)
