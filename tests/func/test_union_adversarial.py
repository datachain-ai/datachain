"""Adversarial / differential tests for multi-arm ``Union[...]`` support.

Each probe asserts the *logically correct* value, so running this file on SQLite
(reference) and ClickHouse surfaces backend divergences as failures. The probes
mirror every bug class found during the Optional[X] work (PR #1791) — func/aggregate
over an inactive leaf coercing to a type default on CH, NULL propagation through
arithmetic, internal-column leaks into schema/export, NULLS ordering, group-by over
all-absent partitions, parquet/JSON reconstruction — plus union-specific hazards
(arm-order stability across serialization, bool/int and int/float arm matching, a
union fed into a UDF, a File arm, issubclass-on-Union safety).

Run: pytest tests/func/test_union_adversarial.py            # SQLite
     TEST_TARGET=.../test_union_adversarial.py scripts/run-clickhouse-tests.sh
"""

from collections.abc import Iterator
from typing import Literal, Optional, Union

import pandas as pd

import datachain as dc
from datachain import C, func
from datachain.lib.data_model import DataModel
from datachain.lib.file import File


class UFoo(DataModel):
    a: int = 0
    b: str = ""


class UBar(DataModel):
    x: float = 0.0


def _si(test_session, ids, values, *, none=False, name="value"):
    """A chain with an id column and a Union[str,int] (optionally +None) signal."""
    anno = Union[str, int, None] if none else Union[str, int]
    return dc.read_values(
        id=ids,
        **{name: values},
        output={"id": int, name: anno},
        session=test_session,
    )


def _vals(chain, col):
    return [v for _, v in chain.order_by("id").to_list("id", col)]


# ---- func / aggregate over an inactive arm leaf (CH type-default trap) -------


def test_row_func_over_inactive_arm_is_none(test_session):
    # value._1 is the str arm; for int rows it is NULL, so string length is None
    # (must not coerce to 0 on ClickHouse).
    chain = _si(test_session, [1, 2, 3], ["hi", 42, "yoyo"])
    got = _vals(chain.mutate(n=func.string.length("value._1")), "n")
    assert got == [2, None, 4]


def test_arithmetic_over_inactive_arm_is_none(test_session):
    # value._0 is the int arm; NULL + 1 must stay None, not become 1.
    chain = _si(test_session, [1, 2, 3], ["hi", 42, 7])
    got = _vals(chain.mutate(y=C("value._0") + 1), "y")
    assert got == [None, 43, 8]


def test_sum_over_arm_leaf(test_session):
    chain = _si(test_session, [1, 2, 3, 4], ["hi", 42, "yo", 7])
    assert chain.group_by(s=func.sum("value._0")).to_values("s") == [49]


def test_sum_over_all_inactive_partition_is_none(test_session):
    # Group g=0 has only str values -> sum over the int arm is None, not 0 (CH trap).
    chain = dc.read_values(
        id=[1, 2, 3, 4],
        g=[0, 0, 1, 1],
        value=["a", "b", 10, 20],
        output={"id": int, "g": int, "value": Union[str, int]},
        session=test_session,
    )
    rows = chain.group_by(s=func.sum("value._0"), partition_by="g").to_records()
    by_g = {r["g"]: r["s"] for r in rows}
    assert by_g == {0: None, 1: 30}


def test_min_max_over_arm_leaf(test_session):
    chain = _si(test_session, [1, 2, 3, 4], [5, "x", 2, 9])
    rows = chain.group_by(lo=func.min("value._0"), hi=func.max("value._0")).to_records()
    assert (rows[0]["lo"], rows[0]["hi"]) == (2, 9)


# ---- ordering / distinct ----------------------------------------------------


def test_order_by_arm_leaf_nulls_last(test_session):
    chain = _si(test_session, [1, 2, 3, 4], [5, "x", 2, 9])
    # int values ascending, then the str row (NULL int arm) last on both backends.
    assert chain.order_by("value._0").to_values("value") == [2, 5, 9, "x"]


def test_distinct_arm_leaf_values(test_session):
    chain = _si(test_session, [1, 2, 3, 4, 5], [5, "x", 5, "y", 2])
    got = sorted(
        v for v in chain.distinct("value._0").to_values("value") if isinstance(v, int)
    )
    assert got == [2, 5]


# ---- internal-column leaks --------------------------------------------------


def test_print_schema_no_type_tag_leak(test_session):
    chain = _si(test_session, [1], ["hi"])
    assert "_type_tag" not in str(chain.schema)


def test_to_records_no_internal_columns(test_session):
    chain = _si(test_session, [1, 2], ["hi", 42])
    keys = set().union(*(r.keys() for r in chain.to_records()))
    # neither the _type_tag discriminator nor positional arm slots (_0/_1) leak
    assert not any("_type_tag" in k or "_0" in k or "_1" in k for k in keys)
    # arms appear under readable type names instead
    assert {"value__int", "value__str"} <= keys


# ---- JSON / parquet reconstruction -----------------------------------------


def test_to_json_arms_and_none(test_session, tmp_path):
    import json as _json

    path = str(tmp_path / "u.jsonl")
    _si(test_session, [1, 2, 3], ["hi", 42, None], none=True).order_by("id").to_jsonl(
        path
    )
    with open(path) as f:
        rows = sorted(
            (_json.loads(line) for line in f if line.strip()), key=lambda r: r["id"]
        )
    assert [r["value"] for r in rows] == ["hi", 42, None]


def test_parquet_union_in_model_in_union(test_session, tmp_path):
    # A model arm that itself contains a union field, exported and re-read.
    class Inner(DataModel):
        tag: str | int = 0

    class Outer(DataModel):
        inner: Inner = Inner()

    items = [Outer(inner=Inner(tag="x")), Outer(inner=Inner(tag=7))]
    path = str(tmp_path / "n.parquet")
    dc.read_values(
        id=[1, 2],
        o=items,
        output={"id": int, "o": Union[Outer, UBar]},
        session=test_session,
    ).order_by("id").to_parquet(path)
    back = dc.read_parquet(path, session=test_session)
    got = [v for _, v in back.order_by("id").to_list("id", "o")]
    assert [o.inner.tag for o in got] == ["x", 7]


# ---- to_pandas null cells ---------------------------------------------------


def test_to_pandas_inactive_cells_are_null(test_session):
    df = _si(test_session, [1, 2], ["hi", 42]).order_by("id").to_pandas()
    # arm columns are shown by type name (not positional _0); int arm is NULL on
    # the str row.
    assert list(df.columns) == [("id", ""), ("value", "int"), ("value", "str")]
    int_col = df[("value", "int")].tolist()
    assert pd.isna(int_col[0]) and int_col[1] == 42


# ---- union fed into a UDF ---------------------------------------------------


def test_union_value_as_udf_input(test_session):
    chain = _si(test_session, [1, 2, 3], ["hi", 42, "yo"])

    def kind(value) -> str:
        return type(value).__name__

    got = _vals(chain.map(k=kind, output={"k": str}), "k")
    assert got == ["str", "int", "str"]


def test_union_annotated_udf_param(test_session):
    # An annotated Union param exercises SignalSchema.slice type matching.
    chain = _si(test_session, [1, 2, 3], ["hi", 42, "yo"])

    def kind(value: str | int) -> str:
        return type(value).__name__

    assert _vals(chain.map(k=kind, output={"k": str}), "k") == ["str", "int", "str"]


def test_nullable_union_annotated_udf_param(test_session):
    # A nullable Union param must not collapse to its first arm in slice() (that
    # made the UDF input expect a single column instead of the tag + arm slots).
    chain = _si(test_session, [1, 2, 3], ["hi", 42, None], none=True)

    def kind(value: str | int | None) -> str:
        return "none" if value is None else type(value).__name__

    assert _vals(chain.map(k=kind, output={"k": str}), "k") == ["str", "int", "none"]


# ---- issubclass-on-Union safety (schema has a union + a File signal) --------


def test_file_signal_coexists_with_union(test_session):
    chain = _si(test_session, [1, 2], ["hi", 42])
    # Exercising schema introspection paths that historically did issubclass()
    # on the signal type (raises on a Union under py<3.11).
    assert chain.signals_schema.get_file_signal() is None
    assert "value" in chain.signals_schema.values
    assert chain.count() == 2


def test_union_with_file_arm_roundtrip(test_session):
    f = File(path="a.txt", source="s3://b")
    dc.read_values(
        id=[1, 2],
        v=[f, "plain"],
        output={"id": int, "v": Union[File, str]},
        session=test_session,
    ).save("u_file_arm")
    back = dc.read_dataset("u_file_arm", session=test_session)
    got = [v for _, v in back.order_by("id").to_list("id", "v")]
    assert isinstance(got[0], File) and got[0].path == "a.txt"
    assert got[1] == "plain"


# ---- arm-order stability across serialization -------------------------------


def test_arm_order_stable_after_save_reload(test_session):
    # Persisted schema serializes the Union; on reload the _type_tag index must
    # still map to the same arm (canonical ordering is the guarantee).
    _si(test_session, [1, 2, 3], [10, "ten", 20]).save("u_stable")
    back = dc.read_dataset("u_stable", session=test_session)
    assert _vals(back, "value") == [10, "ten", 20]
    # re-save the reloaded chain and read again (schema round-tripped twice).
    back.save("u_stable2")
    back2 = dc.read_dataset("u_stable2", session=test_session)
    assert _vals(back2, "value") == [10, "ten", 20]


# ---- arm matching hazards ---------------------------------------------------


def test_bool_int_union_preserves_type(test_session):
    # bool is a subclass of int; each value must round-trip as its own type.
    vals = [True, 1, False, 0]
    dc.read_values(
        id=[1, 2, 3, 4],
        v=vals,
        output={"id": int, "v": Union[int, bool]},
        session=test_session,
    ).save("u_boolint")
    back = dc.read_dataset("u_boolint", session=test_session)
    got = [v for _, v in back.order_by("id").to_list("id", "v")]
    assert [(type(x).__name__, x) for x in got] == [
        ("bool", True),
        ("int", 1),
        ("bool", False),
        ("int", 0),
    ]


def test_int_float_union_preserves_type(test_session):
    vals = [5, 2.5, 7]
    dc.read_values(
        id=[1, 2, 3],
        v=vals,
        output={"id": int, "v": Union[int, float]},
        session=test_session,
    ).save("u_intfloat")
    back = dc.read_dataset("u_intfloat", session=test_session)
    got = [v for _, v in back.order_by("id").to_list("id", "v")]
    assert [type(x).__name__ for x in got] == ["int", "float", "int"]
    assert got == [5, 2.5, 7]


# ---- group_by / merge -------------------------------------------------------


def test_group_by_partition_arm_leaf(test_session):
    chain = dc.read_values(
        id=[1, 2, 3, 4],
        value=["a", "a", "b", 1],
        output={"id": int, "value": Union[str, int]},
        session=test_session,
    )
    rows = chain.group_by(c=func.count(), partition_by="value._1").to_records()
    counts = sorted(r["c"] for r in rows)
    assert counts == [1, 1, 2]  # group 'a' (2), group 'b' (1), group NULL/int (1)


def test_count_present_union(test_session):
    chain = _si(test_session, [1, 2, 3, 4, 5], ["a", 1, None, "b", 2], none=True)
    assert chain.group_by(c=func.count("value")).to_values("c") == [4]


def test_union_same_type_combination(test_session):
    left = _si(test_session, [1, 2], ["a", 1])
    right = _si(test_session, [3, 4], [2, "b"])
    left.union(right).save("u_adv_comb")
    back = dc.read_dataset("u_adv_comb", session=test_session)
    assert sorted(str(v) for v in back.to_values("value")) == ["1", "2", "a", "b"]


def test_union_with_plain_scalar_raises_cleanly(test_session):
    import pytest

    from datachain.lib.utils import DataChainColumnError

    u = _si(test_session, [1, 2], ["a", 1])
    plain = dc.read_values(
        id=[3], value=["z"], output={"id": int, "value": str}, session=test_session
    )
    with pytest.raises(DataChainColumnError) as exc:
        u.union(plain).count()
    # error names the signal + logical types, never the internal arm/tag columns.
    assert "_0" not in str(exc.value) and "_type_tag" not in str(exc.value)


# ---- modern LLM API response shapes ----------------------------------------
# The dominant LLM-response union is `content: list[Union[Block, ...]]` (Anthropic
# content blocks, OpenAI Responses output items, Gemini parts) — a union *inside a
# list*, which is stored as a JSON column (not tagged) and round-trips via pydantic's
# `type` discriminator. A *field-level* union uses the tagged columns. Both must
# round-trip on SQLite and ClickHouse.


class _TextBlock(DataModel):
    type: Literal["text"] = "text"
    text: str = ""


class _ToolUseBlock(DataModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = {}  # noqa: RUF012


class _ThinkingBlock(DataModel):
    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    signature: str = ""


class _Usage(DataModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: Optional[int] = None


class _AnthropicMessage(DataModel):
    id: str = ""
    role: str = "assistant"
    content: list[Union[_TextBlock, _ToolUseBlock, _ThinkingBlock]] = []  # noqa: RUF012
    stop_reason: Optional[str] = None
    usage: _Usage = _Usage()


def test_llm_anthropic_message_roundtrip(test_session):
    msg = _AnthropicMessage(
        id="msg_1",
        content=[
            _ThinkingBlock(thinking="hmm", signature="sig"),
            _TextBlock(text="Hello"),
            _ToolUseBlock(id="tu_1", name="get_weather", input={"city": "SF"}),
        ],
        stop_reason="tool_use",
        usage=_Usage(input_tokens=10, output_tokens=5, cache_read_input_tokens=None),
    )
    dc.read_values(
        id=[1],
        msg=[msg],
        output={"id": int, "msg": _AnthropicMessage},
        session=test_session,
    ).save("llm_anthropic")
    back = dc.read_dataset("llm_anthropic", session=test_session).to_values("msg")[0]
    assert [type(b).__name__ for b in back.content] == [
        "_ThinkingBlock",
        "_TextBlock",
        "_ToolUseBlock",
    ]
    assert back.content[2].input == {"city": "SF"}
    assert back.stop_reason == "tool_use"
    assert back.usage.cache_read_input_tokens is None
    assert back == msg


class _OutputText(DataModel):
    type: Literal["output_text"] = "output_text"
    text: str = ""


class _OutputMessage(DataModel):
    type: Literal["message"] = "message"
    content: list[_OutputText] = []  # noqa: RUF012


class _FunctionToolCall(DataModel):
    type: Literal["function_call"] = "function_call"
    name: str = ""
    arguments: str = ""


class _OpenAIResponse(DataModel):
    id: str = ""
    output: list[Union[_OutputMessage, _FunctionToolCall]] = []  # noqa: RUF012


def test_llm_openai_response_roundtrip(test_session):
    r = _OpenAIResponse(
        id="r1",
        output=[
            _OutputMessage(content=[_OutputText(text="hi")]),
            _FunctionToolCall(name="f", arguments="{}"),
        ],
    )
    dc.read_values(
        id=[1], r=[r], output={"id": int, "r": _OpenAIResponse}, session=test_session
    ).save("llm_openai")
    back = dc.read_dataset("llm_openai", session=test_session).to_values("r")[0]
    assert [type(x).__name__ for x in back.output] == [
        "_OutputMessage",
        "_FunctionToolCall",
    ]
    assert back == r


def test_llm_block_as_field_level_union_is_tagged(test_session):
    # A single block as a field-level union DOES use the tagged columns (queryable).
    dc.read_values(
        id=[1, 2],
        block=[_TextBlock(text="hi"), _ToolUseBlock(name="wx")],
        output={"id": int, "block": Union[_TextBlock, _ToolUseBlock]},
        session=test_session,
    ).save("llm_block")
    chain = dc.read_dataset("llm_block", session=test_session)
    cols = chain.signals_schema.db_signals()
    assert "block___type_tag" in cols  # tagged, not a single JSON column
    assert [type(b).__name__ for b in chain.order_by("id").to_values("block")] == [
        "_TextBlock",
        "_ToolUseBlock",
    ]


def test_llm_explode_content_blocks_with_id(test_session):
    # The bridge over the queryability gap: explode `content: list[Union[...]]`
    # (an opaque JSON column) into one row per block via a multi-output generator
    # carrying the parent id; each block becomes a field-level (tagged) Union.
    msgs = [
        _AnthropicMessage(
            id="m1",
            content=[
                _ThinkingBlock(thinking="h"),
                _TextBlock(text="hi"),
                _ToolUseBlock(name="wx", input={"c": "SF"}),
            ],
        ),
        _AnthropicMessage(id="m2", content=[_TextBlock(text="bye")]),
    ]
    chain = dc.read_values(
        msg=msgs, output={"msg": _AnthropicMessage}, session=test_session
    )
    Block = Union[_TextBlock, _ToolUseBlock, _ThinkingBlock]

    def explode(msg: _AnthropicMessage) -> Iterator[tuple[str, Block]]:
        for block in msg.content:
            yield msg.id, block

    chain.gen(explode, output={"msg_id": str, "block": Block}).save("llm_exploded")
    out = dc.read_dataset("llm_exploded", session=test_session)

    assert out.count() == 4  # 3 blocks from m1 + 1 from m2
    # the exploded block is a tagged (queryable) union, not opaque JSON
    assert "block___type_tag" in out.signals_schema.db_signals()
    pairs = sorted((mid, type(b).__name__) for mid, b in out.to_list("msg_id", "block"))
    assert pairs == [
        ("m1", "_TextBlock"),
        ("m1", "_ThinkingBlock"),
        ("m1", "_ToolUseBlock"),
        ("m2", "_TextBlock"),
    ]


# ---- readable arm access (no _0 / __ in user code) -------------------------

_LLM_Block = Union[_TextBlock, _ToolUseBlock, _ThinkingBlock]


def _exploded_blocks(test_session):
    msgs = [
        _AnthropicMessage(
            id="m1",
            content=[
                _ThinkingBlock(thinking=".."),
                _TextBlock(text="hi"),
                _ToolUseBlock(name="search"),
                _ToolUseBlock(name="search"),
            ],
        ),
        _AnthropicMessage(
            id="m2", content=[_TextBlock(text="ok"), _ToolUseBlock(name="get_weather")]
        ),
    ]
    chain = dc.read_values(
        msg=msgs, output={"msg": _AnthropicMessage}, session=test_session
    )

    def explode(msg: _AnthropicMessage):  # -> Iterator[tuple[str, _LLM_Block]]
        for block in msg.content:
            yield msg.id, block

    return chain.gen(explode, output={"msg_id": str, "block": _LLM_Block}).persist()


def test_readable_arm_access_field_direct(test_session):
    blocks = _exploded_blocks(test_session)
    # C("block.text") -> the TextBlock arm's text; NULL on other rows. No _0 / __.
    texts = blocks.mutate(t=C("block.text")).to_values("t")
    assert sorted(v for v in texts if v is not None) == ["hi", "ok"]
    # filter / count on a readable arm field
    assert blocks.filter(func.not_(func.isnone("block.name"))).count() == 3
    assert blocks.group_by(c=func.count("block.text")).to_values("c") == [2]


def test_readable_arm_access_type_qualified(test_session):
    blocks = _exploded_blocks(test_session)
    # type-qualified by the arm's class name (these test models are _ToolUseBlock)
    names = blocks.mutate(n=C("block._ToolUseBlock.name")).to_values("n")
    assert sorted(v for v in names if v is not None) == [
        "get_weather",
        "search",
        "search",
    ]


def test_readable_arm_access_vectorized_group_by(test_session):
    blocks = _exploded_blocks(test_session)
    counts = (
        blocks.mutate(tool=C("block.name"))
        .filter(func.not_(func.isnone("tool")))
        .group_by(n=func.count(), partition_by="tool")
        .to_list("tool", "n")
    )
    assert sorted(counts) == [("get_weather", 1), ("search", 2)]


def test_readable_arm_access_ambiguous_field_raises(test_session):
    import pytest

    from datachain.lib.signal_schema import SignalResolvingError

    blocks = dc.read_values(
        id=[1],
        block=[_TextBlock(text="x")],
        output={"id": int, "block": _LLM_Block},
        session=test_session,
    )
    # `type` exists on every arm -> strict resolution errors (never silently picks).
    with pytest.raises(SignalResolvingError):
        blocks.mutate(x=C("block.type")).to_values("x")
    # type-qualifying the arm disambiguates it.
    assert blocks.mutate(x=C("block._TextBlock.type")).to_values("x") == ["text"]


def test_union_display_is_readable_not_positional(test_session):
    # schema print and to_pandas show arm columns by type name, never _0 / __.
    chain = dc.read_values(
        id=[1, 2],
        item=[UFoo(a=1, b="z"), UBar(x=2.5)],
        output={"id": int, "item": Union[UFoo, UBar]},
        session=test_session,
    ).order_by("id")
    schema_str = str(chain.schema)
    assert "_0" not in schema_str and "_type_tag" not in schema_str
    assert "UFoo" in schema_str and "UBar" in schema_str
    cols = list(chain.to_pandas().columns)
    assert ("item", "UFoo", "a") in cols
    assert ("item", "UBar", "x") in cols
    assert not any("_0" in str(c) or "_1" in str(c) for c in cols)


def test_variant_type(test_session):
    blocks = _exploded_blocks(test_session)  # Text x2, Thinking x1, ToolUse x3
    assert sorted(blocks.mutate(t=func.variant_type("block")).to_values("t")) == [
        "_TextBlock",
        "_TextBlock",
        "_ThinkingBlock",
        "_ToolUseBlock",
        "_ToolUseBlock",
        "_ToolUseBlock",
    ]
    # count blocks by arm type in one group_by
    by_arm = blocks.group_by(
        n=func.count(), partition_by=func.variant_type("block").label("kind")
    ).to_list("kind", "n")
    assert sorted(by_arm) == [
        ("_TextBlock", 2),
        ("_ThinkingBlock", 1),
        ("_ToolUseBlock", 3),
    ]


def test_variant_type_nullable_and_errors(test_session):
    import pytest

    from datachain.lib.utils import DataChainColumnError

    chain = _si(test_session, [1, 2, 3], ["a", 1, None], none=True)
    vals = chain.mutate(t=func.variant_type("value")).to_values("t")
    assert sorted(vals, key=lambda x: (x is None, x)) == ["int", "str", None]
    with pytest.raises(DataChainColumnError):
        chain.mutate(t=func.variant_type("id")).to_values("t")  # not a union
