import pytest
from pydantic import BaseModel

import datachain as dc
from datachain import llm
from datachain.llm import engine
from tests.llm_fakes import FakeLiteLLM


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLiteLLM()
    monkeypatch.setattr(engine, "_litellm", lambda: fake)
    return fake


class Scene(BaseModel):
    objects: list[str]
    risk: float


class Chunk(BaseModel):
    text: str


class Doc(BaseModel):
    body: str


def base(session):
    return dc.read_values(
        text=["a frame", "another frame", "third"], session=session
    ).settings(llm="anthropic/claude-haiku-4-5")


def test_map_materializes_typed_columns(fake_llm, test_session):
    fake_llm.text_response = "summary"
    fake_llm.embedding_response = [0.5, 0.6]

    chain = (
        base(test_session)
        .map(topic=llm.classify("text", into=["accident", "normal"]))
        .map(risk=llm.score("text", "accident risk 0..1"))
        .map(scene=llm.complete("text", schema=Scene))
        .map(vec=llm.embed("text"))
        .map(summary=llm.complete("text", "summarize"))
    )

    assert chain.schema["topic"] is str
    assert chain.schema["risk"] is float
    assert chain.schema["scene"] is Scene
    assert chain.schema["summary"] is str

    records = chain.to_records()
    assert len(records) == 3
    row = records[0]
    assert row["topic"] in {"accident", "normal"}
    assert isinstance(row["risk"], float)
    assert row["scene__objects"] == ["x"]
    assert row["vec"] == [0.5, 0.6]
    assert row["summary"] == "summary"


def test_save_and_reload_preserves_types(fake_llm, test_session):
    base(test_session).map(scene=llm.complete("text", schema=Scene)).save("scenes")

    reloaded = dc.read_dataset("scenes", session=test_session)
    assert reloaded.schema["scene"] is Scene
    rows = reloaded.to_records()
    assert all(r["scene__risk"] == 0.5 for r in rows)


def test_filter_on_nested_llm_field(fake_llm, test_session):
    fake_llm.structured_overrides["Scene"] = '{"objects": ["car"], "risk": 0.9}'
    chain = base(test_session).map(scene=llm.complete("text", schema=Scene))
    assert chain.filter(dc.C("scene.risk") > 0.8).count() == 3
    assert chain.filter(dc.C("scene.risk") > 0.95).count() == 0


def test_score_then_filter_recall_pattern(fake_llm, test_session):
    fake_llm.structured_overrides["LLMScore"] = '{"score": 0.7}'
    chain = base(test_session).map(spoiler=llm.score("text", "spoiler 0..1"))
    # Re-thresholding is a cheap recall on the materialized column.
    assert chain.filter(dc.C("spoiler") > 0.5).count() == 3
    assert chain.filter(dc.C("spoiler") > 0.8).count() == 0


def test_llm_column_as_only_selected_signal(fake_llm, test_session):
    fake_llm.text_response = "label"
    out = (
        base(test_session)
        .map(label=llm.complete("text", "x"))
        .select("label")
        .to_records()
    )
    assert out == [{"label": "label"}] * 3


def test_gen_one_to_many(fake_llm, test_session):
    fake_llm.structured_overrides["LLMListOutput"] = (
        '{"items": [{"text": "one"}, {"text": "two"}]}'
    )
    chain = (
        dc.read_values(text=["doc"], session=test_session)
        .settings(llm="m")
        .gen(chunk=llm.complete("text", schema=list[Chunk], prompt="split"))
    )

    assert chain.schema["chunk"] is Chunk
    records = chain.to_records()
    assert [r["chunk__text"] for r in records] == ["one", "two"]


def test_list_schema_in_map_rejected(test_session):
    from datachain.llm.spec import LLMConfigError

    with pytest.raises(LLMConfigError, match="use .gen"):
        base(test_session).map(chunk=llm.complete("text", schema=list[Chunk]))


def test_list_schema_in_agg_rejected(test_session):
    from datachain.llm.spec import LLMConfigError

    with pytest.raises(LLMConfigError, match="use .gen"):
        base(test_session).agg(
            chunk=llm.complete("text", schema=list[Chunk]), partition_by="text"
        )


def test_one_to_one_op_in_gen_rejected(test_session):
    from datachain.llm.spec import LLMConfigError

    with pytest.raises(LLMConfigError, match="use .map"):
        base(test_session).gen(label=llm.complete("text", "x"))


def test_nested_column_input(fake_llm, test_session):
    fake_llm.text_response = "ok"
    out = (
        dc.read_values(doc=[Doc(body="hello")], session=test_session)
        .settings(llm="m")
        .map(label=llm.complete("doc.body", "summarize"))
        .to_records()
    )
    assert out == [{"doc__body": "hello", "label": "ok"}]


@pytest.mark.parametrize("swap", [False, True])
def test_union_both_arm_orders(fake_llm, test_session, swap):
    fake_llm.text_response = "L"
    left = (
        dc.read_values(text=["a", "b"], session=test_session)
        .settings(llm="m")
        .map(label=llm.complete("text", "x"))
    )
    right = (
        dc.read_values(text=["c"], session=test_session)
        .settings(llm="m")
        .map(label=llm.complete("text", "x"))
    )

    a, b = (right, left) if swap else (left, right)
    unioned = a.union(b)
    assert unioned.count() == 3
    assert {r["label"] for r in unioned.to_records()} == {"L"}


def test_export_to_pandas_roundtrip(fake_llm, test_session):
    fake_llm.embedding_response = [1.0, 2.0, 3.0]
    df = base(test_session).map(vec=llm.embed("text")).to_pandas()
    assert "vec" in df.columns
    assert list(df["vec"].iloc[0]) == [1.0, 2.0, 3.0]


def test_settings_inherited_by_all_downstream_ops(fake_llm, test_session):
    # One settings(llm=) governs every llm.* below it.
    (
        base(test_session)
        .map(a=llm.complete("text", "x"))
        .map(b=llm.classify("text", into=["p", "q"]))
        .to_records()
    )
    assert all(c["model"] == "anthropic/claude-haiku-4-5" for c in fake_llm.calls)


def test_per_call_model_overrides_chain_setting(fake_llm, test_session):
    base(test_session).map(
        a=llm.complete("text", "x", llm="openai/gpt-5-mini")
    ).to_records()
    assert fake_llm.calls[-1]["model"] == "openai/gpt-5-mini"


def test_missing_model_raises_at_build_time(test_session, monkeypatch):
    monkeypatch.delenv("DATACHAIN_AI_MODEL", raising=False)
    from datachain.llm.spec import LLMConfigError

    with pytest.raises(LLMConfigError):
        dc.read_values(text=["a"], session=test_session).map(
            x=llm.complete("text", "y")
        )
