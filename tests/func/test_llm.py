import pytest
from pydantic import BaseModel

import datachain as dc
from datachain import llm
from datachain.llm import engine
from datachain.llm.types import Usage
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

    assert chain.schema["topic"] == (str | None)
    assert chain.schema["risk"] == (float | None)
    assert chain.schema["scene"] == (Scene | None)
    assert chain.schema["summary"] == (str | None)

    rows = chain.to_list("topic", "risk", "scene", "vec", "summary")
    assert len(rows) == 3
    topic, risk, scene, vec, summary = rows[0]
    assert topic in {"accident", "normal"}
    assert isinstance(risk, float)
    assert scene.objects == ["x"]
    assert vec == [0.5, 0.6]
    assert summary == "summary"


def test_include_usage_multi_output_map(fake_llm, test_session):
    chain = base(test_session).map(
        llm.complete("text", schema=Scene, include_usage=True),
        output={"res": Scene, "tok": Usage},
    )
    assert chain.schema["res"] is Scene
    assert chain.schema["tok"] is Usage

    assert all(s.risk == 0.5 for s in chain.to_values("res"))
    toks = chain.to_values("tok")
    assert all(t.input_tokens == 11 for t in toks)
    assert all(t.output_tokens == 7 for t in toks)


def test_include_usage_gen_usage_at_call_grain(fake_llm, test_session):
    fake_llm.structured_overrides["LLMListOutput"] = (
        '{"items": [{"text": "a"}, {"text": "b"}]}'
    )
    chain = base(test_session).gen(
        llm.complete("text", schema=list[Chunk], include_usage=True),
        output={"chunk": Chunk, "tok": Usage},
    )
    assert chain.schema["chunk"] is Chunk
    assert chain.schema["tok"] is Usage

    toks = chain.to_values("tok")
    assert len(toks) == 6  # 3 input rows x 2 chunks
    # 3 calls x 11 tokens = 33, not 66
    assert sum(t.input_tokens for t in toks) == 33
    assert sorted(t.input_tokens for t in toks) == [0, 0, 0, 11, 11, 11]


def test_save_and_reload_preserves_types(fake_llm, test_session):
    base(test_session).map(scene=llm.complete("text", schema=Scene)).save("scenes")

    reloaded = dc.read_dataset("scenes", session=test_session)
    assert reloaded.schema["scene"] == (Scene | None)
    assert all(s.risk == 0.5 for s in reloaded.to_values("scene"))


def test_filter_on_nested_llm_field(fake_llm, test_session):
    fake_llm.structured_overrides["Scene"] = '{"objects": ["car"], "risk": 0.9}'
    chain = base(test_session).map(scene=llm.complete("text", schema=Scene))
    assert chain.filter(dc.C("scene.risk") > 0.8).count() == 3
    assert chain.filter(dc.C("scene.risk") > 0.95).count() == 0


def test_score_then_filter_recall_pattern(fake_llm, test_session):
    fake_llm.structured_overrides["LLMScore"] = '{"score": 0.7}'
    chain = base(test_session).map(spoiler=llm.score("text", "spoiler 0..1"))
    assert chain.filter(dc.C("spoiler") > 0.5).count() == 3
    assert chain.filter(dc.C("spoiler") > 0.8).count() == 0


def test_llm_column_as_only_selected_signal(fake_llm, test_session):
    fake_llm.text_response = "label"
    out = (
        base(test_session)
        .map(label=llm.complete("text", "x"))
        .select("label")
        .to_values("label")
    )
    assert out == ["label"] * 3


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
    assert sorted(chain.to_values("chunk.text")) == ["one", "two"]


def test_list_schema_in_map_yields_list_column(fake_llm, test_session):
    fake_llm.structured_overrides["LLMListOutput"] = (
        '{"items": [{"text": "one"}, {"text": "two"}]}'
    )
    chain = base(test_session).map(chunks=llm.complete("text", schema=list[Chunk]))

    assert chain.schema["chunks"] == (list[Chunk] | None)
    first = chain.to_values("chunks")[0]
    assert [c.text for c in first] == ["one", "two"]


def test_none_input_propagates_without_calling_model(fake_llm, test_session):
    chain = (
        dc.read_values(text=["hi", None], session=test_session)
        .settings(llm="anthropic/claude-haiku-4-5")
        .map(scene=llm.complete("text", schema=Scene))
    )
    scenes = chain.to_values("scene")
    assert None in scenes  # None input -> None output
    assert any(isinstance(s, Scene) for s in scenes)  # real input -> real call
    assert len(fake_llm.calls) == 1  # the None row never hit the model


def test_list_schema_in_agg_rejected(test_session):
    from datachain.llm.spec import LLMConfigError

    with pytest.raises(LLMConfigError, match=r"use \.gen"):
        base(test_session).agg(
            chunk=llm.complete("text", schema=list[Chunk]), partition_by="text"
        )


def test_one_to_one_op_in_gen_rejected(test_session):
    from datachain.llm.spec import LLMConfigError

    with pytest.raises(LLMConfigError, match=r"use \.map"):
        base(test_session).gen(label=llm.complete("text", "x"))


def test_nested_column_input(fake_llm, test_session):
    fake_llm.text_response = "ok"
    out = (
        dc.read_values(doc=[Doc(body="hello")], session=test_session)
        .settings(llm="m")
        .map(label=llm.complete("doc.body", "summarize"))
        .to_list("doc.body", "label")
    )
    assert out == [("hello", "ok")]


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
    assert set(unioned.to_values("label")) == {"L"}


def test_settings_inherited_by_all_downstream_ops(fake_llm, test_session):
    (
        base(test_session)
        .map(a=llm.complete("text", "x"))
        .map(b=llm.classify("text", into=["p", "q"]))
        .to_values("a")
    )
    assert all(c["model"] == "anthropic/claude-haiku-4-5" for c in fake_llm.calls)


def test_per_call_model_overrides_chain_setting(fake_llm, test_session):
    base(test_session).map(
        a=llm.complete("text", "x", llm="openai/gpt-5-mini")
    ).to_values("a")
    assert fake_llm.calls[-1]["model"] == "openai/gpt-5-mini"


def test_missing_model_raises_at_build_time(test_session):
    from datachain.llm.spec import LLMConfigError

    with pytest.raises(LLMConfigError):
        dc.read_values(text=["a"], session=test_session).map(
            x=llm.complete("text", "y")
        )
