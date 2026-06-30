from collections.abc import Iterator
from unittest import mock

import cloudpickle
import pytest
from pydantic import BaseModel

from datachain import llm
from datachain.lib.file import File, ImageFile, TextFile, VideoFile, VideoFrame
from datachain.lib.settings import Settings
from datachain.llm import engine
from datachain.llm.content import build_messages, serialize_value, value_to_parts
from datachain.llm.spec import MODEL_ENV_VAR, LLMConfigError
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


def bind(spec, **settings_kwargs):
    return spec.__datachain_bind__(Settings(**settings_kwargs))


def test_complete_default_output_is_str():
    assert llm.complete("text").output_type() is str


def test_complete_schema_output_is_model():
    assert llm.complete("text", schema=Scene).output_type() is Scene


def test_complete_list_schema_returns_iterator_annotation():
    spec = llm.complete("text", schema=list[Chunk])
    assert spec.output_type() is Chunk
    assert spec.return_annotation() == Iterator[Chunk]


def test_classify_output_is_str():
    assert llm.classify("text", into=["a", "b"]).output_type() is str


def test_score_output_is_float():
    assert llm.score("text").output_type() is float


def test_embed_output_is_list_float():
    assert llm.embed("text").output_type() == list[float]


def test_bound_callable_declares_input_column_and_return_type():
    import inspect

    f = bind(llm.complete("file", schema=Scene), llm="m")
    assert f.__datachain_params__ == ["file"]
    assert inspect.signature(f).return_annotation is Scene


def test_bound_callable_declares_context_column():
    f = bind(llm.complete("file", context="meta"), llm="m")
    assert f.__datachain_params__ == ["file", "meta"]


def test_nested_column_name_is_supported():
    f = bind(llm.complete("file.path"), llm="m")
    assert f.__datachain_params__ == ["file.path"]


def test_per_call_model_overrides_settings(fake_llm):
    bind(llm.complete("t", llm="call/m"), llm="settings/m")("hi")
    assert fake_llm.calls[-1]["model"] == "call/m"


def test_settings_model_used(fake_llm):
    bind(llm.complete("t"), llm="settings/m")("hi")
    assert fake_llm.calls[-1]["model"] == "settings/m"


def test_env_model_is_fallback(fake_llm, monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "env/m")
    bind(llm.complete("t"))("hi")
    assert fake_llm.calls[-1]["model"] == "env/m"


def test_missing_model_raises(monkeypatch):
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    with pytest.raises(LLMConfigError, match="no model configured"):
        bind(llm.complete("t"))


def test_llm_params_dict_forwarded(fake_llm):
    bind(llm.complete("t"), llm="m", llm_params={"api_key": "K", "api_base": "B"})("hi")
    assert fake_llm.calls[-1]["api_key"] == "K"
    assert fake_llm.calls[-1]["api_base"] == "B"


def test_llm_params_callable_resolved_at_call_time(fake_llm):
    resolved = []

    def factory():
        resolved.append(1)
        return {"api_key": "LAZY"}

    f = bind(llm.complete("t"), llm="m", llm_params=factory)
    assert not resolved  # not called at bind time
    f("hi")
    assert fake_llm.calls[-1]["api_key"] == "LAZY"
    assert resolved == [1]


def test_llm_params_callable_resolved_once_per_worker(fake_llm):
    resolved = []

    def factory():
        resolved.append(1)
        return {"api_key": "K"}

    f = bind(llm.complete("t"), llm="m", llm_params=factory)
    f("a")
    f("b")
    f("c")
    assert resolved == [1]  # resolved once, not once per row


def test_per_call_params_override_settings_params(fake_llm):
    f = bind(
        llm.complete("t", temperature=0.0),
        llm="m",
        llm_params={"temperature": 1.0},
    )
    f("hi")
    assert fake_llm.calls[-1]["temperature"] == 0.0


def test_fallback_forwarded(fake_llm):
    bind(llm.complete("t", fallback="openai/x"), llm="m")("hi")
    assert fake_llm.calls[-1]["fallbacks"] == ["openai/x"]


def test_fallback_list_forwarded(fake_llm):
    bind(llm.complete("t", fallback=["a", "b"]), llm="m")("hi")
    assert fake_llm.calls[-1]["fallbacks"] == ["a", "b"]


def test_retries_are_not_multiplied(fake_llm):
    fake_llm.invalid_json_attempts = 99
    with pytest.raises(engine.LLMError):
        bind(llm.complete("t", schema=Scene, retries=3), llm="m")("hi")
    assert len(fake_llm.calls) == 4  # retries + 1, never (retries + 1) ** 2


def test_negative_retries_still_makes_one_attempt(fake_llm):
    fake_llm.text_response = "ok"
    assert bind(llm.complete("t", retries=-5), llm="m")("hi") == "ok"
    assert len(fake_llm.calls) == 1


def test_structured_retries_on_invalid_json_then_succeeds(fake_llm):
    fake_llm.invalid_json_attempts = 1
    out = bind(llm.complete("t", schema=Scene, retries=1), llm="m")("hi")
    assert isinstance(out, Scene)
    assert len(fake_llm.calls) == 2


def test_structured_raises_after_exhausting_retries(fake_llm):
    fake_llm.invalid_json_attempts = 5
    with pytest.raises(engine.LLMError, match="did not match schema"):
        bind(llm.complete("t", schema=Scene, retries=1), llm="m")("hi")


def test_retries_delegated_to_litellm(fake_llm):
    bind(llm.complete("t", retries=4), llm="m")("hi")
    assert fake_llm.calls[-1]["num_retries"] == 4


def test_complete_text(fake_llm):
    fake_llm.text_response = "a summary"
    assert bind(llm.complete("t", "summarize"), llm="m")("doc") == "a summary"


def test_classify_returns_a_category(fake_llm):
    out = bind(llm.classify("t", into=["accident", "normal"]), llm="m")("x")
    assert out in {"accident", "normal"}
    # response_format constrains output to the categories
    schema = fake_llm.calls[-1]["response_format"]
    assert schema.model_fields["category"].annotation.__args__ == ("accident", "normal")


def test_score_returns_float(fake_llm):
    out = bind(llm.score("t", "risk 0..1"), llm="m")("x")
    assert isinstance(out, float)


def test_score_rejects_non_finite(fake_llm):
    fake_llm.structured_overrides["LLMScore"] = '{"score": "nan"}'
    with pytest.raises(engine.LLMError):
        bind(llm.score("t", "x"), llm="m")("v")


def test_embed_returns_vector(fake_llm):
    fake_llm.embedding_response = [1.0, 2.0]
    out = bind(llm.embed("t"), llm="m")("x")
    assert out == [1.0, 2.0]
    assert fake_llm.embedding_calls[-1]["input"] == ["x"]


def test_complete_list_schema_returns_list(fake_llm):
    out = bind(llm.complete("t", schema=list[Chunk], prompt="split"), llm="m")("doc")
    assert isinstance(out, list)
    assert all(isinstance(c, Chunk) for c in out)


def test_list_schema_tolerates_bare_array_response(fake_llm):
    fake_llm.structured_overrides["LLMListOutput"] = '[{"text": "a"}, {"text": "b"}]'
    out = bind(llm.complete("t", schema=list[Chunk]), llm="m")("doc")
    assert [c.text for c in out] == ["a", "b"]


def test_classify_requires_categories():
    with pytest.raises(ValueError, match="non-empty list of strings"):
        llm.classify("t", into=[])


def test_classify_rejects_non_string_categories():
    with pytest.raises(ValueError, match="non-empty list of strings"):
        llm.classify("t", into=[1, 2])


def test_classify_rejects_duplicate_categories():
    with pytest.raises(ValueError, match="distinct"):
        llm.classify("t", into=["a", "a"])


def test_extract_is_complete_with_schema():
    spec = llm.extract("t", Scene)
    assert spec.kind == "complete"
    assert spec.schema is Scene


def test_text_only_message_collapses_to_string():
    msgs = build_messages("hi", "world")
    assert msgs[0]["content"] == "hi\n\nworld"


def test_image_file_encodes_to_data_uri():
    img = ImageFile(path="a/pic.png", source="s3://x")
    with mock.patch.object(ImageFile, "read_bytes", return_value=b"PNGDATA"):
        parts = value_to_parts(img)
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_makes_message_content_a_list():
    img = ImageFile(path="a/pic.jpg", source="s3://x")
    with mock.patch.object(ImageFile, "read_bytes", return_value=b"JPG"):
        msgs = build_messages("describe", img)
    assert isinstance(msgs[0]["content"], list)


def test_video_frame_encodes_as_image():
    frame = VideoFrame(
        video=VideoFile(path="v.mp4", source="s3://x"), frame=0, timestamp=0.0
    )
    with mock.patch.object(VideoFrame, "read_bytes", return_value=b"JPG"):
        parts = value_to_parts(frame)
    assert parts[0]["type"] == "image_url"


def test_text_file_reads_text():
    tf = TextFile(path="a.txt", source="s3://x")
    with mock.patch.object(TextFile, "read_text", return_value="contents"):
        parts = value_to_parts(tf)
    assert parts == [{"type": "text", "text": "contents"}]


def test_explicit_text_type_is_never_sent_as_image():
    # type="text" must win over an image-looking extension.
    tf = TextFile(path="report.png", source="s3://x")
    with mock.patch.object(TextFile, "read_text", return_value="text body"):
        parts = value_to_parts(tf)
    assert parts == [{"type": "text", "text": "text body"}]


def test_pydantic_value_serialized_as_json():
    result = serialize_value(Scene(objects=["c"], risk=0.1))
    assert result == '{"objects":["c"],"risk":0.1}'


def test_serialize_text_file_returns_content_not_metadata():
    tf = TextFile(path="a.txt", source="s3://x")
    with mock.patch.object(TextFile, "read_text", return_value="hello world"):
        assert serialize_value(tf) == "hello world"


def test_binary_file_raises_clear_error():
    err = UnicodeDecodeError("utf-8", b"\x89", 0, 1, "bad")
    with mock.patch.object(File, "read_text", side_effect=err):
        with pytest.raises(engine.LLMError, match="cannot read"):
            value_to_parts(File(path="doc.pdf", source="s3://x"))


def test_uncommon_image_extension_still_encoded():
    with mock.patch.object(File, "read_bytes", return_value=b"<svg/>"):
        parts = value_to_parts(File(path="x.svg", source="s3://x"))
    assert parts[0]["type"] == "image_url"


def test_embed_image_errors_instead_of_returning_metadata():
    err = UnicodeDecodeError("utf-8", b"\x89", 0, 1, "bad")
    with mock.patch.object(File, "read_text", side_effect=err):
        with pytest.raises(engine.LLMError):
            serialize_value(ImageFile(path="a.png", source="s3://x"))


@pytest.mark.parametrize("bad", [list, dict, int, "Scene", list[int], Scene | None])
def test_invalid_schema_rejected(bad):
    with pytest.raises(TypeError, match="pydantic model"):
        llm.complete("t", schema=bad)


def test_context_appended_to_message():
    msgs = build_messages("p", "v", context=Scene(objects=["c"], risk=0.2))
    assert "Context:" in msgs[0]["content"]
    assert '"risk":0.2' in msgs[0]["content"]


def test_identity_changes_with_model():
    a = llm.complete("t", "p").identity("m1")
    b = llm.complete("t", "p").identity("m2")
    assert a != b


def test_identity_changes_with_prompt():
    a = llm.complete("t", "p").identity("m")
    b = llm.complete("t", "q").identity("m")
    assert a != b


def test_identity_changes_with_schema():
    class Other(BaseModel):
        a: int

    a = llm.complete("t", schema=Scene).identity("m")
    b = llm.complete("t", schema=Other).identity("m")
    assert a != b


def test_identity_stable_for_same_config():
    a = llm.complete("t", "p", schema=Scene).identity("m")
    b = llm.complete("t", "p", schema=Scene).identity("m")
    assert a == b


def test_identity_changes_with_param_value():
    a = llm.complete("t", "p", temperature=0.0).identity("m")
    b = llm.complete("t", "p", temperature=1.0).identity("m")
    assert a != b


def test_param_value_changes_udf_hash():
    from datachain.lib.signal_schema import SignalSchema
    from datachain.lib.udf import Mapper
    from datachain.lib.udf_signature import UdfSignature

    def udf_hash(spec):
        f = spec.__datachain_bind__(Settings(llm="m"))
        sign = UdfSignature.parse("", {"x": f}, None, None, None, False)
        return Mapper._create(sign, SignalSchema({"text": str})).hash()

    cold = udf_hash(llm.complete("text", temperature=0.0))
    hot = udf_hash(llm.complete("text", temperature=1.0))
    assert cold != hot


def test_bound_callable_is_picklable(fake_llm):
    f = bind(llm.complete("t", "p", schema=Scene), llm="m")
    restored = cloudpickle.loads(cloudpickle.dumps(f))
    assert isinstance(restored("hi"), Scene)


def test_settings_validates_llm_type():
    from datachain.lib.settings import SettingsError

    with pytest.raises(SettingsError, match="'llm' argument"):
        Settings(llm=123)


def test_settings_validates_llm_params_type():
    from datachain.lib.settings import SettingsError

    with pytest.raises(SettingsError, match="'llm_params' argument"):
        Settings(llm_params="nope")


def test_settings_llm_not_in_to_dict():
    # llm/llm_params are consumed at build time, never forwarded to the executor.
    assert "llm" not in Settings(llm="m").to_dict()
    assert "llm_params" not in Settings(llm_params={"k": 1}).to_dict()
