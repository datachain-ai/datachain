import types
from collections.abc import Iterator
from unittest import mock

import cloudpickle
import pytest
from pydantic import BaseModel, create_model

from datachain import llm
from datachain.lib.file import (
    AudioFile,
    File,
    ImageFile,
    TextFile,
    VideoFile,
    VideoFrame,
)
from datachain.lib.settings import Settings
from datachain.lib.udf import BindContext
from datachain.llm import engine
from datachain.llm.content import (
    Document,
    Image,
    Text,
    build_messages,
    resolve,
    to_text,
)
from datachain.llm.spec import LLMConfigError
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


GEN = types.SimpleNamespace(is_output_batched=True, is_input_batched=False)  # .gen()


def bind(spec, target=None, **settings_kwargs):
    return spec.bind(BindContext(settings=Settings(**settings_kwargs), target=target))


def test_complete_default_output_is_str():
    assert llm.complete("text").output_type() is str


def test_complete_schema_output_is_model():
    assert llm.complete("text", schema=Scene).output_type() is Scene


def test_complete_list_schema_annotation_depends_on_verb():
    spec = llm.complete("text", schema=list[Chunk])
    assert spec.output_type() == list[Chunk]
    assert spec.return_annotation() == (list[Chunk] | None)  # .map(): list column
    assert spec.return_annotation(to_many=True) == Iterator[Chunk]  # .gen(): fan out


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
    assert inspect.signature(f).return_annotation == (Scene | None)


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


def test_missing_model_raises():
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


def test_structured_raises_on_unparsable_output(fake_llm):
    fake_llm.invalid_json_attempts = 1
    with pytest.raises(engine.LLMError, match="could not be parsed"):
        bind(llm.complete("t", schema=Scene), llm="m")("hi")
    assert len(fake_llm.calls) == 1


def test_retries_delegated_to_litellm(fake_llm):
    bind(llm.complete("t", retries=4), llm="m")("hi")
    assert fake_llm.calls[-1]["num_retries"] == 4


def test_complete_text(fake_llm):
    fake_llm.text_response = "a summary"
    assert bind(llm.complete("t", "summarize"), llm="m")("doc") == "a summary"


def test_classify_returns_a_category(fake_llm):
    out = bind(llm.classify("t", into=["accident", "normal"]), llm="m")("x")
    assert out in {"accident", "normal"}
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


def test_list_schema_raises_on_unparsable_output(fake_llm):
    fake_llm.structured_overrides["LLMListOutput"] = "not a json list"
    with pytest.raises(engine.LLMError, match=r"list\[Chunk\]"):
        bind(llm.complete("t", schema=list[Chunk], retries=0), llm="m")("doc")


def test_classify_requires_categories():
    with pytest.raises(ValueError, match="non-empty list of strings"):
        llm.classify("t", into=[])


def test_classify_rejects_non_string_categories():
    with pytest.raises(ValueError, match="non-empty list of strings"):
        llm.classify("t", into=[1, 2])


def test_classify_rejects_duplicate_categories():
    with pytest.raises(ValueError, match="distinct"):
        llm.classify("t", into=["a", "a"])


def test_classify_rejects_string_into():
    with pytest.raises(ValueError, match="not a single string"):
        llm.classify("t", into="yes")


def test_complete_with_schema_kind():
    spec = llm.complete("t", schema=Scene)
    assert spec.kind == "complete"
    assert spec.schema is Scene


def _png() -> bytes:
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (2, 2), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def test_str_resolves_to_text():
    assert resolve("hi") == Text("hi")


def test_model_resolves_to_json_text():
    assert resolve(Scene(objects=["c"], risk=0.1)) == Text(
        '{"objects":["c"],"risk":0.1}'
    )


def test_text_file_resolves_to_text():
    tf = TextFile(path="a.txt", source="s3://x")
    with mock.patch.object(TextFile, "read_text", return_value="contents"):
        assert resolve(tf) == Text("contents")


def test_image_file_resolves_to_image_with_mime():
    img = ImageFile(path="a/pic.png", source="s3://x")
    with mock.patch.object(ImageFile, "read_bytes", return_value=b"PNGDATA"):
        c = resolve(img)
    assert isinstance(c, Image)
    assert c.mime == "image/png"


def test_image_file_without_extension_sniffed_from_bytes():
    img = ImageFile(path="snapshot", source="s3://x")
    with mock.patch.object(ImageFile, "read_bytes", return_value=_png()):
        assert resolve(img).mime == "image/png"


def test_image_file_unrecognized_bytes_rejected():
    img = ImageFile(path="snapshot", source="s3://x")
    with mock.patch.object(ImageFile, "read_bytes", return_value=b"x"):
        with pytest.raises(engine.LLMError, match="not a recognized image"):
            resolve(img)


def test_video_frame_resolves_to_jpeg_image():
    frame = VideoFrame(
        video=VideoFile(path="v.mp4", source="s3://x"), frame=0, timestamp=0.0
    )
    with mock.patch.object(VideoFrame, "read_bytes", return_value=b"JPG"):
        c = resolve(frame)
    assert isinstance(c, Image)
    assert c.mime == "image/jpeg"


def test_explicit_text_type_never_guessed_as_image():
    tf = TextFile(path="report.png", source="s3://x")  # image-looking name, but text
    with mock.patch.object(TextFile, "read_text", return_value="body"):
        assert resolve(tf) == Text("body")


def test_untyped_file_defaults_to_text_not_image():
    f = File(path="x.png", source="s3://x")  # untyped: no image auto-detection
    with mock.patch.object(File, "read_text", return_value="textual"):
        assert resolve(f) == Text("textual")


def test_audio_and_video_files_require_decoding():
    for f in (
        AudioFile(path="a.mp3", source="s3://x"),
        VideoFile(path="v.mp4", source="s3://x"),
    ):
        with pytest.raises(engine.LLMError, match="decode it"):
            resolve(f)


def test_binary_file_as_text_errors():
    err = UnicodeDecodeError("utf-8", b"\x89", 0, 1, "bad")
    with mock.patch.object(File, "read_text", side_effect=err):
        with pytest.raises(engine.LLMError, match="looks binary"):
            resolve(File(path="data.bin", source="s3://x"))


def test_binary_bytes_as_text_errors():
    with pytest.raises(engine.LLMError, match="binary"):
        resolve(b"\x00\x01\xff not text")


def test_utf8_bytes_resolve_to_text():
    assert resolve(b"hello") == Text("hello")


def test_binary_error_hint_depends_on_context():
    # resolve() (complete) may set type; to_text() (embed/context) cannot
    with pytest.raises(engine.LLMError, match="type="):
        resolve(b"\x00\xff")
    with pytest.raises(engine.LLMError) as exc:
        to_text(b"\x00\xff")
    assert "type=" not in str(exc.value)


def test_type_image_on_bytes_sniffs_format():
    c = resolve(_png(), type_="image")
    assert isinstance(c, Image)
    assert c.mime == "image/png"


def test_type_image_on_non_image_bytes_errors():
    with pytest.raises(engine.LLMError, match="not a recognized image"):
        resolve(b"\x00\x01 not an image", type_="image")


def test_type_image_on_untyped_file_sniffs_bytes():
    f = File(path="blob", source="s3://x")  # no image extension -> sniff bytes
    with mock.patch.object(File, "read_bytes", return_value=_png()):
        c = resolve(f, type_="image")
    assert isinstance(c, Image)
    assert c.mime == "image/png"


def test_type_image_on_untyped_file_rejects_non_image():
    f = File(path="blob", source="s3://x")
    with mock.patch.object(File, "read_bytes", return_value=b"raw"):
        with pytest.raises(engine.LLMError, match="not a recognized image"):
            resolve(f, type_="image")


def test_type_document_on_pdf_bytes():
    c = resolve(b"%PDF-1.7\nbody", type_="document")
    assert isinstance(c, Document)
    part = c.as_part()
    assert part["type"] == "file"
    assert part["file"]["format"] == "application/pdf"
    assert part["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_type_document_on_pdf_file():
    f = File(path="r.pdf", source="s3://x")
    with mock.patch.object(File, "read_bytes", return_value=b"%PDF-1.4 x"):
        c = resolve(f, type_="document")
    assert isinstance(c, Document)


def test_type_document_on_non_pdf_errors():
    with pytest.raises(engine.LLMError, match="expects a PDF"):
        resolve(b"not a pdf at all", type_="document")


def test_type_image_on_unsupported_value_type():
    with pytest.raises(engine.LLMError, match="cannot send int as an image"):
        resolve(123, type_="image")


def test_type_document_on_unsupported_value_type():
    with pytest.raises(engine.LLMError, match="cannot send str as a document"):
        resolve("just text", type_="document")


def test_build_messages_rejects_empty_input():
    with pytest.raises(engine.LLMError, match="empty message"):
        build_messages(None, "")


def test_type_text_forces_model_json():
    c = resolve(Scene(objects=[], risk=0.0), type_="text")
    assert c == Text('{"objects":[],"risk":0.0}')


def test_invalid_type_rejected_at_build_time():
    with pytest.raises(ValueError, match="type must be"):
        llm.complete("t", type="vidjo")


def test_build_messages_text_only_collapses_to_string():
    assert build_messages("hi", "world")[0]["content"] == "hi\n\nworld"


def test_build_messages_with_image_is_a_list():
    img = ImageFile(path="a/pic.jpg", source="s3://x")
    with mock.patch.object(ImageFile, "read_bytes", return_value=b"JPG"):
        msgs = build_messages("describe", img)
    assert isinstance(msgs[0]["content"], list)


def test_build_messages_appends_context():
    msgs = build_messages("p", "v", context=Scene(objects=["c"], risk=0.2))
    assert "Context:" in msgs[0]["content"]
    assert '"risk":0.2' in msgs[0]["content"]


def test_to_text_reads_text_file():
    tf = TextFile(path="a.txt", source="s3://x")
    with mock.patch.object(TextFile, "read_text", return_value="hello"):
        assert to_text(tf) == "hello"


def test_to_text_on_image_file_errors():
    err = UnicodeDecodeError("utf-8", b"\x89", 0, 1, "bad")
    with mock.patch.object(File, "read_text", side_effect=err):
        with pytest.raises(engine.LLMError):
            to_text(ImageFile(path="a.png", source="s3://x"))


def test_document_passed_to_model(fake_llm):
    fake_llm.text_response = "summary"
    pdf = File(path="r.pdf", source="s3://x")
    with mock.patch.object(File, "read_bytes", return_value=b"%PDF-1.4 x"):
        out = bind(llm.complete("file", "summarize", type="document"), llm="m")(pdf)
    assert out == "summary"
    content = fake_llm.calls[-1]["messages"][0]["content"]
    assert any(p["type"] == "file" for p in content)


@pytest.mark.parametrize(
    "key", ["model", "messages", "input", "num_retries", "fallbacks", "response_format"]
)
def test_reserved_params_rejected(key):
    with pytest.raises(ValueError, match=r"managed by datachain\.llm"):
        llm.complete("c", **{key: "x"})


def test_col_equal_context_rejected():
    with pytest.raises(ValueError, match="different columns"):
        llm.complete("text", context="text")


def test_retries_must_be_int():
    with pytest.raises(ValueError, match="retries must be an int"):
        llm.complete("c", retries=1.5)


def test_retries_must_be_non_negative():
    with pytest.raises(ValueError, match="retries must be >= 0"):
        llm.complete("c", retries=-5)


def test_fallback_must_be_string_or_list_of_strings():
    with pytest.raises(ValueError, match="fallback must be"):
        llm.complete("c", fallback=123)
    with pytest.raises(ValueError, match="fallback must be"):
        llm.complete("c", fallback=["ok", 5])


def test_llm_params_callable_returning_non_dict_raises(fake_llm):
    f = bind(llm.complete("t"), llm="m", llm_params=lambda: ["not", "a", "dict"])
    with pytest.raises(TypeError, match="llm_params must resolve to a dict"):
        f("hi")


def test_identity_stable_across_param_dict_order():
    a = llm.complete("c", opt={"x": 1, "y": 2}).identity("m")
    b = llm.complete("c", opt={"y": 2, "x": 1}).identity("m")
    assert a == b


def test_canonical_orders_sets_and_dicts():
    from datachain.llm.spec import _canonical

    assert _canonical({"a", "b", "c"}) == _canonical({"c", "b", "a"})
    assert _canonical({"x": 1, "y": 2}) == _canonical({"y": 2, "x": 1})


def test_param_clobber_is_blocked(fake_llm):
    bind(llm.complete("t", "real prompt", temperature=0.0), llm="real/model")("hi")
    call = fake_llm.calls[-1]
    assert call["model"] == "real/model"
    assert call["messages"][0]["content"] == "real prompt\n\nhi"


def test_embed_empty_data_errors(fake_llm):
    fake_llm.embedding_empty = True
    with pytest.raises(engine.LLMError, match="no data"):
        bind(llm.embed("t"), llm="m")("x")


def test_empty_fallback_is_noop(fake_llm):
    bind(llm.complete("t", fallback=[]), llm="m")("hi")
    assert "fallbacks" not in fake_llm.calls[-1]


def test_structured_truncation_raises(fake_llm):
    fake_llm.finish_reason = "length"
    with pytest.raises(engine.LLMError, match="truncated"):
        bind(llm.complete("c", schema=Scene), llm="m")("hi")


def test_structured_accepts_fenced_json(fake_llm):
    fake_llm.structured_overrides["Scene"] = (
        '```json\n{"objects": [], "risk": 0.3}\n```'
    )
    out = bind(llm.complete("c", schema=Scene), llm="m")("hi")
    assert out.risk == 0.3


@pytest.mark.parametrize("bad", [list, dict, int, "Scene", list[int], Scene | None])
def test_invalid_schema_rejected(bad):
    with pytest.raises(TypeError, match="pydantic model"):
        llm.complete("t", schema=bad)


def test_usage_defaults():
    u = Usage()
    assert (u.input_tokens, u.output_tokens) == (0, 0)


def test_none_input_skips_model_call(fake_llm):
    assert bind(llm.complete("t", schema=Scene), llm="m")(None) is None
    assert bind(llm.score("t"), llm="m")(None) is None
    assert bind(llm.embed("t"), llm="m")(None) is None
    assert fake_llm.calls == []
    assert fake_llm.embedding_calls == []


def test_none_input_return_annotation_is_optional():
    assert llm.complete("t", schema=Scene).return_annotation() == (Scene | None)
    assert llm.score("t").return_annotation() == (float | None)


def test_none_input_with_usage_pairs_none_and_zero_usage(fake_llm):
    value, usage = bind(llm.complete("t", include_usage=True), llm="m")(None)
    assert value is None
    assert (usage.input_tokens, usage.output_tokens) == (0, 0)
    assert fake_llm.calls == []


def test_none_input_in_gen_yields_no_rows(fake_llm):
    out = bind(llm.complete("t", schema=list[Chunk]), target=GEN, llm="m")(None)
    assert list(out) == []
    assert fake_llm.calls == []


def test_include_usage_return_annotation():
    text = llm.complete("c", include_usage=True).return_annotation()
    assert text == tuple[str | None, Usage]
    ann = llm.complete("c", schema=Scene, include_usage=True).return_annotation()
    assert ann == tuple[Scene | None, Usage]


def test_include_usage_list_return_annotation():
    spec = llm.complete("c", schema=list[Scene], include_usage=True)
    assert spec.return_annotation() == tuple[list[Scene] | None, Usage]  # .map()
    gen = spec.return_annotation(to_many=True)  # .gen()
    assert gen == Iterator[tuple[Scene, Usage]]


def test_include_usage_runtime_pairs_value_and_usage(fake_llm):
    fake_llm.text_response = "hi"
    value, usage = bind(llm.complete("c", include_usage=True), llm="m")("x")
    assert value == "hi"
    assert isinstance(usage, Usage)
    assert usage.input_tokens == 11
    assert usage.output_tokens == 7


def test_without_usage_returns_bare_value(fake_llm):
    fake_llm.text_response = "hi"
    assert bind(llm.complete("c"), llm="m")("x") == "hi"


def test_include_usage_gen_attributes_usage_to_first_row(fake_llm):
    fake_llm.structured_overrides["LLMListOutput"] = (
        '{"items": [{"objects": [], "risk": 0.1}, {"objects": [], "risk": 0.2}]}'
    )
    spec = llm.complete("c", schema=list[Scene], include_usage=True)
    rows = bind(spec, target=GEN, llm="m")("x")  # .gen() fans out
    assert len(rows) == 2
    (i0, u0), (i1, u1) = rows
    assert isinstance(i0, Scene) and isinstance(i1, Scene)
    assert u0.input_tokens == 11
    assert (u1.input_tokens, u1.output_tokens) == (0, 0)


def test_list_schema_in_map_returns_list_value(fake_llm):
    value = bind(llm.complete("c", schema=list[Scene]), llm="m")("x")  # .map()
    assert isinstance(value, list)
    assert all(isinstance(s, Scene) for s in value)


def test_embed_usage_has_no_output_tokens(fake_llm):
    _, usage = bind(llm.embed("c", include_usage=True), llm="m")("x")
    assert usage.input_tokens == 5
    assert usage.output_tokens == 0


def test_include_usage_in_identity():
    a = llm.complete("c").identity("m")
    b = llm.complete("c", include_usage=True).identity("m")
    assert a != b


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


def test_identity_changes_with_input_column():
    a = llm.complete("a", "p").identity("m")
    b = llm.complete("b", "p").identity("m")
    assert a != b


@pytest.mark.parametrize(
    ("a_spec", "b_spec"),
    [
        (llm.complete("t", "p"), llm.complete("t", "p", type="image")),
        (llm.complete("t", "p"), llm.complete("t", "p", context="ctx")),
    ],
)
def test_identity_changes_with_output_affecting_field(a_spec, b_spec):
    assert a_spec.identity("m") != b_spec.identity("m")


def test_retries_and_fallback_not_in_identity():
    base = llm.complete("t", "p").identity("m")
    assert llm.complete("t", "p", retries=5).identity("m") == base
    assert llm.complete("t", "p", fallback="x/y").identity("m") == base


def test_identity_changes_with_list_schema_fields():
    # Same class name, edited fields → must invalidate (the list[Model] path).
    a_model = create_model("Same", x=(int, ...))
    b_model = create_model("Same", x=(int, ...), y=(int, ...))
    a = llm.complete("t", schema=list[a_model]).identity("m")
    b = llm.complete("t", schema=list[b_model]).identity("m")
    assert a != b


def test_identity_changes_with_dict_llm_params():
    spec = llm.complete("t", "p")
    assert spec.identity("m", {"temperature": 0.0}) != spec.identity(
        "m", {"temperature": 1.0}
    )


def test_callable_llm_params_not_in_identity():
    # Callable llm_params is per-worker/runtime (e.g. credentials), not cache-keyed;
    # output-affecting params belong in the dict form or per-call kwargs.
    spec = llm.complete("t", "p")
    assert spec.identity("m", lambda: {"k": "v"}) == spec.identity("m")


def test_opaque_param_value_warns_about_unstable_cache_key():
    class Opaque:
        pass

    with pytest.warns(UserWarning, match="no stable repr"):
        llm.complete("t", client=Opaque()).identity("m")


def test_stable_param_values_do_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        llm.complete("t", temperature=0.0, opt={"a": 1}).identity("m")


def test_secret_params_not_in_identity():
    base = llm.complete("t", "p").identity("m")
    secret_keys = ("api_key", "api_token", "access_token", "client_secret")
    for k in secret_keys:
        assert llm.complete("t", "p", **{k: "SECRET"}).identity("m") == base
    # rotating a nested secret must not change the key
    a = llm.complete("t", "p", extra_headers={"Authorization": "OLD"}).identity("m")
    b = llm.complete("t", "p", extra_headers={"Authorization": "NEW"}).identity("m")
    assert a == b
    # max_tokens is output-affecting, not a secret
    assert llm.complete("t", "p", max_tokens=100).identity("m") != llm.complete(
        "t", "p", max_tokens=200
    ).identity("m")


def test_embed_object_style_data_item(fake_llm):
    fake_llm.embedding_as_object = True
    fake_llm.embedding_response = [0.1, 0.2]
    assert bind(llm.embed("c"), llm="m")("x") == [0.1, 0.2]


def test_retries_delegated_to_litellm_on_structured_path(fake_llm):
    bind(llm.complete("c", schema=Scene, retries=3), llm="m")("hi")
    assert fake_llm.calls[-1]["num_retries"] == 3


def test_provider_error_propagates(fake_llm):
    fake_llm.fatal_status = 500
    with pytest.raises(RuntimeError):
        bind(llm.complete("c", schema=Scene, retries=5), llm="m")("hi")
    assert len(fake_llm.calls) == 1


def test_fallback_engages_on_primary_failure(fake_llm):
    fake_llm.fail_models = {"primary/m"}
    fake_llm.text_response = "ok"
    out = bind(llm.complete("c", fallback="backup/m"), llm="primary/m")("x")
    assert out == "ok"
    assert fake_llm.calls[-1]["fallbacks"] == ["backup/m"]


def test_primary_failure_without_fallback_propagates(fake_llm):
    fake_llm.fail_models = {"primary/m"}
    with pytest.raises(RuntimeError):
        bind(llm.complete("c"), llm="primary/m")("x")


def test_param_value_changes_udf_hash():
    from datachain.lib.signal_schema import SignalSchema
    from datachain.lib.udf import Mapper
    from datachain.lib.udf_signature import UdfSignature

    def udf_hash(spec):
        f = bind(spec, llm="m")
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


def test_litellm_helper_returns_module():
    litellm = pytest.importorskip("litellm")
    assert engine._litellm() is litellm


def test_content_raises_on_no_choices():
    with pytest.raises(engine.LLMError, match="no choices"):
        engine._content(types.SimpleNamespace(choices=[]))


def test_content_raises_on_no_message():
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=None)])
    with pytest.raises(engine.LLMError, match="no message"):
        engine._content(resp)


def test_finish_reason_empty_on_no_choices():
    assert engine._finish_reason(types.SimpleNamespace(choices=[])) == ""
