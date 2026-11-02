import pytest

import datachain as dc
from datachain.lib.signal_schema import SignalResolvingError
from datachain.lib.udf_signature import UdfSignatureError
from datachain.lib.utils import DataChainParamsError
from datachain.llm.spec import LLMConfigError


@pytest.mark.parametrize("style", ["signature", "params", "multi"])
def test_batch_map_rejects_scalar_column_annotation(test_session, style):
    def bad(value: int) -> list[int]:
        return [value]

    def good(value: list[int]) -> list[int]:
        return value

    source = dc.read_values(value=[1], session=test_session)
    with pytest.raises(SignalResolvingError, match=r"is not a list.*list\[T\]"):
        if style == "signature":
            source.batch_map(result=bad, batch=2)
        elif style == "params":
            source.batch_map(result=bad, params=["value"], batch=2)
        else:
            source.batch_map(bad=bad, good=good, batch=2)


@pytest.mark.parametrize(
    "method,kwargs,hint",
    [
        ("map", {"batch": 32}, "Use batch_map"),
        ("map", {"batch_size": 32}, "database insert buffering"),
        ("batch_map", {"batch_size": 32, "batch": 2}, "Use batch_map"),
        ("batch_map", {"parallel": 2, "batch": 2}, "Use settings"),
        ("batch_map", {"prefetch": 8, "batch": 2}, "Use settings"),
    ],
)
def test_map_options_have_actionable_errors(test_session, method, kwargs, hint):
    def encode(value: list[int]) -> list[int]:
        return value

    source = dc.read_values(value=[1], session=test_session)
    with pytest.raises(DataChainParamsError, match=hint) as exc:
        getattr(source, method)(result=encode, **kwargs)
    assert next(iter(kwargs)) in str(exc.value)


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("operation", [dc.llm.embed, dc.llm.complete])
def test_batch_map_rejects_scalar_llm_operations(test_session, multi, operation):
    def good(text: list[str]) -> list[str]:
        return text

    source = dc.read_values(text=["hello"], session=test_session)
    with pytest.raises(LLMConfigError, match=r"use \.map\(\)"):
        if multi:
            source.batch_map(result=operation("text"), good=good, batch=2)
        else:
            source.batch_map(result=operation("text"), batch=2)


@pytest.mark.parametrize("output_style", ["type", "dict"])
def test_batch_map_rejects_unresolved_model_annotation_before_execution(
    test_session, output_style
):
    class Local(dc.DataModel):
        value: int

    def predict(value: list[int]) -> "list[Local]":
        return [Local(value=item) for item in value]

    source = dc.read_values(value=[1, 2], session=test_session)
    with pytest.raises(UdfSignatureError, match=r"list\[T\].*output="):
        source.batch_map(result=predict, batch=2)

    output = Local if output_style == "type" else {"result": Local}
    assert source.batch_map(result=predict, batch=2, output=output).to_values(
        "result.value"
    ) == [1, 2]
