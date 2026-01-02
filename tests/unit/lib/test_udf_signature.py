import typing as t
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from datachain.lib.data_model import DataType
from datachain.lib.file import File
from datachain.lib.udf import Mapper
from datachain.lib.udf_signature import UdfSignature, UdfSignatureError


def get_sign(
    func: Callable | None = None,
    params: str | Sequence[str] | None = None,
    output: DataType | Sequence[str] | dict[str, DataType] | None = None,
    **signal_map,
):
    return UdfSignature.parse(
        "test",
        signal_map,
        func,
        params,
        output,
        is_input_batched=False,
        is_generator=False,
    )


def get_sign_batched(
    func: Callable | None = None,
    params: str | Sequence[str] | None = None,
    output: DataType | Sequence[str] | dict[str, DataType] | None = None,
    **signal_map,
):
    """Aggregator-style input: the UDF receives a batch (list) per call."""
    return UdfSignature.parse(
        "test",
        signal_map,
        func,
        params,
        output,
        is_input_batched=True,
        is_generator=True,
    )


def func_str(p1) -> str:
    return "qwe"


def func_tuple(p1) -> tuple[BaseModel, str, int]:
    return File(name="n1"), "qwe", 33


def func_args(*args):
    return 12345


def func_typed(p1: int) -> int:
    return p1 * 2


def test_basic():
    sign = get_sign(s1=func_str)

    assert sign.func == func_str
    assert sign.params == {"p1": Any}
    assert sign.output_schema.values == {"s1": str}


def test_basic_func():
    sign1 = get_sign(s1=func_str)
    sign2 = get_sign(func_str, output="s1")
    sign3 = get_sign(func_str, output=["s1"])
    sign4 = get_sign(s1=func_str, params="p1")
    sign5 = get_sign(s1=func_str, params=["p1"])

    assert sign1 == sign2
    assert sign1 == sign3
    assert sign1 == sign4
    assert sign1 == sign5


def test_signature_overwrite():
    sign = get_sign(s1=func_str, output={"my_sign": int}, params="some_prm")

    assert sign.func == func_str
    assert sign.params == {"some_prm": Any}
    assert sign.output_schema.values == {"my_sign": int}


def test_output_feature():
    sign = get_sign(s1=func_str, output={"my_sign": File})

    assert sign.output_schema.values == {"my_sign": File}


def test_output_as_value():
    sign = get_sign(s1=func_str, output="my_sign")

    assert sign.func == func_str
    assert sign.params == {"p1": Any}
    assert sign.output_schema.values == {"my_sign": str}


def test_output_as_list():
    sign = get_sign(s1=func_str, output=["my_sign"])

    assert sign.func == func_str
    assert sign.params == {"p1": Any}
    assert sign.output_schema.values == {"my_sign": str}


def test_multi_outputs_not_supported_yet():
    sign = get_sign(s1=func_tuple, output=["o1", "o2", "o3"])

    assert sign.output_schema.values == {"o1": BaseModel, "o2": str, "o3": int}


def test_multiple_signals_error():
    with pytest.raises(UdfSignatureError):
        get_sign(my_out=func_tuple, my_out2=func_str)

    with pytest.raises(UdfSignatureError):
        get_sign(func_tuple, my_out=func_str)


def test_no_outputs():
    with pytest.raises(UdfSignatureError):
        get_sign(func_tuple)

    with pytest.raises(UdfSignatureError):
        get_sign()


def test_tuple_output_number_mismatch():
    with pytest.raises(UdfSignatureError):
        get_sign(func_tuple, output=["a1", "a2", "a3", "a4", "a5"])


def test_no_params():
    with pytest.raises(UdfSignatureError):
        get_sign(lambda: 4)


def test_func_with_args():
    sign = get_sign(func_args, params=["prm1", "prm2"], output={"res": int})
    assert sign.params == {"prm1": Any, "prm2": Any}


def test_output_type_error():
    with pytest.raises(UdfSignatureError):
        get_sign(func_str, output={"res": complex})

    with pytest.raises(UdfSignatureError):

        class TestCls:
            pass

        get_sign(func_str, output={"res": TestCls})


def test_feature_to_tuple_string_as_default_type():
    sign = get_sign(val1=lambda file: "asd")
    assert sign.output_schema.values == {"val1": str}


def test_callable_class():
    class MyTest:
        def __call__(self, file, p2) -> float:
            return 2.72

    sign = get_sign(s1=MyTest())
    assert sign.output_schema.values == {"s1": float}


def test_not_callable():
    class MyTest:
        def my_func(self, file, p2) -> float:
            return 2.72

    with pytest.raises(UdfSignatureError):
        get_sign(s1=MyTest())

    with pytest.raises(UdfSignatureError):
        get_sign(s1=123)


def test_udf_class():
    class MyTest(Mapper):
        def process(self, file, p2) -> int:
            return 42

    sign = get_sign(s1=MyTest())

    assert sign.output_schema.values == {"s1": int}
    assert sign.params == {"file": Any, "p2": Any}


def test_udf_flatten_value():
    class MyTest(Mapper):
        def process(self, file, pp) -> int:
            return 42

    sign = get_sign(MyTest(), output={"res1": int})

    assert sign.output_schema.values == {"res1": int}


def test_udf_flatten_feature():
    class MyData(BaseModel):
        text: str
        count: int

    class MyTest(Mapper):
        def process(self, file, pp) -> MyData:
            return MyData(text="asdf", count=135)

    sign = get_sign(r1=MyTest())

    assert sign.output_schema.values == {"r1": MyData}


def test_udf_typed_param():
    sign = get_sign(s1=func_typed)
    assert sign.params == {"p1": int}
    assert sign.output_schema.values == {"s1": int}


def test_unparameterized_iterator_defaults_to_str():
    def iter_func(p1) -> t.Iterator:
        yield "never"

    sign = get_sign(s1=iter_func)
    assert sign.output_schema.values == {"s1": str}


def test_params_all_sets_all_params_and_input_name_for_map():
    def f(row: dict[str, Any]) -> int:
        return 1

    sign = get_sign(r=f, params="*")
    assert sign.all_params is True
    assert sign.all_params_input == "row"
    assert sign.params == {"*": Any}


def test_params_all_accepts_annotated_dict_for_map():
    def f(row: t.Annotated[dict[str, Any], "payload"]) -> int:
        return 1

    sign = get_sign(r=f, params="*")
    assert sign.all_params is True
    assert sign.all_params_input == "row"


def test_params_all_in_list_sets_all_params_and_input_name_for_map():
    def f(row: dict[str, Any]) -> int:
        return 1

    sign = get_sign(r=f, params=["*", "a"])
    assert sign.all_params is True
    assert sign.all_params_input == "row"


def test_params_all_requires_dict_annotation_for_map():
    def f(row) -> int:
        return 1

    with pytest.raises(UdfSignatureError):
        get_sign(r=f, params="*")


def test_params_all_sets_all_params_and_input_name_for_agg():
    def agg(rows: list[dict[str, Any]]) -> t.Iterator[int]:
        yield len(rows)

    sign = get_sign_batched(r=agg, params="*")
    assert sign.all_params is True
    assert sign.all_params_input == "rows"


def test_params_all_requires_list_of_dict_annotation_for_agg():
    def agg(rows: list[int]) -> t.Iterator[int]:
        yield len(rows)

    with pytest.raises(UdfSignatureError):
        get_sign_batched(r=agg, params="*")


def test_dict_annotation_does_not_enable_all_params_by_default():
    def f(row: dict[str, Any]) -> int:
        return 1

    sign = get_sign(r=f)
    assert sign.all_params is False
