import copy
from collections.abc import Iterable
from typing import (  # noqa: UP035
    Annotated,
    Any,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import pytest
from pydantic import BaseModel

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.data_model import DataModel
from datachain.lib.model_store import ModelStore
from datachain.lib.utils import (
    callable_name,
    normalize_col_names,
    rebase_path,
    type_to_str,
)
from datachain.sql.types import Array, String


class MyModel(BaseModel):
    val1: str


class MyFeature(BaseModel):
    val1: str


@pytest.mark.parametrize(
    "typ,expected",
    (
        (list[str], Array(String())),
        (Iterable[str], Array(String())),
        (list[list[str]], Array(Array(String()))),
    ),
)
def test_convert_type_to_datachain_array(typ, expected):
    assert python_to_sql(typ).to_dict() == expected.to_dict()


@pytest.mark.parametrize(
    "typ",
    (
        str | int,
        list[str | int],
        MyFeature,
        MyModel,
    ),
)
def test_convert_type_to_datachain_error(typ):
    with pytest.raises(TypeError):
        python_to_sql(typ)


def test_normalize_column_names():
    res = normalize_col_names(
        [
            "UpperCase",
            "_underscore_start",
            "double__underscore",
            "1start_with_number",
            "не_ascii_start",
            "  space_start",
            "space_end  ",
            "dash-end-",
            "-dash-start",
            "--multiple--dash--",
            "-_ mix_  -dash_ -",
            "__2digit_after_uderscore",
            "",
            "_-_-  _---_ _",
            "_-_-  _---_ _1",
        ]
    )
    assert list(res.keys()) == [
        "uppercase",
        "underscore_start",
        "double_underscore",
        "c0_1start_with_number",
        "ascii_start",
        "space_start",
        "space_end",
        "dash_end",
        "dash_start",
        "multiple_dash",
        "mix_dash",
        "c1_2digit_after_uderscore",
        "c2",
        "c3",
        "c4_1",
    ]


def test_normalize_column_names_case_repeat():
    res = normalize_col_names(["UpperCase", "UpPerCase"])

    assert list(res.keys()) == ["uppercase", "c0_uppercase"]


def test_normalize_column_names_exists_after_normalize():
    res = normalize_col_names(["1digit", "c0_1digit"])

    assert list(res.keys()) == ["c1_1digit", "c0_1digit"]


def test_normalize_column_names_normalized_repeat():
    res = normalize_col_names(["column", "_column"])

    assert list(res.keys()) == ["column", "c0_column"]


def test_normalize_column_names_normalized_case_repeat():
    res = normalize_col_names(["CoLuMn", "_column"])

    assert res == {"column": "CoLuMn", "c0_column": "_column"}


def test_normalize_column_names_repeat_generated_after_normalize():
    res = normalize_col_names(["c0_CoLuMn", "_column", "column"])

    assert res == {"c0_column": "c0_CoLuMn", "c1_column": "_column", "column": "column"}


def test_rebase_path_basic():
    result = rebase_path(
        "/data/audio/folder1/file.wav", "/data/audio", "/output/waveforms"
    )
    assert result == "/output/waveforms/folder1/file.wav"


def test_rebase_path_with_s3_uri():
    result = rebase_path(
        "s3://bucket/data/audio/folder/file.wav",
        "data/audio",
        "s3://output-bucket/waveforms",
    )
    assert result == "s3://output-bucket/waveforms/folder/file.wav"


def test_rebase_path_mixed_uri_schemes():
    result = rebase_path(
        "/local/data/audio/file.mp3", "/local/data/audio", "s3://bucket/output"
    )
    assert result == "s3://bucket/output/file.mp3"


def test_rebase_path_with_suffix():
    result = rebase_path(
        "/data/audio/file.wav", "/data/audio", "/output", suffix="_processed"
    )
    assert result == "/output/file_processed.wav"


def test_rebase_path_with_extension_change():
    result = rebase_path("/data/audio/file.wav", "audio", "/output", extension="npy")
    assert result == "/output/file.npy"


def test_rebase_path_base_dir_not_in_path():
    with pytest.raises(
        ValueError, match="old_base '/data/audio' not found in src_path"
    ):
        rebase_path("/different/path/file.wav", "/data/audio", "/output")


def test_rebase_path_partial_match_base_dir():
    result = rebase_path("/home/user/data/audio/file.wav", "data/audio", "/output")
    assert result == "/output/file.wav"


def test_rebase_path_complex_s3_paths():
    result = rebase_path(
        "s3://bucket/balanced_train_segments/audio/folder/file.flac",
        "s3://bucket/balanced_train_segments",
        "s3://output-bucket/waveforms",
        suffix="_ch1",
        extension="npy",
    )
    assert result == "s3://output-bucket/waveforms/audio/folder/file_ch1.npy"


def test_rebase_path_file_without_extension():
    result = rebase_path("/data/audio/file_no_ext", "/data/audio", "/output")
    assert result == "/output/file_no_ext"

    # With new extension
    result = rebase_path(
        "/data/audio/file_no_ext", "/data/audio", "/output", extension="txt"
    )
    assert result == "/output/file_no_ext.txt"


def test_callable_name_function():
    def f():
        return 1

    assert callable_name(f) == "f"


def test_callable_name_lambda():
    g = lambda x: x  # noqa: E731
    assert callable_name(g) == "<lambda>"


def test_callable_name_bound_method():
    class Bar:
        def method(self):
            return 2

    b = Bar()
    assert callable_name(b.method) == "method"


@pytest.mark.parametrize(
    "type_, expected",
    [
        (int, "int"),
        (float, "float"),
        (None, "NoneType"),
        (Ellipsis, "..."),
        (Any, "Any"),
        (Final[int], "Final"),
        (Optional[int], "Optional[int]"),  # noqa: UP045
        (int | str, {"Union[int, str]", "Union[str, int]"}),
        (
            str | int | bool,
            {
                "Union[int, str, bool]",
                "Union[str, int, bool]",
                "Union[str, bool, int]",
                "Union[int, bool, str]",
                "Union[bool, str, int]",
                "Union[bool, int, str]",
            },
        ),
        (Annotated[int, "meta"], "int"),
        (list[Any], "list[Any]"),
        (list[bool], "list[bool]"),
        (List[bool], "list[bool]"),  # noqa: UP006
        (list[bool | None], "list[Optional[bool]]"),
        (List[bool | None], "list[Optional[bool]]"),  # noqa: UP006
        (List[int], "list[int]"),  # noqa: UP006
        (List[str], "list[str]"),  # noqa: UP006
        (Optional[list[bytes]], "Optional[list[bytes]]"),  # noqa: UP045
        (Literal["x"] | None, "Optional[Literal]"),
        (tuple[int, float], "tuple[int, float]"),
        (tuple[int, ...], "tuple[int, ...]"),
        (Optional[tuple[int, float]], "Optional[tuple[int, float]]"),  # noqa: UP045
        (dict[str], "dict[str, Any]"),  # type: ignore[misc]
        (dict[str, bool], "dict[str, bool]"),
        (Dict[str, bool], "dict[str, bool]"),  # noqa: UP006
        (dict[str, int], "dict[str, int]"),
        (Dict[str, int], "dict[str, int]"),  # noqa: UP006
        (Union[list[bytes], None], "Optional[list[bytes]]"),  # noqa: UP007
        (Union[List[bytes], None], "Optional[list[bytes]]"),  # noqa: UP006, UP007
    ],
)
def test_type_to_str_matrix(type_, expected):
    result = type_to_str(type_)
    if isinstance(expected, set):
        assert result in expected
    else:
        assert result == expected


def test_type_to_str_typing_module_vs_builtin_generics():
    """Ensure typing.List/Dict and built-in generics stringify identically.

    Confirms Python 3.10+ behavior where get_origin() normalizes both forms
    to built-ins, and type_to_str produces the same string.
    """
    from typing import get_origin

    assert get_origin(List[int]) is list  # noqa: UP006
    assert get_origin(list[int]) is list
    assert get_origin(Dict[str, int]) is dict  # noqa: UP006
    assert get_origin(dict[str, int]) is dict

    assert type_to_str(List[int]) == type_to_str(list[int])  # noqa: UP006
    assert type_to_str(Dict[str, int]) == type_to_str(dict[str, int])  # noqa: UP006
    assert type_to_str(List[str]) == "list[str]"  # noqa: UP006
    assert type_to_str(list[str]) == "list[str]"
    assert type_to_str(Dict[str, bool]) == "dict[str, bool]"  # noqa: UP006
    assert type_to_str(dict[str, bool]) == "dict[str, bool]"


def test_type_to_str_warn_with_called_for_unknown():
    # Unknown types should fall back to Any but emit a warning via the callback.
    calls: list[str] = []

    def collect(msg: str) -> None:
        calls.append(msg)

    result = type_to_str(object(), warn_with=collect)
    assert result == "Any"
    assert calls and "Unable to determine name" in calls[0]


def test_type_to_str_warns_without_callback():
    with pytest.warns(RuntimeWarning, match="Unable to determine name"):
        assert type_to_str(object()) == "Any"


def test_type_to_str_empty_generics():
    assert type_to_str(List) == "list"  # noqa: UP006
    assert type_to_str(Dict) == "dict"  # noqa: UP006
    assert type_to_str(Tuple) == "tuple"  # noqa: UP006


def test_type_to_str_pydantic_model_uses_model_store():
    snapshot = copy.deepcopy(ModelStore.store)
    ModelStore.store = {}
    try:

        class Sample(DataModel):
            a: int

        assert type_to_str(Sample) == ModelStore.get_name(Sample)
    finally:
        ModelStore.store = snapshot


def test_callable_name_callable_instance():
    class Foo:
        def __call__(self, x):
            return x

    foo = Foo()
    assert callable_name(foo) == "Foo"


def test_callable_name_udf_like():
    from datachain.lib.utils import AbstractUDF

    class MyUDF(AbstractUDF):
        def process(self, *args, **kwargs):
            # No return value expected by AbstractUDF interface
            pass

        def setup(self):
            pass

        def teardown(self):
            pass

    u = MyUDF()
    assert callable_name(u) == "MyUDF"


@pytest.mark.parametrize(
    "obj, expected",
    [
        ("hello", "hello"),
        (42, "42"),
    ],
)
def test_callable_name_non_callable(obj, expected):
    assert callable_name(obj) == expected
