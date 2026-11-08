from collections.abc import Iterator

import pytest

import datachain as dc


@pytest.mark.parametrize("method", ["map", "gen"])
def test_plain_udf_inputs_skip_async_prefetch(test_session, mocker, method):
    mocker.patch(
        "datachain.lib.udf.AsyncMapper", side_effect=AssertionError("prefetch")
    )
    chain = dc.read_values(text=["a/b", "c/d"], session=test_session)

    def split(text: str) -> tuple[str, str]:
        left, right = text.split("/")
        return left, right

    def generate(text: str) -> Iterator[tuple[str, str]]:
        yield split(text)

    result = getattr(chain, method)(
        split if method == "map" else generate,
        output={"left": str, "right": str},
    )
    assert result.order_by("left").to_list("left", "right") == [("a", "b"), ("c", "d")]


@pytest.mark.parametrize("method", ["map", "gen"])
@pytest.mark.parametrize(
    "annotation,value",
    [
        (int, 7),
        (float, 1.5),
        (bool, True),
        (str, "hello"),
        (bytes, b"hello"),
        (int | None, None),
        (int | None, 7),
        (int | str, 7),
        (int | str, "hello"),
        (list[str], ["a", "b"]),
        (list[float], [1.5, 2.5]),
        (dict[str, int], {"a": 7}),
        (dict[int, str], {7: "a"}),
    ],
)
def test_udf_input_types_survive_saved_dataset(test_session, annotation, value, method):
    expected = value
    dc.read_values(
        value=[value], output={"value": annotation}, session=test_session
    ).save("typed_input")

    def inspect(value) -> str:
        assert type(value) is type(expected)
        assert value == expected
        return "ok"

    def generate(value) -> Iterator[str]:
        yield inspect(value)

    result = getattr(dc.read_dataset("typed_input", session=test_session), method)(
        result=inspect if method == "map" else generate
    )
    assert result.to_values("result") == ["ok"]


@pytest.mark.parametrize("method", ["map", "gen"])
def test_udf_dotted_model_input_and_reordered_params(test_session, mocker, method):
    from datachain.lib.signal_schema import SignalSchema

    prepare = mocker.spy(SignalSchema, "row_to_objs")

    class Item(dc.DataModel):
        text: str

    chain = dc.read_values(
        item=[Item(text="a"), Item(text="b")], suffix=["x", "y"], session=test_session
    )

    def combine(suffix: str, text: str) -> str:
        return text + suffix

    def generate(suffix: str, text: str) -> Iterator[str]:
        yield combine(suffix, text)

    assert getattr(chain, method)(
        combine if method == "map" else generate,
        params=["suffix", "item.text"],
        output={"result": str},
    ).order_by("result").to_values("result") == ["ax", "by"]

    prepare.assert_not_called()


@pytest.mark.parametrize("method", ["map", "gen"])
@pytest.mark.parametrize("values", [[], [1, 2]])
def test_udf_without_inputs_and_with_setup_preserves_values(
    test_session, method, values
):
    chain = dc.read_values(x=values, output={"x": int}, session=test_session)

    def constant() -> str:
        return "fixed"

    def generate_constant() -> Iterator[str]:
        yield constant()

    assert getattr(chain, method)(
        result=constant if method == "map" else generate_constant
    ).to_values("result") == ["fixed"] * len(values)

    def configured(prefix: str, x: int) -> str:
        return prefix + str(x)

    def generate_configured(prefix: str, x: int) -> Iterator[str]:
        yield configured(prefix, x)

    assert getattr(chain.setup(prefix=lambda: "p"), method)(
        result=configured if method == "map" else generate_configured
    ).order_by("result").to_values("result") == [f"p{x}" for x in values]


@pytest.mark.parametrize("method", ["map", "gen"])
def test_udf_model_input_is_hydrated(test_session, method):
    class Item(dc.DataModel):
        value: tuple[int, ...]

    def inspect(item: Item) -> int:
        assert isinstance(item, Item)
        assert type(item.value) is tuple
        return item.value[0]

    def generate(item: Item) -> Iterator[int]:
        yield inspect(item)

    chain = dc.read_values(
        item=[Item(value=(1, 3)), Item(value=(2, 4))], session=test_session
    )
    assert getattr(chain, method)(
        result=inspect if method == "map" else generate
    ).order_by("result").to_values("result") == [1, 2]


@pytest.mark.parametrize("method", ["map", "gen"])
@pytest.mark.parametrize("setup", [False, True])
def test_file_udf_inputs_keep_prefetch_and_streams(
    test_session, tmp_path, mocker, method, setup
):
    path = tmp_path / "input.txt"
    path.write_text("content")
    prefetch = mocker.spy(dc.File, "_prefetch")
    if setup:
        chain = dc.read_values(x=[1, 2], session=test_session).setup(
            file=lambda: dc.File.at(path)
        )
    else:
        chain = dc.read_storage(path.as_uri(), session=test_session)

    def read(file: dc.File) -> str:
        return file.read().decode()

    def generate(file: dc.File) -> Iterator[str]:
        yield read(file)

    result = getattr(chain, method)(result=read if method == "map" else generate)
    assert result.to_values("result") == ["content"] * (2 if setup else 1)
    assert prefetch.call_count >= 1


@pytest.mark.parametrize("setup", [False, True])
def test_batch_map_plain_inputs_and_setup(test_session, mocker, setup):
    def transform(value: list[int]) -> list[str]:
        return [f"p{v}" for v in value]

    def configured(value: list[int], prefix: str) -> list[str]:
        return [f"{prefix}{v}" for v in value]

    chain = dc.read_values(value=[1, 2, 3], session=test_session)
    if setup:
        chain = chain.setup(prefix=lambda: "p")
    else:
        mocker.patch(
            "datachain.lib.udf.AsyncMapper", side_effect=AssertionError("prefetch")
        )
    result = chain.batch_map(result=configured if setup else transform, batch=2)
    assert sorted(result.to_values("result")) == ["p1", "p2", "p3"]


def test_batch_map_without_inputs(test_session):
    def constant() -> list[str]:
        return ["fixed"]

    chain = dc.read_values(value=[1, 2], session=test_session)
    assert (
        chain.batch_map(result=constant, batch=1).to_values("result") == ["fixed"] * 2
    )


@pytest.mark.parametrize("method", ["map", "batch_map"])
def test_multiple_plain_udfs_share_prepared_inputs(test_session, method):
    def left(first: int) -> int:
        return first * 2

    def right(second: int) -> int:
        return second * 3

    def batch_left(first: list[int]) -> list[int]:
        return [v * 2 for v in first]

    def batch_right(second: list[int]) -> list[int]:
        return [v * 3 for v in second]

    chain = dc.read_values(first=[1, 2], second=[3, 4], session=test_session)
    result = (
        chain.map(a=left, b=right)
        if method == "map"
        else chain.batch_map(a=batch_left, b=batch_right, batch=2)
    )
    assert result.order_by("first").to_list("a", "b") == [(2, 9), (4, 12)]
