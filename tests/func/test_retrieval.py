import math
from typing import Any, cast

import pytest

import datachain as dc
from datachain.lib.data_model import DataModel
from datachain.lib.file import File


class MyFr(DataModel):
    nnn: str
    count: int


class MyNested(DataModel):
    label: str
    fr: MyFr


class MyWithList(DataModel):
    """Model containing a list of primitives."""

    name: str
    values: list[int]


class MyWithModelList(DataModel):
    """Model containing a list of DataModels."""

    title: str
    items: list[MyFr]


class MyWithDict(DataModel):
    """Model containing a dict with primitive values."""

    name: str
    metadata: dict[str, str]


class MyWithModelDict(DataModel):
    """Model containing a dict with DataModel values."""

    title: str
    mapping: dict[str, MyFr]


def sort_files(files: list[File]) -> list[File]:
    """Sort files by path then size."""
    return sorted(files, key=lambda f: (f.path, f.size))


# Test data
features = [
    MyFr(nnn="n1", count=1),
    MyFr(nnn="n1", count=3),
    MyFr(nnn="n2", count=5),
]

features_nested = [
    MyNested(label="label_0", fr=MyFr(nnn="n1", count=1)),
    MyNested(label="label_1", fr=MyFr(nnn="n1", count=3)),
    MyNested(label="label_2", fr=MyFr(nnn="n2", count=5)),
]


def test_iterable_chain(test_session):
    """Test iterating over DataChain using for loop (calls __iter__ internally)."""
    chain = dc.read_values(f1=features, num=range(len(features)), session=test_session)

    n = 0
    for sample in chain.order_by("f1.nnn", "f1.count"):
        assert len(sample) == 2
        fr, num = sample

        assert isinstance(fr, MyFr)
        assert isinstance(num, int)
        assert num == n
        assert fr == features[n]

        n += 1

    assert n == len(features)


def test_to_list_nested_feature(test_session):
    """Test to_list() with nested DataModel features."""
    chain = dc.read_values(sign1=features_nested, session=test_session)

    for n, sample in enumerate(
        chain.order_by("sign1.fr.nnn", "sign1.fr.count").to_list()
    ):
        assert len(sample) == 1
        nested = sample[0]

        assert isinstance(nested, MyNested)
        assert nested == features_nested[n]


def test_collect_deprecated(test_session):
    """Test deprecated collect() method (should use to_iter instead)."""
    chain = dc.read_values(fib=[1, 1, 2, 3, 5], session=test_session)

    with pytest.warns(DeprecationWarning, match="Method `collect` is deprecated"):
        vals = list(chain.collect("fib"))
        assert set(vals) == {1, 2, 3, 5}


def test_to_values_and_to_list(test_session):
    """Test to_values() and to_list() with File objects and nested fields."""
    names = ["f1.jpg", "f1.json", "f1.txt", "f2.jpg", "f2.json"]
    sizes = [1, 2, 3, 4, 5]
    files = sort_files(
        [File(path=name, size=size) for name, size in zip(names, sizes, strict=False)]
    )

    scores = [0.1, 0.2, 0.3, 0.4, 0.5]

    chain = dc.read_values(file=files, score=scores, session=test_session)
    chain = chain.order_by("file.path", "file.size")

    # Test to_values() with File objects and nested fields
    assert chain.to_values("file") == files
    assert chain.to_values("file.path") == names
    assert chain.to_values("file.size") == sizes
    assert chain.to_values("file.source") == [""] * len(names)

    # Test to_values() with floats
    actual_scores = chain.to_values("score")
    for actual, expected in zip(actual_scores, scores, strict=False):
        assert math.isclose(actual, expected, rel_tol=1e-7)

    # Test to_list() with multiple columns
    for actual, expected in zip(
        chain.to_list("file.size", "score"),
        [[x, y] for x, y in zip(sizes, scores, strict=False)],
        strict=False,
    ):
        assert len(actual) == 2
        assert actual[0] == expected[0]
        assert math.isclose(actual[1], expected[1], rel_tol=1e-7)


def test_to_values_list_of_models(test_session):
    """Retrieval: list[DataModel] should round-trip via to_values/to_list."""
    rows = [
        [{"nnn": "n1", "count": 1}],
        [{"nnn": "n2", "count": 2}, {"nnn": "n3", "count": 3}],
    ]

    chain = dc.read_values(
        items=rows, session=test_session, output={"items": list[MyFr]}
    )

    # Schema should reflect parameterized element type
    schema_ser = chain.signals_schema.serialize()
    assert schema_ser["items"].startswith("list[MyFr@v1]")

    # Single-column retrieval as values
    items_vals = cast("list[list[MyFr]]", chain.to_values("items"))
    assert isinstance(items_vals, list)
    # Row order may vary across backends, so check lengths flexibly
    assert sorted([len(v) for v in items_vals]) == [1, 2]
    # Find the row with 2 items
    two_item_row = next(v for v in items_vals if len(v) == 2)
    assert all(isinstance(two_item_row[i], MyFr) for i in range(2))
    assert sorted([m.count for m in two_item_row]) == [2, 3]

    # Full-row retrieval
    rows_list = chain.to_list()
    assert len(rows_list) == 2
    assert len(rows_list[0]) == 1 and isinstance(rows_list[0][0], list)
    assert isinstance(rows_list[0][0][0], MyFr)


def test_to_values_dict_of_models_supported(test_session):
    """dict[str, DataModel] as an output type is now supported."""
    rows = [
        {
            "first": {"label": "a", "fr": {"nnn": "n1", "count": 1}},
            "second": {"label": "b", "fr": {"nnn": "n2", "count": 2}},
        }
    ]

    chain = dc.read_values(
        items=rows,
        session=test_session,
        output={"items": dict[str, MyNested]},
    )

    # Schema should reflect dict[str, MyNested]
    schema_ser = chain.signals_schema.serialize()
    assert schema_ser["items"].startswith("dict[str, MyNested@v1]")

    vals = cast("list[dict[str, MyNested]]", chain.to_values("items"))
    assert len(vals) == 1
    items_dict = vals[0]
    assert isinstance(items_dict, dict)
    assert set(items_dict.keys()) == {"first", "second"}
    assert isinstance(items_dict["first"], MyNested)
    assert items_dict["first"].label == "a"
    assert items_dict["first"].fr.nnn == "n1"
    assert items_dict["first"].fr.count == 1
    assert isinstance(items_dict["second"], MyNested)
    assert items_dict["second"].label == "b"


def test_optional_collection_roundtrip(test_session):
    """Retrieval: Optional[list[DataModel]] should handle None and non-None."""
    rows = [None, [{"nnn": "test", "count": 5}]]

    chain = dc.read_values(
        items=rows,
        session=test_session,
        output=cast("Any", {"items": list[MyFr] | None}),
    )

    # Schema should reflect Optional[list[MyFr]]
    schema_ser = chain.signals_schema.serialize()
    assert schema_ser["items"].startswith("Optional[list[MyFr@v1]]")

    vals = cast("list[list[MyFr] | None]", chain.to_values("items"))
    # Row order may vary across backends
    assert len(vals) == 2
    # ClickHouse may convert top-level None to a default.
    # Accept either None or an empty collection for the "None" case.
    has_none_or_empty = any(v is None or v in ([], {}) for v in vals)
    assert has_none_or_empty
    # Find the non-empty list and validate its content
    lists = [v for v in vals if isinstance(v, list)]
    # There must be at least one list value
    assert len(lists) >= 1
    non_empty_list = next((lst for lst in lists if len(lst) > 0), [])
    assert len(non_empty_list) == 1
    assert isinstance(non_empty_list[0], MyFr)
    assert (non_empty_list[0].nnn, non_empty_list[0].count) == ("test", 5)


def test_read_values_list_of_models(test_session):
    """Test that read_values works with lists of DataModel instances."""
    model1 = MyFr(nnn="n1", count=1)
    model2 = MyFr(nnn="n2", count=2)
    model3 = MyFr(nnn="n3", count=3)

    chain = dc.read_values(
        items=[[model1, model2], [model3]],
        session=test_session,
    )

    # Schema should reflect list[MyFr]
    schema_ser = chain.signals_schema.serialize()
    assert schema_ser["items"].startswith("list[MyFr@v1]")

    vals = cast("list[list[MyFr]]", chain.to_values("items"))
    assert len(vals) == 2
    # Row order may vary, check both rows flexibly
    assert sorted([len(v) for v in vals]) == [1, 2]
    # Find row with 2 items to verify type
    two_item_row = next(v for v in vals if len(v) == 2)
    assert isinstance(two_item_row[0], MyFr)
    # Check all items are present across both rows
    all_counts = sorted([m.count for v in vals for m in v])
    assert all_counts == [1, 2, 3]
    all_nnns = sorted([m.nnn for v in vals for m in v])
    assert all_nnns == ["n1", "n2", "n3"]


def test_read_values_dict_of_models(test_session):
    """Test that read_values works with dicts of DataModel instances."""
    model1 = MyFr(nnn="alpha", count=10)
    model2 = MyFr(nnn="beta", count=20)

    chain = dc.read_values(
        mapping=[{"a": model1, "b": model2}],
        session=test_session,
    )

    # Schema should reflect dict[str, MyFr] (auto-detected)
    schema_ser = chain.signals_schema.serialize()
    assert schema_ser["mapping"].startswith("dict[str, MyFr@v1]")

    vals = cast("list[dict[str, MyFr]]", chain.to_values("mapping"))
    assert len(vals) == 1
    mapping = vals[0]
    assert isinstance(mapping, dict)
    assert set(mapping.keys()) == {"a", "b"}
    assert isinstance(mapping["a"], MyFr)
    assert (mapping["a"].nnn, mapping["a"].count) == ("alpha", 10)
    assert (mapping["b"].nnn, mapping["b"].count) == ("beta", 20)


def test_read_values_dict_int_keys_of_models(test_session):
    """Test that read_values works with dict[int, DataModel].

    Note: JSON stores int keys as strings, but we convert them back to int
    when reading based on the type annotation.
    """
    model1 = MyFr(nnn="first", count=100)
    model2 = MyFr(nnn="second", count=200)

    chain = dc.read_values(
        mapping=[{1: model1, 2: model2}],
        session=test_session,
    )

    # Schema should reflect dict[int, MyFr] (auto-detected)
    schema_ser = chain.signals_schema.serialize()
    assert schema_ser["mapping"].startswith("dict[int, MyFr@v1]")

    # Keys should be converted back to int based on the type annotation
    vals = cast("list[dict[int, MyFr]]", chain.to_values("mapping"))
    assert len(vals) == 1
    mapping = vals[0]
    assert isinstance(mapping, dict)
    # Keys should be int, not str
    assert set(mapping.keys()) == {1, 2}
    assert all(isinstance(k, int) for k in mapping)
    assert isinstance(mapping[1], MyFr)
    assert (mapping[1].nnn, mapping[1].count) == ("first", 100)
    assert (mapping[2].nnn, mapping[2].count) == ("second", 200)


def test_read_values_model_with_list(test_session):
    """Test read_values with DataModel containing list[int]."""
    model1 = MyWithList(name="first", values=[1, 2, 3])
    model2 = MyWithList(name="second", values=[4, 5])

    chain = dc.read_values(
        items=[model1, model2],
        session=test_session,
    )

    vals = cast("list[MyWithList]", chain.to_values("items"))
    assert len(vals) == 2
    # Row order may vary, check by finding each model by name
    names_to_models = {v.name: v for v in vals}
    assert set(names_to_models.keys()) == {"first", "second"}
    assert isinstance(names_to_models["first"], MyWithList)
    assert names_to_models["first"].values == [1, 2, 3]
    assert names_to_models["second"].values == [4, 5]


def test_read_values_model_with_model_list(test_session):
    """Test read_values with DataModel containing list[DataModel]."""
    fr1 = MyFr(nnn="a", count=1)
    fr2 = MyFr(nnn="b", count=2)
    fr3 = MyFr(nnn="c", count=3)

    model1 = MyWithModelList(title="first", items=[fr1, fr2])
    model2 = MyWithModelList(title="second", items=[fr3])

    chain = dc.read_values(
        data=[model1, model2],
        session=test_session,
    )

    vals = cast("list[MyWithModelList]", chain.to_values("data"))
    assert len(vals) == 2
    # Row order may vary, check by finding each model by title
    titles_to_models = {v.title: v for v in vals}
    assert set(titles_to_models.keys()) == {"first", "second"}
    first_model = titles_to_models["first"]
    assert isinstance(first_model, MyWithModelList)
    assert len(first_model.items) == 2
    assert isinstance(first_model.items[0], MyFr)
    assert sorted([item.nnn for item in first_model.items]) == ["a", "b"]
    assert sorted([item.count for item in first_model.items]) == [1, 2]
    second_model = titles_to_models["second"]
    assert len(second_model.items) == 1
    assert second_model.items[0].nnn == "c"


def test_read_values_model_with_dict(test_session):
    """Test read_values with DataModel containing dict[str, str]."""
    model1 = MyWithDict(name="first", metadata={"key1": "val1", "key2": "val2"})
    model2 = MyWithDict(name="second", metadata={"key3": "val3"})

    chain = dc.read_values(
        items=[model1, model2],
        session=test_session,
    )

    vals = cast("list[MyWithDict]", chain.to_values("items"))
    assert len(vals) == 2
    # Row order may vary, check by finding each model by name
    names_to_models = {v.name: v for v in vals}
    assert set(names_to_models.keys()) == {"first", "second"}
    assert isinstance(names_to_models["first"], MyWithDict)
    assert names_to_models["first"].metadata == {"key1": "val1", "key2": "val2"}
    assert names_to_models["second"].metadata == {"key3": "val3"}


def test_read_values_model_with_model_dict(test_session):
    """Test read_values with DataModel containing dict[str, DataModel]."""
    fr1 = MyFr(nnn="alpha", count=10)
    fr2 = MyFr(nnn="beta", count=20)
    fr3 = MyFr(nnn="gamma", count=30)

    model1 = MyWithModelDict(title="first", mapping={"a": fr1, "b": fr2})
    model2 = MyWithModelDict(title="second", mapping={"c": fr3})

    chain = dc.read_values(
        data=[model1, model2],
        session=test_session,
    )

    vals = cast("list[MyWithModelDict]", chain.to_values("data"))
    assert len(vals) == 2
    # Row order may vary, check by finding each model by title
    titles_to_models = {v.title: v for v in vals}
    assert set(titles_to_models.keys()) == {"first", "second"}
    first_model = titles_to_models["first"]
    assert isinstance(first_model, MyWithModelDict)
    assert len(first_model.mapping) == 2
    assert isinstance(first_model.mapping["a"], MyFr)
    assert first_model.mapping["a"].nnn == "alpha"
    assert first_model.mapping["b"].count == 20
    second_model = titles_to_models["second"]
    assert len(second_model.mapping) == 1
    assert second_model.mapping["c"].nnn == "gamma"


def test_read_values_dict_float_keys(test_session):
    """Test that read_values works with dict[float, str].

    Float keys are converted back from JSON strings based on type annotation.
    """
    chain = dc.read_values(
        data=[{1.5: "one-half", 2.7: "two-seven", 3.0: "three"}],
        session=test_session,
    )

    # Schema should reflect dict[float, str]
    schema_ser = chain.signals_schema.serialize()
    assert schema_ser["data"] == "dict[float, str]"

    vals = cast("list[dict[float, str]]", chain.to_values("data"))
    assert len(vals) == 1
    result = vals[0]
    assert isinstance(result, dict)
    assert set(result.keys()) == {1.5, 2.7, 3.0}
    assert all(isinstance(k, float) for k in result)
    assert result[1.5] == "one-half"
    assert result[2.7] == "two-seven"
    assert result[3.0] == "three"


def test_read_values_dict_bool_keys(test_session):
    """Test that read_values works with dict[bool, str].

    Bool keys are converted back from JSON strings based on type annotation.
    """
    chain = dc.read_values(
        data=[{True: "yes", False: "no"}],
        session=test_session,
    )

    # Schema should reflect dict[bool, str]
    schema_ser = chain.signals_schema.serialize()
    assert schema_ser["data"] == "dict[bool, str]"

    vals = cast("list[dict[bool, str]]", chain.to_values("data"))
    assert len(vals) == 1
    result = vals[0]
    assert isinstance(result, dict)
    assert set(result.keys()) == {True, False}
    assert all(isinstance(k, bool) for k in result)
    assert result[True] == "yes"
    assert result[False] == "no"
