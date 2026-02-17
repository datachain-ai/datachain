import numpy as np
import pytest
from pydantic import BaseModel

import datachain as dc
from datachain import File, func
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalRemoveError, SignalResolvingTypeError


class MyFr(BaseModel):
    nnn: str
    count: int


class MyNested(BaseModel):
    label: str
    fr: MyFr


_features = sorted(
    [MyFr(nnn="n1", count=3), MyFr(nnn="n2", count=5), MyFr(nnn="n1", count=1)],
    key=lambda f: (f.nnn, f.count),
)
_features_nested = [
    MyNested(fr=fr, label=f"label_{num}") for num, fr in enumerate(_features)
]


def test_select_read_values_without_sys_columns(test_session):
    chain = (
        dc.read_values(
            name=["a", "a", "b", "b", "b", "c"],
            val=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            session=test_session,
        )
        .group_by(cnt=func.count(), partition_by="name")
        .order_by("name")
        .select("name", "cnt")
    )

    assert chain.to_records() == [
        {"name": "a", "cnt": 2},
        {"name": "b", "cnt": 3},
        {"name": "c", "cnt": 1},
    ]


def test_select_feature(test_session):
    chain = dc.read_values(my_n=_features_nested, session=test_session)
    ordered = chain.order_by("my_n.fr.nnn", "my_n.fr.count")

    samples = ordered.select("my_n").to_list()
    for idx, sample in enumerate(samples):
        assert sample[0] == _features_nested[idx]

    samples = ordered.select("my_n.fr").to_list()
    for idx, sample in enumerate(samples):
        assert sample[0].fr == _features[idx]

    samples = ordered.select("my_n.label", "my_n.fr.count").to_list()
    for idx, sample in enumerate(samples):
        my_n = sample[0]
        assert my_n.label == _features_nested[idx].label
        assert my_n.fr.count == _features_nested[idx].fr.count


def test_select_columns_intersection(test_session):
    chain = dc.read_values(my_n=_features_nested, session=test_session)

    samples = (
        chain.order_by("my_n.fr.nnn", "my_n.fr.count")
        .select("my_n.fr", "my_n.fr.count")
        .to_list()
    )
    for idx, sample in enumerate(samples):
        my_n = sample[0]
        assert my_n.fr == _features_nested[idx].fr
        assert my_n.fr.count == _features_nested[idx].fr.count


def test_select_except(test_session):
    chain = dc.read_values(fr1=_features_nested, fr2=_features, session=test_session)

    samples = (
        chain.order_by("fr1.fr.nnn", "fr1.fr.count").select_except("fr2").to_list()
    )
    for idx, sample in enumerate(samples):
        (fr,) = sample
        assert fr.label == _features_nested[idx].label
        assert fr.fr == _features_nested[idx].fr


def test_select_except_after_gen(test_session):
    # https://github.com/datachain-ai/datachain/issues/1359
    chain = dc.read_values(id=range(10), session=test_session)

    chain = chain.gen(lambda id: [(id, 0)], output={"id": int, "x": int})
    chain = chain.select_except("x")
    chain = chain.merge(chain, on="id")
    chain = chain.select_except("right_id")

    assert set(chain.to_values("id")) == set(range(10))


def test_select_wrong_type(test_session):
    chain = dc.read_values(fr1=_features_nested, fr2=_features, session=test_session)

    with pytest.raises(SignalResolvingTypeError):
        chain.select(4).to_list()

    with pytest.raises(SignalResolvingTypeError):
        chain.select_except(_features[0]).to_list()


def test_select_except_error(test_session):
    chain = dc.read_values(fr1=_features_nested, fr2=_features, session=test_session)

    with pytest.raises(SignalRemoveError):
        chain.select_except("not_exist", "file").to_list()

    with pytest.raises(SignalRemoveError):
        chain.select_except("fr1.label", "file").to_list()


def test_select_restore_from_saving(test_session):
    chain = dc.read_values(my_n=_features_nested, session=test_session)

    name = "test_select_restore_from_saving"
    chain.select("my_n.fr").save(name)

    restored = dc.read_dataset(name)
    restored_sorted = sorted(restored.to_list(), key=lambda x: x[0].fr.count)
    features_sorted = sorted(_features, key=lambda x: x.count)

    for idx, sample in enumerate(restored_sorted):
        assert sample[0].fr == features_sorted[idx]


def test_select_distinct(test_session):
    class Embedding(BaseModel):
        id: int
        filename: str
        values: list[float]

    expected = [
        [0.1, 0.3],
        [0.1, 0.4],
        [0.1, 0.5],
        [0.1, 0.6],
    ]

    actual = (
        dc.read_values(
            embedding=[
                Embedding(id=1, filename="a.jpg", values=expected[0]),
                Embedding(id=2, filename="b.jpg", values=expected[2]),
                Embedding(id=3, filename="c.jpg", values=expected[1]),
                Embedding(id=4, filename="d.jpg", values=expected[1]),
                Embedding(id=5, filename="e.jpg", values=expected[3]),
            ],
            session=test_session,
        )
        .select("embedding.values", "embedding.filename")
        .distinct("embedding.values")
        .order_by("embedding.values")
        .to_list()
    )

    values = [row[0].values for row in actual]
    assert len(values) == 4

    values_sorted = sorted(values)
    expected_sorted = sorted(expected)
    for got, exp in zip(values_sorted, expected_sorted, strict=True):
        assert np.allclose(got, exp)


def test_select_nested_creates_partial_model(test_session):
    chain = dc.read_values(
        file=[File(path="a.txt", source="file://")],
        session=test_session,
    )

    selected = chain.select("file.path")

    assert set(selected.schema.keys()) == {"file"}
    file_type = selected.schema["file"]
    assert ModelStore.is_partial(file_type)

    (file_obj,) = selected.to_list()[0]
    assert file_obj.path == "a.txt"


def test_select_preserves_sys_columns(test_session):
    chain = dc.read_values(name=["a"], session=test_session)

    selected = chain.select("name")

    # sys columns are hidden by default, but should remain available.
    selected_with_sys = selected.settings(sys=True)
    sys_ids = selected_with_sys.to_values("sys.id")
    assert len(sys_ids) == 1
    assert isinstance(sys_ids[0], int)


def test_select_no_args_returns_self(test_session):
    chain = dc.read_values(name=["a", "b"], session=test_session)
    before_hash = chain.hash()

    selected = chain.select()

    assert selected is chain
    assert selected.hash() == before_hash


def test_select_except_no_args_returns_self(test_session):
    chain = dc.read_values(name=["a", "b"], session=test_session)
    before_hash = chain.hash()

    selected = chain.select_except()

    assert selected is chain
    assert selected.hash() == before_hash


def test_to_values_and_to_iter_keep_flattening(test_session):
    chain = dc.read_values(
        file=[File(path="a.txt", source="file://")],
        session=test_session,
    )

    assert chain.to_values("file.path") == ["a.txt"]
    assert list(chain.to_iter("file.path")) == [("a.txt",)]


def test_select_except_nested_excludes_field_via_partial(test_session):
    chain = dc.read_values(
        file=[File(path="a.txt", source="file://", location={})],
        session=test_session,
    )

    selected = chain.select_except("file.path")

    assert set(selected.schema.keys()) == {"file"}
    file_type = selected.schema["file"]
    assert ModelStore.is_partial(file_type)

    (file_obj,) = selected.to_list()[0]
    assert not hasattr(file_obj, "path")
