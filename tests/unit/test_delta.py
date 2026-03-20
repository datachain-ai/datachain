import pytest

import datachain as dc
from datachain.delta import delta_disabled
from datachain.lib.dc import C


def _delta_unsafe_required_message(method_name: str) -> str:
    return (
        f"Cannot use {method_name} with delta datasets - may cause inconsistency."
        " Set delta_unsafe=True on every participating delta source"
        " to allow this operation."
    )


def test_delta_disabled_ignores_non_datachain_args(test_session):
    @delta_disabled
    def guarded(self, *args, **kwargs):
        return self, args, kwargs

    chain = dc.read_values(id=[1], session=test_session)

    result_self, result_args, result_kwargs = guarded(
        chain,
        C("id"),
        "id",
        123,
        flag=True,
        partition_by=C("id"),
    )

    assert result_self is chain
    assert result_args[0].name == "id"
    assert result_args[1:] == ("id", 123)
    assert result_kwargs["flag"] is True
    assert result_kwargs["partition_by"].name == "id"


def test_delta_disabled_blocks_delta_datachain_argument(test_session):
    @delta_disabled
    def guarded(self, other, *args, **kwargs):
        return self, other, args, kwargs

    left = dc.read_values(id=[1], session=test_session)
    dc.read_values(id=[1], session=test_session).save("decorator_delta_source")
    right = dc.read_dataset(
        "decorator_delta_source",
        session=test_session,
        delta=True,
        delta_on="id",
    )

    with pytest.raises(NotImplementedError) as excinfo:
        guarded(left, right, C("id"), on="id")

    assert str(excinfo.value) == _delta_unsafe_required_message("guarded")


def test_delta_disabled_rejects_non_datachain_self():
    @delta_disabled
    def guarded(self, *args, **kwargs):
        return self, args, kwargs

    with pytest.raises(TypeError) as excinfo:
        guarded(object())

    assert str(excinfo.value).startswith(
        "delta_disabled can only wrap DataChain methods"
    )
