import os
import pickle

import pytest

import datachain as dc
from datachain.lib.dc import DatasetPrepareError
from datachain.lib.signal_schema import SetupError
from datachain.lib.udf import Mapper


def test_udf(test_session, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    class MyMapper(Mapper):
        DEFAULT_VALUE = 84
        BOOTSTRAP_VALUE = 1452
        TEARDOWN_VALUE = 98763

        def __init__(self):
            super().__init__()
            self.value = MyMapper.DEFAULT_VALUE
            self._had_teardown = False

        def process(self, key) -> int:
            return self.value

        def setup(self):
            self.value = MyMapper.BOOTSTRAP_VALUE

        def teardown(self):
            self.value = MyMapper.TEARDOWN_VALUE

    vals = ["a", "b", "c", "d", "e", "f"]
    chain = dc.read_values(key=vals, session=test_session)

    udf = MyMapper()
    res = chain.map(res=udf).to_values("res")

    assert res == [MyMapper.BOOTSTRAP_VALUE] * len(vals)
    assert udf.value == MyMapper.TEARDOWN_VALUE


def test_no_bootstrap_for_callable(test_session):
    class MyMapper:
        def __init__(self):
            self._had_bootstrap = False
            self._had_teardown = False

        def __call__(self, key):
            return None

        def bootstrap(self):
            self._had_bootstrap = True

        def teardown(self):
            self._had_teardown = True

    udf = MyMapper()

    chain = dc.read_values(key=["a", "b", "c"], session=test_session)
    list(chain.map(res=udf).to_list())

    assert udf._had_bootstrap is False
    assert udf._had_teardown is False


def test_bootstrap_in_chain(test_session):
    base = 1278
    prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    res = list(
        dc.read_values(val=prime, session=test_session)
        .setup(init_val=lambda: base)
        .map(x=lambda val, init_val: val + init_val, output=int)
        .order_by("x")
        .to_values("x")
    )

    assert res == [base + val for val in prime]


def test_vars_duplication_error(test_session):
    with pytest.raises(DatasetPrepareError):
        (
            dc.read_values(
                val=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29], session=test_session
            )
            .setup(init_val=lambda: 11, connection=lambda: 123)
            .setup(init_val=lambda: 599)
        )


def test_setup_error_is_picklable():
    err = SetupError("client", "error when call function: 'conn refused'")
    restored = pickle.loads(pickle.dumps(err))  # noqa: S301
    assert str(restored) == str(err)
    assert type(restored) is SetupError


def test_setup_error_message(test_session):
    def failing_setup():
        raise RuntimeError("connection refused")

    def identity(val, client):
        return val

    chain = (
        dc.read_values(val=[1, 2, 3], session=test_session)
        .setup(client=failing_setup)
        .map(res=identity, output=int)
    )
    with pytest.raises(SetupError, match="cannot setup value 'client'"):
        chain.to_list()


@pytest.mark.xdist_group(name="tmpfile")
def test_setup_error_message_parallel(test_session_tmpfile, capfd):
    test_pid = os.getpid()

    def bad_setup():
        if os.getpid() == test_pid:
            raise AssertionError("setup ran in the test process, not in parallel")
        raise RuntimeError("connection refused")

    def noop(val, client):
        return val

    chain = (
        dc.read_values(val=list(range(100)), session=test_session_tmpfile)
        .setup(client=bad_setup)
        .settings(parallel=2)
        .map(res=noop, output=int)
    )
    with pytest.raises(RuntimeError, match="UDF Execution Failed!"):
        chain.to_list()
    captured = capfd.readouterr()
    assert "cannot setup value 'client'" in captured.err
    assert "connection refused" in captured.err
