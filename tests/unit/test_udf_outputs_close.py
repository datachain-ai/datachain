from collections.abc import Iterator

import pytest

import datachain as dc


def test_an_insert_error_closes_the_udf_outputs(test_session, mocker):
    closed = []

    def numbers(n: int) -> Iterator[int]:
        try:
            yield from range(n)
        finally:
            closed.append(n)

    def failing_insert(self, table, columns, rows, batch_size=None):
        next(iter(rows))
        raise OSError(28, "No space left on device")

    chain = dc.read_values(n=[1000], session=test_session).save("inputs")
    mocker.patch.object(
        type(test_session.catalog.warehouse), "insert_tuples", failing_insert
    )
    with pytest.raises(OSError, match="No space left") as error:
        chain.gen(x=numbers, output=int).save("numbers")
    assert closed == [1000]  # closed while the error, as a notebook holds it, lives
    assert error.value.__traceback__ is not None
