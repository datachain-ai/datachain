import pickle
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

import datachain as dc
from datachain.lib.convert.flatten import iter_flat_columns
from datachain.lib.file import File, FileRow
from datachain.lib.udf import UdfError
from datachain.query import dataset


def test_file_row_fields_are_the_flat_columns_of_file():
    assert FileRow._fields == tuple(
        "__".join(c.path) for c in iter_flat_columns(dc.File)
    )
    assert (
        FileRow("s3://bucket", "a.txt")[2:]
        == tuple(dc.File(source="s3://bucket", path="a.txt").model_dump().values())[2:]
    )


def test_file_rows_pickle():
    row = FileRow("s3://bucket", "a.txt")
    assert pickle.loads(pickle.dumps(row)) == row  # noqa: S301


def _stored(chain: dc.DataChain, session) -> list[tuple]:
    ds = chain.dataset
    assert ds is not None
    warehouse = session.catalog.warehouse
    table = warehouse.dataset_rows(ds, ds.latest_version).get_table()
    cols = [c for c in table.c if not c.name.startswith("sys__")]
    rows = warehouse.db.execute(sa.select(*cols))
    return sorted(((*r, *map(type, r)) for r in rows), key=repr)


def _files(n: int, as_rows: bool) -> Iterator[dc.File]:
    when = datetime(2026, 10, 13, 1, 2, 3, 456789, tzinfo=UTC)
    for i in range(n):
        args = ("s3://bucket", f"{n}/{i}", i, "", f"e{i}", bool(i % 2), when)
        if as_rows and i != 1:
            yield FileRow(*args)  # type: ignore[misc]
        else:
            yield dc.File(**dict(zip(FileRow._fields, args, strict=False)))


@pytest.mark.parametrize("file_type", [dc.File, dc.ImageFile])
def test_gen_file_rows_are_stored_as_files(test_session, file_type):
    chain = dc.read_values(n=[0, 1, 3], session=test_session)

    rows = chain.gen(file=lambda n: _files(n, True), output=file_type).save("rows")
    files = chain.gen(file=lambda n: _files(n, False), output=file_type).save("files")

    assert _stored(rows, test_session) == _stored(files, test_session)
    assert {type(f) for f in rows.to_values("file")} == {file_type}


def test_gen_file_rows_skip_conversion(test_session, mocker):
    spy = mocker.spy(dataset, "adjust_outputs")

    dc.read_values(n=[3], session=test_session).gen(
        file=lambda n: _files(n, True), output=dc.File
    ).save("files")

    files = [c for c in spy.call_args_list if "file__path" in c.args[1]]
    assert len(files) == 1  # the File object among the rows


def test_file_row_is_no_optional_file(test_session):
    def rows(n: int) -> Iterator[dc.File | None]:
        yield FileRow("s3://bucket", f"{n}")  # type: ignore[misc]

    chain = dc.read_values(n=[1], session=test_session).gen(
        file=rows, output=dc.File | None
    )

    with pytest.raises(UdfError, match="returned 8 values while 9 are expected"):
        chain.save("optional")


def test_file_validators_are_known():
    # FileRow rows skip File's validators: a new one must hold for listing rows too
    validators = dc.File.__pydantic_decorators__
    assert set(validators.field_validators) == {"validate_location", "validate_source"}
    assert set(validators.model_validators) == {"_normalize_path"}


def test_file_rows_take_file_attributes():
    row = FileRow("s3://bucket", "dir/a.txt", 3)

    assert row.get_fs_path() == "s3://bucket/dir/a.txt"
    assert row.name == "a.txt"
    assert row.get_file_ext() == "txt"
    with pytest.raises(AttributeError):
        row.missing  # noqa: B018


def test_file_rows_only_take_path_helpers():
    row = FileRow("s3://bucket", "dir/a.txt")

    assert row.parent == "dir"
    assert not hasattr(row, "read")
    assert not hasattr(row, "model_dump")
    with pytest.raises(AttributeError):  # rows read no files: see to_objects()
        row.open()


def test_invalid_file_rows_have_path_helpers_only():
    row = FileRow("no-scheme", "d/a.txt")

    assert (row.name, row.parent, row.get_file_ext()) == ("a.txt", "d", "txt")
    with pytest.raises(ValidationError):  # it needs a valid File
        row.get_fs_path()


@pytest.mark.parametrize("path", ["d/e.tar.gz", "e", "d//e.txt", "a/b/.hidden", ""])
def test_file_row_path_helpers_match_file(path):
    row = FileRow("s3://b", path)
    file = File(source="s3://b", path=path)

    assert (row.name, row.parent) == (file.name, file.parent)
    for helper in ("get_file_ext", "get_file_stem", "get_file_suffix"):
        assert getattr(row, helper)() == getattr(file, helper)()
