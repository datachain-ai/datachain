import pytest

import datachain as dc
from datachain import Column, func
from datachain.func import path as pathfunc
from datachain.lib.data_model import DataModel
from datachain.lib.file import File
from datachain.lib.signal_schema import SignalResolvingError
from datachain.query.dataset import DatasetQuery


def _create_test_files(names_and_sizes):
    """Helper function to create test files with given names and sizes."""
    return [File(path=name, size=size) for name, size in names_and_sizes]


def test_mutate_overwrite_twice(test_session):
    files = _create_test_files(
        [("file1.txt", 100), ("file2.txt", 200), ("file3.txt", 300)]
    )
    ds = dc.read_values(file=files, session=test_session)

    orig_sizes = ds.order_by("file.path").to_values("file.size")
    mutated_ds = ds.mutate(file__size=Column("file.size") + 1).mutate(
        file__size=Column("file.size") + 1
    )

    result_sizes = mutated_ds.order_by("file.path").to_values("file.size")
    expected = [s + 2 for s in orig_sizes]
    assert result_sizes == expected


def test_mutate_overwrite_then_order_by(test_session):
    files = _create_test_files(
        [("small.txt", 50), ("medium.txt", 100), ("large.txt", 200), ("huge.txt", 500)]
    )
    ds = dc.read_values(file=files, session=test_session)
    mutated_ds = ds.mutate(file__size=0 - Column("file.size"))
    rows = mutated_ds.order_by("file.size").to_list()

    sizes = [row[0].size for row in rows]
    assert sizes == sorted(sizes)
    assert len(rows) == 4

    expected_paths = ["huge.txt", "large.txt", "medium.txt", "small.txt"]
    actual_paths = [row[0].path for row in rows]
    assert actual_paths == expected_paths


def test_mutate_multiple_columns(test_session):
    files = _create_test_files([("dir/file1.txt", 100), ("dir/file2.txt", 200)])
    ds = dc.read_values(file=files, session=test_session)
    result = ds.mutate(
        size_doubled=Column("file.size") * 2,
        filename=pathfunc.name(Column("file.path")),
        is_large=True,
    )

    rows = result.order_by("file.path").to_list()
    assert rows[0][1] == 200
    assert rows[0][2] == "file1.txt"
    assert rows[0][3]

    assert rows[1][1] == 400
    assert rows[1][2] == "file2.txt"
    assert rows[1][3]


def test_mutate_references_new_column_in_same_call(test_session):
    ds = dc.read_values(path=["a/b", "root"], session=test_session)

    with pytest.raises(SignalResolvingError):
        ds.mutate(
            path_parts=func.string.split("path", "/"),
            path_depth=func.array.length("path_parts"),
        )


def test_mutate_chaining_with_different_operations(test_session):
    files = _create_test_files(
        [("test1.txt", 10), ("test2.txt", 20), ("test3.txt", 30)]
    )
    ds = dc.read_values(file=files, session=test_session)
    result = (
        ds.mutate(doubled=Column("file.size") * 2)
        .mutate(added=Column("doubled") + 5)
        .mutate(final=Column("added") * 10)
    )

    expected_values = [250, 450, 650]  # (size*2+5)*10

    actual_values = result.order_by("file.path").to_values("final")
    assert actual_values == expected_values


def test_mutate_existing_column(test_session):
    ds = dc.read_values(ids=[1, 2, 3], session=test_session)
    ds = ds.mutate(ids=Column("ids") + 1)
    assert ds.order_by("ids").to_list() == [(2,), (3,), (4,)]


def test_mutate_with_primitives_save_load(test_session):
    original_data = [1, 2, 3]
    ds = dc.read_values(data=original_data, session=test_session).mutate(
        str_col="test_string",
        int_col=42,
        float_col=3.14,
        bool_col=True,
    )

    schema = ds.signals_schema.values
    assert schema.get("str_col") is str
    assert schema.get("int_col") is int
    assert schema.get("float_col") is float
    assert schema.get("bool_col") is bool

    ds.save("test_mutate_primitives")
    loaded_ds = dc.read_dataset("test_mutate_primitives", session=test_session)

    loaded_schema = loaded_ds.signals_schema.values
    assert loaded_schema.get("str_col") is str
    assert loaded_schema.get("int_col") is int
    assert loaded_schema.get("float_col") is float
    assert loaded_schema.get("bool_col") is bool

    results = set(loaded_ds.to_list())
    expected_results = {
        (1, "test_string", 42, 3.14, True),
        (2, "test_string", 42, 3.14, True),
        (3, "test_string", 42, 3.14, True),
    }
    assert len(results) == 3
    assert results == expected_results


def test_persist_after_mutate(test_session):
    chain = (
        dc.read_values(fib=[1, 1, 2, 3, 5, 8, 13, 21], session=test_session)
        .map(mod3=lambda fib: fib % 3, output=int)
        .group_by(cnt=dc.func.count(), partition_by="mod3")
        .mutate(x=1)
        .persist()
    )

    assert chain.count() == 3
    assert set(chain.to_values("mod3")) == {0, 1, 2}


def test_mutate_rename_column_no_db_leakage(test_session):
    files = _create_test_files(
        [("file1.txt", 100), ("file2.txt", 200), ("file3.txt", 300)]
    )
    ds = dc.read_values(file=files, session=test_session)
    renamed_ds = ds.mutate(new_file=Column("file"))
    renamed_ds.save("test_rename_column")

    loaded_ds = dc.read_dataset("test_rename_column", session=test_session)

    # Check actual database records using DatasetQuery
    query = DatasetQuery("test_rename_column", catalog=test_session.catalog)
    db_records = query.limit(5).to_db_records()
    db_columns = set(db_records[0].keys())
    file_columns = {col for col in db_columns if col.startswith("file__")}

    assert len(file_columns) == 0, f"Found leaked old file__ columns: {file_columns}"

    assert len(loaded_ds.to_list()) == 3

    paths = loaded_ds.order_by("new_file.path").to_values("new_file.path")
    assert paths == ["file1.txt", "file2.txt", "file3.txt"]


def test_mutate_with_window_functions(test_session):
    """Test mutate with window functions"""

    files = _create_test_files(
        [
            ("cats/cat1", 4),
            ("cats/cat2", 4),
            ("dogs/dog1", 4),
            ("dogs/dog2", 3),
            ("dogs/dog3", 4),
            ("dogs/others/dog4", 4),
            ("description", 13),
        ]
    )

    class FileInfo(DataModel):
        path: str = ""
        name: str = ""

    def file_info(file: File) -> FileInfo:
        path_parts = file.path.split("/", 1)
        return FileInfo(
            path=path_parts[0] if len(path_parts) > 1 else "",
            name=path_parts[1] if len(path_parts) > 1 else path_parts[0],
        )

    window = func.window(
        partition_by="file_info.path", order_by="file_info.name", desc=True
    )

    ds = (
        dc.read_values(file=files, session=test_session)
        .settings(prefetch=False)
        .map(file_info, params=["file"], output={"file_info": FileInfo})
        .mutate(row_number=func.row_number().over(window))
        .filter(dc.C("row_number") < 3)
        .select_except("row_number")
        .save("test-window-mutate")
    )

    results = {}
    for r in ds.to_records():
        results.setdefault(r["file_info__path"], []).append(r["file_info__name"])

    assert results[""] == ["description"]  # Only one file in root
    assert sorted(results["cats"]) == sorted(["cat1", "cat2"])  # Both cats files

    assert len(results["dogs"]) == 2
    all_dogs = ["dog1", "dog2", "dog3", "others/dog4"]
    for dog in results["dogs"]:
        assert dog in all_dogs
        all_dogs.remove(dog)
    assert len(all_dogs) == 2


def test_mutate_keeps_nested_column_on_rename(test_session):
    files = _create_test_files([("1.flac", 0), ("2.flac", 0), ("3.flac", 0)])

    ds = (
        dc.read_values(file=files, session=test_session)
        .mutate(tmp=Column("file.path"))
        .save("test_nested_keep_schema")
    )

    results = ds.order_by("file.path").to_list()  # file still exists
    assert len(results) == 3
    assert results[0][1] == "1.flac"  # tmp column should have file path


def test_mutate_nested_column_same_type(test_session):
    files = _create_test_files([("1.flac", 0), ("2.flac", 0), ("3.flac", 0)])

    (
        dc.read_values(file=files, session=test_session)
        .mutate(file__path="aaa")
        .save("test_nested_same_type")
    )

    results = dc.read_dataset("test_nested_same_type").to_list()
    assert len(results) == 3
    for row in results:
        assert row[0].path == "aaa"


def test_mutate_nested_column_type_change(test_session):
    files = _create_test_files([("1.flac", 0), ("2.flac", 0), ("3.flac", 0)])

    expected_msg = "Altering nested column type is not allowed"
    with pytest.raises(ValueError, match=expected_msg):
        dc.read_values(file=files, session=test_session).mutate(file__path=123).save(
            "test_nested_type_change"
        )


def test_mutate_nested_column_complex_mutation(test_session):
    files = _create_test_files([("1.flac", 0), ("2.flac", 0), ("3.flac", 0)])

    ds = (
        dc.read_values(file=files, session=test_session)
        .mutate(tmp=func.string.replace(Column("file.path"), ".flac", ".mp3"))
        .mutate(file__path=Column("tmp"))
        .save("test_nested_complex")
    )

    schema = ds.signals_schema.values
    assert "file" in schema
    assert "file__path" not in schema
    assert "tmp" not in schema

    results = ds.order_by("file.path").to_list()
    assert len(results) == 3
    paths = [row[0].path for row in results]
    assert paths == ["1.mp3", "2.mp3", "3.mp3"]


def test_mutate_reject_new_nested_columns(test_session):
    files = _create_test_files([("1.flac", 0), ("2.flac", 0), ("3.flac", 0)])

    expected_msg = "Creating new nested columns directly is not allowed"
    with pytest.raises(ValueError, match=expected_msg):
        dc.read_values(file=files, session=test_session).mutate(
            something__new=Column("file.source")
        ).save("test_reject_nested")
