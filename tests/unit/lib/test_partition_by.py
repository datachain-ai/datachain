from collections.abc import Iterator

import pytest
from pydantic import BaseModel

import datachain as dc
from datachain import func
from datachain.lib.data_model import DataModel
from datachain.lib.file import File
from datachain.lib.signal_schema import SignalResolvingError
from datachain.lib.utils import DataChainColumnError


def test_complex_signal_partition_by_file(test_session):
    """Test partitioning by File objects using column names."""
    files = [
        File(source="s3://bucket", path="file1.txt", size=100),
        File(source="s3://bucket", path="file2.txt", size=200),
        File(source="s3://bucket", path="file1.txt", size=100),  # duplicate
        File(source="s3://bucket", path="file3.txt", size=300),
    ]
    amounts = [10, 20, 30, 40]

    # Create a chain with File objects
    chain = dc.read_values(
        file=files,
        amount=amounts,
        session=test_session,
    )

    def my_agg(files: list[File], amounts: list[int]) -> Iterator[tuple[File, int]]:
        # Group files and sum amounts
        yield files[0], sum(amounts)

    # Test partitioning by File column name (automatically expands to unique keys)
    result = chain.agg(
        my_agg,
        params=("file", "amount"),
        output={"file": File, "total": int},
        partition_by="file",
    ).to_list("file", "total")

    # We should have 3 groups (file1.txt appears twice, so should be grouped)
    assert len(result) == 3

    # Check that files with same unique attributes are grouped together
    assert {f[0].path for f in result} == {"file1.txt", "file2.txt", "file3.txt"}

    # Check total amounts - create mapping from path to total
    path_to_total = {f.path: total for f, total in result}
    assert path_to_total["file1.txt"] == 40  # 10 + 30 (grouped)
    assert path_to_total["file2.txt"] == 20
    assert path_to_total["file3.txt"] == 40


def test_complex_signal_partition_by_mixed(test_session):
    """Test partitioning by mixed types (complex signal column and string)."""
    files = [
        File(source="s3://bucket", path="file1.txt", size=100),
        File(source="s3://bucket", path="file2.txt", size=200),
        File(source="s3://bucket", path="file1.txt", size=100),  # duplicate
    ]
    categories = ["A", "B", "A"]
    amounts = [10, 20, 30]

    chain = dc.read_values(
        file=files,
        category=categories,
        amount=amounts,
        session=test_session,
    )

    def my_agg(
        files: list[File], categories: list[str], amounts: list[int]
    ) -> Iterator[tuple[File, str, int]]:
        yield files[0], categories[0], sum(amounts)

    # Test partitioning by both File column and string column
    result = chain.agg(
        my_agg,
        params=("file", "category", "amount"),
        output={"file": File, "category": str, "total": int},
        partition_by=("file", "category"),
    ).to_list("file", "category", "total")

    # We should have 2 groups: (file1.txt, A), (file2.txt, B)
    assert len(result) == 2

    # Check grouping by both file and category
    groups = {(f.path, cat): total for f, cat, total in result}
    assert groups[("file1.txt", "A")] == 40  # 10 + 30
    assert groups[("file2.txt", "B")] == 20


def test_complex_signal_partition_by_error_handling(test_session):
    """Test error handling for invalid column names."""
    chain = dc.read_values(
        value=[1, 2, 3],
        session=test_session,
    )

    def my_agg(values: list[int]) -> Iterator[tuple[int]]:
        yield (sum(values),)

    # Test with non-existent column name
    with pytest.raises(
        SignalResolvingError,
        match="cannot resolve signal name 'nonexistent_column': is not found",
    ):
        chain.agg(
            my_agg,
            params=("value",),
            output={"total": int},
            partition_by="nonexistent_column",
        ).to_records()


def test_complex_signal_partition_by_not_in_schema(test_session):
    """Test error handling when column name is not in schema."""
    chain = dc.read_values(
        value=[1, 2, 3],
        session=test_session,
    )

    def my_agg(values: list[int]) -> Iterator[tuple[int]]:
        yield (sum(values),)

    # Test with column name not in schema
    with pytest.raises(
        SignalResolvingError, match="cannot resolve signal name 'file': is not found"
    ):
        chain.agg(
            my_agg,
            params=("value",),
            output={"total": int},
            partition_by="file",  # file column is not in this schema
        ).to_records()


def test_complex_signal_group_by_file(test_session):
    """Test group_by with File objects using column names."""
    files = [
        File(source="s3://bucket", path="file1.txt", size=100),
        File(source="s3://bucket", path="file2.txt", size=200),
        File(source="s3://bucket", path="file1.txt", size=100),  # duplicate
        File(source="s3://bucket", path="file3.txt", size=300),
    ]
    amounts = [10, 20, 30, 40]

    # Create a chain with File objects
    chain = dc.read_values(
        file=files,
        amount=amounts,
        session=test_session,
    )

    # Test group_by with File column name (automatically expands to unique keys)
    result = chain.group_by(
        total=dc.func.sum("amount"),
        count=dc.func.count(),
        partition_by="file",
    ).to_list("file", "total", "count")

    # We should have 3 groups (file1.txt appears twice, so should be grouped)
    assert len(result) == 3

    # Check that files with same unique attributes are grouped together
    assert {f[0].path for f in result} == {"file1.txt", "file2.txt", "file3.txt"}

    # Check total amounts and counts
    groups = {f.path: (total, count) for f, total, count in result}
    assert groups["file1.txt"] == (40, 2)  # 10 + 30 (2 groups)
    assert groups["file2.txt"] == (20, 1)
    assert groups["file3.txt"] == (40, 1)


def test_complex_signal_group_by_mixed(test_session):
    """Test group_by with mixed types (complex signal column and string)."""
    files = [
        File(source="s3://bucket", path="file1.txt", size=100),
        File(source="s3://bucket", path="file2.txt", size=200),
        File(source="s3://bucket", path="file1.txt", size=100),  # duplicate
    ]
    categories = ["A", "B", "A"]
    amounts = [10, 20, 30]

    chain = dc.read_values(
        file=files,
        category=categories,
        amount=amounts,
        session=test_session,
    )

    # Test group_by with both File column and string column
    result = chain.group_by(
        total=dc.func.sum("amount"),
        count=dc.func.count(),
        partition_by=["file", "category"],
    ).to_list("file", "category", "total")

    # We should have 2 groups: (file1.txt, A), (file2.txt, B)
    assert len(result) == 2

    # Check grouping by both file and category
    groups = {(f.path, category): total for f, category, total in result}
    assert groups[("file1.txt", "A")] == 40  # 10 + 30
    assert groups[("file2.txt", "B")] == 20


def test_complex_signal_deep_nesting(test_session):
    class NestedLevel1(BaseModel):
        name: str
        value: int

    class NestedLevel2(BaseModel):
        category: str
        level1: NestedLevel1

    class NestedLevel3(BaseModel):
        id: str
        level2: NestedLevel2
        total: float

    nested_data = [
        NestedLevel3(
            id="item1",
            level2=NestedLevel2(
                category="A", level1=NestedLevel1(name="test1", value=10)
            ),
            total=100.0,
        ),
        NestedLevel3(
            id="item2",
            level2=NestedLevel2(
                category="B", level1=NestedLevel1(name="test2", value=20)
            ),
            total=200.0,
        ),
        NestedLevel3(
            id="item1",  # Same as first item
            level2=NestedLevel2(
                category="A", level1=NestedLevel1(name="test1", value=10)
            ),
            total=100.0,
        ),
    ]
    amounts = [10, 20, 30]

    chain = dc.read_values(
        nested=nested_data,
        amount=amounts,
        session=test_session,
    )

    result = chain.group_by(
        total_amount=dc.func.sum("amount"),
        count=dc.func.count(),
        partition_by="nested",
    )

    assert dict(result.to_list("nested.id", "total_amount")) == {
        "item1": 40,
        "item2": 20,
    }

    assert dict(result.to_list("nested.id", "count")) == {
        "item1": 2,
        "item2": 1,
    }


def test_nested_column_partition_by(test_session):
    class Level1(BaseModel):
        name: str
        value: int

    class Level2(BaseModel):
        category: str
        level1: Level1

    nested_data = [
        Level2(category="A", level1=Level1(name="test1", value=10)),
        Level2(category="B", level1=Level1(name="test2", value=20)),
        Level2(
            category="A",
            level1=Level1(name="test1", value=10),  # Same as first
        ),
        Level2(
            category="A",
            level1=Level1(name="test3", value=30),  # Different name
        ),
    ]
    amounts = [10, 20, 30, 40]

    chain = dc.read_values(
        nested=nested_data,
        amount=amounts,
        session=test_session,
    )

    # Test partition_by with nested column reference
    result = chain.group_by(
        total=dc.func.sum("amount"),
        count=dc.func.count(),
        partition_by="nested.level1.name",  # This should work
    ).to_list("nested.level1.name", "total")

    assert len(result) == 3  # Should have 3 unique names: test1, test2, test3

    # Check the grouped results
    name_to_total = dict(result)
    assert name_to_total["test1"] == 40  # 10 + 30 (grouped by name)
    assert name_to_total["test2"] == 20
    assert name_to_total["test3"] == 40


def test_nested_column_agg_partition_by(test_session):
    class Person(BaseModel):
        name: str
        age: int

    class Team(BaseModel):
        name: str
        leader: Person

    teams = [
        Team(name="Alpha", leader=Person(name="Alice", age=30)),
        Team(name="Beta", leader=Person(name="Bob", age=25)),
        Team(name="Alpha", leader=Person(name="Alice", age=30)),  # Same team/leader
        Team(
            name="Gamma", leader=Person(name="Alice", age=30)
        ),  # Same leader, different team
    ]
    scores = [100, 200, 150, 300]

    chain = dc.read_values(
        team=teams,
        score=scores,
        session=test_session,
    )

    def my_agg(teams: list[Team], scores: list[int]) -> Iterator[tuple[Team, int]]:
        yield teams[0], sum(scores)

    result = chain.agg(
        my_agg,
        params=("team", "score"),
        output={"team": Team, "total": int},
        partition_by="team.leader.name",  # Partition by leader name
    ).to_list("team.leader.name", "total")

    assert len(result) == 2  # Should have 2 unique leaders: Alice, Bob

    leader_to_total = dict(result)
    assert leader_to_total["Alice"] == 550  # 100 + 150 + 300 (all Alice-led teams)
    assert leader_to_total["Bob"] == 200


def test_nested_column_edge_cases(test_session):
    from datachain.lib.signal_schema import SignalResolvingError

    class Simple(BaseModel):
        name: str
        value: int

    simple_data = [
        Simple(name="test1", value=10),
        Simple(name="test2", value=20),
        Simple(name="test1", value=30),
    ]
    amounts = [10, 20, 30]

    chain = dc.read_values(
        simple=simple_data,
        amount=amounts,
        session=test_session,
    )

    result = chain.group_by(
        total=dc.func.sum("amount"),
        count=dc.func.count(),
        partition_by="simple.name",
    ).to_list("simple.name", "total")

    assert len(result) == 2  # Should have 2 unique names

    name_to_total = dict(result)
    assert name_to_total["test1"] == 40  # 10 + 30
    assert name_to_total["test2"] == 20

    with pytest.raises(SignalResolvingError):
        chain.group_by(
            total=dc.func.sum("amount"),
            count=dc.func.count(),
            partition_by="simple.nonexistent",
        ).to_records()

    with pytest.raises(SignalResolvingError):
        chain.group_by(
            total=dc.func.sum("amount"),
            count=dc.func.count(),
            partition_by="nonexistent.field",
        ).to_records()


def test_group_by_with_functions_in_partition_by(test_session):
    class CustomFile(DataModel):
        path: str
        size: int

    custom_data = [
        CustomFile(path="docs/readme.txt", size=100),
        CustomFile(path="docs/guide.txt", size=200),
        CustomFile(path="src/main.py", size=300),
        CustomFile(path="src/utils.py", size=150),
        CustomFile(path="tests/test_main.py", size=250),
        CustomFile(path="config.yaml", size=50),
    ]

    ds = dc.read_values(
        custom_file=custom_data,
        session=test_session,
    ).group_by(
        cnt=func.count(),
        sum=func.sum("custom_file.size"),
        partition_by=func.path.parent("custom_file.path").label("file_dir"),
    )

    assert len(ds.to_list("file_dir")) == 4
    assert ds.filter(dc.C("file_dir") == "docs").to_list("cnt", "sum")[0] == (2, 300)
    assert ds.filter(dc.C("file_dir") == "src").to_list("cnt", "sum")[0] == (2, 450)
    assert ds.filter(dc.C("file_dir") == "tests").to_list("cnt", "sum")[0] == (1, 250)
    assert ds.filter(dc.C("file_dir") == "").to_list("cnt", "sum")[0] == (1, 50)

    persist = ds.save("tmp_ds")
    assert len(persist.to_list("file_dir")) == 4


def test_group_by_with_unlabeled_func_in_partition_by(test_session):
    class CustomFile(DataModel):
        path: str
        size: int

    custom_data = [
        CustomFile(path="docs/readme.txt", size=100),
        CustomFile(path="docs/guide.txt", size=200),
        CustomFile(path="src/main.py", size=300),
        CustomFile(path="src/utils.py", size=150),
        CustomFile(path="tests/test_main.py", size=250),
        CustomFile(path="config.yaml", size=50),
    ]

    ds = dc.read_values(
        custom_file=custom_data,
        session=test_session,
    ).group_by(
        cnt=func.count(),
        sum=func.sum("custom_file.size"),
        partition_by=func.path.parent("custom_file.path"),
    )

    assert set(ds.signals_schema.values.keys()) == {"cnt", "sum", "gr_0"}
    assert sorted(ds.to_list("cnt", "sum")) == [(1, 50), (1, 250), (2, 300), (2, 450)]

    ds2 = dc.read_values(
        custom_file=custom_data,
        session=test_session,
    ).group_by(
        cnt=func.count(),
        partition_by=func.path.parent(dc.C("custom_file.path")),
    )

    assert set(ds2.signals_schema.values.keys()) == {"cnt", "gr_0"}
    assert len(ds2.to_values("cnt")) == 4


def test_partition_by_nested_file(test_session):
    class Signal(DataModel):
        file: File
        amount: int

    signals = [
        Signal(file=File(source="s3://bucket", path="f1.txt"), amount=10),
        Signal(file=File(source="s3://bucket", path="f2.txt"), amount=20),
        Signal(file=File(source="s3://bucket", path="f1.txt"), amount=30),  # duplicate
        Signal(file=File(source="s3://bucket", path="f3.txt"), amount=40),
    ]
    chain = dc.read_values(
        signal=signals,
        session=test_session,
    )

    def test_agg(files: list[File], amounts: list[int]) -> Iterator[tuple[File, int]]:
        yield files[0], sum(amounts)

    # Test partitioning by File type directly
    result = chain.agg(
        test_agg,
        params=("signal.file", "signal.amount"),
        output={"file": File, "total": int},
        partition_by="signal.file",
    ).to_list("file", "total")

    # We should have 3 groups (file1.txt appears twice, so should be grouped)
    assert len(result) == 3

    # Check that files with same unique attributes are grouped together
    file_paths = {f.path for f, _ in result}
    assert file_paths == {"f1.txt", "f2.txt", "f3.txt"}

    # Check total amounts
    totals = {f.path: total for f, total in result}
    assert totals == {
        "f1.txt": 40,  # 10 + 30 (grouped)
        "f2.txt": 20,
        "f3.txt": 40,
    }


def test_partition_by_inherited_file(test_session):
    class MyFile(File):
        amount: int

    my_files = [
        MyFile(source="s3://bucket", path="f1.txt", amount=10),
        MyFile(source="s3://bucket", path="f1.txt", amount=20),  # not a duplicate
        MyFile(source="s3://bucket", path="f1.txt", amount=10),  # duplicate
        MyFile(source="s3://bucket", path="f3.txt", amount=40),
    ]
    chain = dc.read_values(
        file=my_files,
        session=test_session,
    )

    def test_agg(files: list[MyFile]) -> Iterator[tuple[MyFile, int, int]]:
        yield files[0], sum(f.amount for f in files), len(files)

    # Test partitioning by File type directly
    result = chain.agg(
        test_agg,
        params=("file",),
        output={"file": MyFile, "total": int, "cnt": int},
        partition_by="file",
    ).to_list("file", "total", "cnt")

    # We should have 3 groups (file1.txt appears twice, so should be grouped)
    assert len(result) == 3

    # Check that files with same unique attributes are grouped together
    file_paths = [f.path for f, _, _ in result]
    assert sorted(file_paths) == sorted(["f1.txt", "f1.txt", "f3.txt"])

    # Check total amounts
    totals = {(f.path, f.amount): (total, cnt) for f, total, cnt in result}
    assert totals == {
        ("f1.txt", 10): (20, 2),  # 10 + 10 (grouped)
        ("f1.txt", 20): (20, 1),
        ("f3.txt", 40): (40, 1),
    }


def test_no_partition_by(test_session):
    files = [
        File(source="s3://bucket", path="f1.txt", size=100),
        File(source="s3://bucket", path="f2.txt", size=200),
        File(source="s3://bucket", path="f1.txt", size=100),  # duplicate
        File(source="s3://bucket", path="f3.txt", size=300),
    ]
    amounts = [10, 20, 30, 40]

    # Create a chain with File objects
    chain = dc.read_values(
        file=files,
        amount=amounts,
        session=test_session,
    )

    def my_agg(files: list[File], amounts: list[int]) -> Iterator[tuple[str, int, int]]:
        file_names = sorted([f.path for f in files])
        total_size = sum(f.size for f in files)
        yield ",".join(file_names), total_size, sum(amounts)

    # Test aggregate with empty partitioning by (all rows in one group)
    result = chain.agg(
        my_agg,
        params=("file", "amount"),
        output={"file_names": str, "total_size": int, "total_amount": int},
    ).to_list("file_names", "total_size", "total_amount")

    assert len(result) == 1
    assert result[0] == ("f1.txt,f1.txt,f2.txt,f3.txt", 700, 100)


def test_aggregate_after_group_by(test_session):
    files = [
        File(source="s3://bucket", path="f1.txt", size=100),
        File(source="s3://bucket", path="f2.txt", size=200),
        File(source="s3://bucket", path="f1.txt", size=100),  # duplicate
        File(source="s3://bucket", path="f3.txt", size=300),
    ]
    amounts = [10, 20, 30, 40]

    # Create a chain with File objects
    chain = dc.read_values(
        file=files,
        amount=amounts,
        session=test_session,
    )

    # Group by file and aggregate
    grouped = chain.group_by(
        total_size=dc.func.sum("file.size"),
        total_amount=dc.func.sum("amount"),
        partition_by="file",
    )

    assert sorted(grouped.to_list("file.path", "total_size", "total_amount")) == [
        ("f1.txt", 200, 40),
        ("f2.txt", 200, 20),
        ("f3.txt", 300, 40),
    ]

    # Now aggregate over the grouped results
    result = grouped.agg(
        lambda file, total_amount: [(len(file), sum(total_amount))],
        output={"files": int, "total": int},
    ).to_list("files", "total")

    assert result == [(3, 100)]


def test_group_by_with_column_element_in_partition_by(test_session):
    files = [
        File(source="s3://bucket", path="a.txt", size=50),
        File(source="s3://bucket", path="b.txt", size=50),
        File(source="s3://bucket", path="c.txt", size=50),
        File(source="s3://bucket", path="d.txt", size=150),
    ]

    ds = dc.read_values(file=files, session=test_session).group_by(
        cnt=func.count(),
        partition_by=dc.C("file.size") // 100,
    )

    assert set(ds.signals_schema.values.keys()) == {"cnt", "gr_0"}
    assert set(ds.to_values("cnt")) == {1, 3}

    ds2 = dc.read_values(file=files, session=test_session).group_by(
        cnt=func.count(),
        partition_by=[dc.C("file.size") // 100],
    )
    assert set(ds2.to_values("cnt")) == {1, 3}


def test_group_by_partition_by_multi_column_expression(test_session):
    ds = dc.read_values(
        a=[10, 10, 10, 20],
        b=[5, 5, 5, 3],
        session=test_session,
    ).group_by(
        cnt=func.count(),
        partition_by=dc.C("a") + dc.C("b"),
    )
    assert set(ds.signals_schema.values.keys()) == {"cnt", "gr_0"}
    assert set(ds.to_values("cnt")) == {1, 3}


def test_group_by_partition_by_func_and_column_expression(test_session):
    files = [
        File(source="s3://bucket", path="a.txt", size=5),  # 5+5
        File(source="s3://bucket", path="ab.txt", size=5),  # 6+5
        File(source="s3://bucket", path="ab.txt", size=5),  # 6+5
        File(source="s3://bucket", path="a.txt", size=7),  # 5+7
        File(source="s3://bucket", path="a.txt", size=7),  # 5+7
        File(source="s3://bucket", path="a.txt", size=7),  # 5+7
    ]

    ds = dc.read_values(file=files, session=test_session).group_by(
        n=func.count(),
        partition_by=func.string.length(dc.C("file.path")) + dc.C("file.size"),
    )
    assert set(ds.signals_schema.values.keys()) == {"n", "gr_0"}
    assert set(ds.to_values("n")) == {1, 2, 3}


def test_group_by_partition_by_mixed_labeled_and_unlabeled(test_session):
    """One partition_by func has a label, another does not."""

    class CustomFile(DataModel):
        path: str
        size: int

    custom_data = [
        CustomFile(path="docs/readme.txt", size=100),
        CustomFile(path="docs/guide.txt", size=200),
        CustomFile(path="src/main.py", size=300),
        CustomFile(path="src/utils.py", size=150),
    ]

    ds = dc.read_values(
        custom_file=custom_data,
        session=test_session,
    ).group_by(
        cnt=func.count(),
        partition_by=[
            func.path.parent("custom_file.path").label("file_dir"),
            func.string.length(dc.C("custom_file.path")),
        ],
    )

    keys = set(ds.signals_schema.values.keys())
    assert "file_dir" in keys
    assert "gr_0" in keys
    assert "cnt" in keys
    assert len(keys) == 3


def test_group_by_partition_by_expression_type_inference(test_session):
    """Verify correct Python type inference for ColumnElement partition expressions."""
    ds = dc.read_values(
        a=[10, 10, 20],
        b=[5, 5, 3],
        c=[1.5, 2.5, 3.5],
        session=test_session,
    )

    # int + int → int
    r1 = ds.group_by(cnt=func.count(), partition_by=dc.C("a") + dc.C("b"))
    assert r1.signals_schema.values["gr_0"] is int

    # int // literal → int
    r2 = ds.group_by(cnt=func.count(), partition_by=dc.C("a") // 10)
    assert r2.signals_schema.values["gr_0"] is int

    # int + float → float
    r3 = ds.group_by(cnt=func.count(), partition_by=dc.C("a") + dc.C("c"))
    assert r3.signals_schema.values["gr_0"] is float

    # int / literal → float
    r4 = ds.group_by(cnt=func.count(), partition_by=dc.C("a") / 10)
    assert r4.signals_schema.values["gr_0"] is float

    # str + str → str
    ds_str = dc.read_values(
        s=["aa", "bb", "aa"],
        t=["xx", "yy", "xx"],
        session=test_session,
    )
    r5 = ds_str.group_by(cnt=func.count(), partition_by=dc.C("s") + dc.C("t"))
    assert r5.signals_schema.values["gr_0"] is str

    # bool + bool → bool
    ds_bool = dc.read_values(
        x=[True, False, True],
        y=[False, True, False],
        session=test_session,
    )
    r6 = ds_bool.group_by(cnt=func.count(), partition_by=dc.C("x") + dc.C("y"))
    assert r6.signals_schema.values["gr_0"] is bool

    # datetime + datetime → datetime
    from datetime import datetime

    ds_dt = dc.read_values(
        ts=[datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 2)],
        ts2=[datetime(2024, 6, 1), datetime(2024, 6, 1), datetime(2024, 6, 2)],
        session=test_session,
    )
    r7 = ds_dt.group_by(cnt=func.count(), partition_by=dc.C("ts") + dc.C("ts2"))
    assert r7.signals_schema.values["gr_0"] is datetime


def test_group_by_partition_by_name_conflicts_with_agg_column(test_session):
    files = [
        File(source="s3://bucket", path="docs/a.txt", size=100),
    ]
    with pytest.raises(DataChainColumnError, match="conflicts with aggregation"):
        dc.read_values(file=files, session=test_session).group_by(
            parent=func.count(),
            partition_by=func.path.parent("file.path").label("parent"),
        )


def test_name_collision_in_multiple_group_by(test_session):
    ds = dc.read_values(
        a=[1, 1, 2, 2],
        b=[10, 20, 10, 20],
        c=[100, 200, 100, 200],
        session=test_session,
    ).group_by(
        cnt=func.count(),
        partition_by=[dc.C("a") + dc.C("b"), dc.C("c") // 100],
    )
    assert set(ds.signals_schema.values.keys()) == {"gr_0", "gr_1", "cnt"}

    ds2 = ds.group_by(
        total=func.sum("cnt"),
        partition_by=[dc.C("gr_0") * 2, dc.C("gr_1") + 1],
    )
    assert set(ds2.signals_schema.values.keys()) == {"gr_2", "gr_3", "total"}

    result = sorted(
        zip(
            ds2.to_values("gr_2"),
            ds2.to_values("gr_3"),
            ds2.to_values("total"),
            strict=True,
        )
    )
    assert result == [(22, 2, 1), (24, 2, 1), (42, 3, 1), (44, 3, 1)]
