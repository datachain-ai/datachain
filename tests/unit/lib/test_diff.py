from datetime import timezone

import pytest
from pydantic import BaseModel

import datachain as dc
from datachain.diff import CompareStatus, compare_and_split
from datachain.lib.file import File
from datachain.sql.types import Int, String
from tests.utils import sorted_dicts


def _as_utc(d):
    return d.replace(tzinfo=timezone.utc)


@pytest.fixture
def str_default(test_session):
    return String.default_value(test_session.catalog.warehouse.db.dialect)


@pytest.fixture
def int_default(test_session):
    return Int.default_value(test_session.catalog.warehouse.db.dialect)


@pytest.mark.parametrize("added", (True, False))
@pytest.mark.parametrize("deleted", (True, False))
@pytest.mark.parametrize("modified", (True, False))
@pytest.mark.parametrize("same", (True, False))
def test_diff(test_session, str_default, added, deleted, modified, same):
    ds1 = dc.read_values(
        id=[1, 2, 4],
        name=["John1", "Doe", "Andy"],
        session=test_session,
    )

    ds2 = dc.read_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        session=test_session,
    )

    if not any([added, deleted, modified, same]):
        with pytest.raises(ValueError) as exc_info:
            diff = ds1.diff(
                ds2,
                added=added,
                deleted=deleted,
                modified=modified,
                same=same,
                on=["id"],
                status_col="diff",
            )
        assert str(exc_info.value) == (
            "At least one of added, deleted, modified, same flags must be set"
        )
        return

    diff = ds1.diff(
        ds2,
        added=added,
        deleted=deleted,
        modified=modified,
        same=same,
        on=["id"],
        status_col="diff",
    )

    chains = compare_and_split(
        ds1,
        ds2,
        same=True,
        on=["id"],
    )

    expected = []

    if modified:
        assert "diff" not in chains[CompareStatus.MODIFIED].signals_schema.db_signals()
        expected.append((CompareStatus.MODIFIED, 1, "John1"))

    if added:
        assert "diff" not in chains[CompareStatus.ADDED].signals_schema.db_signals()
        expected.append((CompareStatus.ADDED, 2, "Doe"))

    if deleted:
        assert "diff" not in chains[CompareStatus.DELETED].signals_schema.db_signals()
        expected.append((CompareStatus.DELETED, 3, str_default))

    if same:
        assert "diff" not in chains[CompareStatus.SAME].signals_schema.db_signals()
        expected.append((CompareStatus.SAME, 4, "Andy"))

    assert diff.order_by("id").to_list("diff", "id", "name") == expected


def test_diff_no_status_col(test_session, str_default):
    ds1 = dc.read_values(
        id=[1, 2, 4],
        name=["John1", "Doe", "Andy"],
        session=test_session,
    )

    ds2 = dc.read_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        session=test_session,
    )

    diff = ds1.diff(
        ds2,
        same=True,
        on=["id"],
        status_col=None,
    )

    expected = [
        (1, "John1"),
        (2, "Doe"),
        (3, str_default),
        (4, "Andy"),
    ]

    assert diff.order_by("id").to_list() == expected


def test_diff_read_dataset(test_session, str_default):
    ds1 = dc.read_values(
        id=[1, 2, 4],
        name=["John1", "Doe", "Andy"],
        session=test_session,
    ).save("ds1")

    ds2 = dc.read_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        session=test_session,
    ).save("ds2")

    # this adds sys columns to ds1 and ds2
    ds1 = dc.read_dataset("ds1")
    ds2 = dc.read_dataset("ds2")

    diff = ds1.diff(ds2, same=True, on=["id"], status_col="diff")

    assert diff.order_by("id").to_list("diff", "id", "name") == [
        (CompareStatus.MODIFIED, 1, "John1"),
        (CompareStatus.ADDED, 2, "Doe"),
        (CompareStatus.DELETED, 3, str_default),
        (CompareStatus.SAME, 4, "Andy"),
    ]


@pytest.mark.parametrize("right_name", ("other_name", "name"))
def test_diff_with_explicit_compare_fields(test_session, str_default, right_name):
    ds1 = dc.read_values(
        id=[1, 2, 4],
        name=["John1", "Doe", "Andy"],
        city=["New York", "Boston", "San Francisco"],
        session=test_session,
    ).save("ds1")

    ds2_data = {
        "id": [1, 3, 4],
        "city": ["Washington", "Seattle", "Miami"],
        f"{right_name}": ["John", "Mark", "Andy"],
        "session": test_session,
    }

    ds2 = dc.read_values(**ds2_data).save("ds2")

    diff = ds1.diff(
        ds2,
        on=["id"],
        compare=["name"],
        right_compare=[right_name],
        same=True,
        status_col="diff",
    )

    expected = [
        (CompareStatus.MODIFIED, 1, "John1", "New York"),
        (CompareStatus.ADDED, 2, "Doe", "Boston"),
        (CompareStatus.DELETED, 3, str_default, str_default),
        (CompareStatus.SAME, 4, "Andy", "San Francisco"),
    ]

    collect_fields = ["diff", "id", "name", "city"]
    assert diff.order_by("id").to_list(*collect_fields) == expected


def test_diff_different_left_right_on_columns(test_session, str_default):
    ds1 = dc.read_values(
        id=[1, 2, 4],
        name=["John1", "Doe", "Andy"],
        session=test_session,
    ).save("ds1")

    ds2 = dc.read_values(
        other_id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        session=test_session,
    ).save("ds2")

    diff = ds1.diff(
        ds2,
        same=True,
        on=["id"],
        right_on=["other_id"],
        status_col="diff",
    )

    expected = [
        (CompareStatus.MODIFIED, 1, "John1"),
        (CompareStatus.ADDED, 2, "Doe"),
        (CompareStatus.DELETED, 3, str_default),
        (CompareStatus.SAME, 4, "Andy"),
    ]

    collect_fields = ["diff", "id", "name"]
    assert diff.order_by("id").to_list(*collect_fields) == expected


@pytest.mark.parametrize("on_self", (True, False))
def test_diff_on_equal_datasets(test_session, on_self):
    ds1 = dc.read_values(
        id=[1, 2, 3],
        name=["John", "Doe", "Andy"],
        session=test_session,
    )

    if on_self:
        ds2 = ds1
    else:
        ds2 = dc.read_values(
            id=[1, 2, 3],
            name=["John", "Doe", "Andy"],
            session=test_session,
        )

    diff = ds1.diff(
        ds2,
        same=True,
        on=["id"],
        status_col="diff",
    )

    expected = [
        (CompareStatus.SAME, 1, "John"),
        (CompareStatus.SAME, 2, "Doe"),
        (CompareStatus.SAME, 3, "Andy"),
    ]

    collect_fields = ["diff", "id", "name"]
    assert diff.order_by("id").to_list(*collect_fields) == expected


def test_diff_only_on_columns_treated_as_same(test_session):
    ds1 = dc.read_values(
        id=[1, 2],
        session=test_session,
    )
    ds2 = dc.read_values(
        id=[1, 2],
        session=test_session,
    )

    diff = ds1.diff(ds2, on=["id"], same=True, status_col="diff")

    assert diff.order_by("id").to_list("diff", "id") == [
        (CompareStatus.SAME, 1),
        (CompareStatus.SAME, 2),
    ]


def test_diff_multiple_columns(test_session, str_default):
    ds1 = dc.read_values(
        id=[1, 2, 4],
        name=["John", "Doe", "Andy"],
        city=["London", "New York", "Tokyo"],
        session=test_session,
    )
    ds2 = dc.read_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        city=["Paris", "Berlin", "Tokyo"],
        session=test_session,
    )

    diff = ds1.diff(ds2, same=True, on=["id"], status_col="diff")

    assert sorted_dicts(diff.to_records(), "id") == sorted_dicts(
        [
            {"diff": CompareStatus.MODIFIED, "id": 1, "name": "John", "city": "London"},
            {"diff": CompareStatus.ADDED, "id": 2, "name": "Doe", "city": "New York"},
            {
                "diff": CompareStatus.DELETED,
                "id": 3,
                "name": str_default,
                "city": str_default,
            },
            {"diff": CompareStatus.SAME, "id": 4, "name": "Andy", "city": "Tokyo"},
        ],
        "id",
    )


def test_diff_multiple_match_columns(test_session, str_default):
    ds1 = dc.read_values(
        id=[1, 2, 4],
        name=["John", "Doe", "Andy"],
        city=["London", "New York", "Tokyo"],
        session=test_session,
    )
    ds2 = dc.read_values(
        id=[1, 3, 4],
        name=["John", "John", "Andy"],
        city=["Paris", "Berlin", "Tokyo"],
        session=test_session,
    )

    diff = ds1.diff(ds2, same=True, on=["id", "name"], status_col="diff")

    assert sorted_dicts(diff.to_records(), "id") == sorted_dicts(
        [
            {"diff": CompareStatus.MODIFIED, "id": 1, "name": "John", "city": "London"},
            {"diff": CompareStatus.ADDED, "id": 2, "name": "Doe", "city": "New York"},
            {
                "diff": CompareStatus.DELETED,
                "id": 3,
                "name": "John",
                "city": str_default,
            },
            {"diff": CompareStatus.SAME, "id": 4, "name": "Andy", "city": "Tokyo"},
        ],
        "id",
    )


def test_diff_additional_column_on_left(test_session, str_default):
    ds1 = dc.read_values(
        id=[1, 2, 4],
        name=["John", "Doe", "Andy"],
        city=["London", "New York", "Tokyo"],
        session=test_session,
    ).save("ds1")
    ds2 = dc.read_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        session=test_session,
    ).save("ds2")

    diff = ds1.diff(ds2, same=True, on=["id"], status_col="diff")

    assert sorted_dicts(diff.to_records(), "id") == sorted_dicts(
        [
            {"diff": CompareStatus.MODIFIED, "id": 1, "name": "John", "city": "London"},
            {"diff": CompareStatus.ADDED, "id": 2, "name": "Doe", "city": "New York"},
            {
                "diff": CompareStatus.DELETED,
                "id": 3,
                "name": str_default,
                "city": str_default,
            },
            {"diff": CompareStatus.MODIFIED, "id": 4, "name": "Andy", "city": "Tokyo"},
        ],
        "id",
    )


def test_diff_additional_column_on_right(test_session, str_default):
    ds1 = dc.read_values(
        id=[1, 2, 4],
        name=["John", "Doe", "Andy"],
        session=test_session,
    )
    ds2 = dc.read_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        city=["London", "New York", "Tokyo"],
        session=test_session,
    )

    diff = ds1.diff(ds2, same=True, on=["id"], status_col="diff")

    assert sorted_dicts(diff.to_records(), "id") == sorted_dicts(
        [
            {"diff": CompareStatus.MODIFIED, "id": 1, "name": "John"},
            {"diff": CompareStatus.ADDED, "id": 2, "name": "Doe"},
            {"diff": CompareStatus.DELETED, "id": 3, "name": str_default},
            {"diff": CompareStatus.MODIFIED, "id": 4, "name": "Andy"},
        ],
        "id",
    )


def test_diff_missing_on(test_session):
    ds1 = dc.read_values(id=[1, 2, 4], session=test_session)
    ds2 = dc.read_values(id=[1, 2, 4], session=test_session)

    with pytest.raises(ValueError) as exc_info:
        ds1.diff(ds2, on=None)  # type: ignore[arg-type]

    assert str(exc_info.value) == "'on' must be specified"


def test_diff_right_on_wrong_length(test_session):
    ds1 = dc.read_values(id=[1, 2, 4], session=test_session)
    ds2 = dc.read_values(id=[1, 2, 4], session=test_session)

    with pytest.raises(ValueError) as exc_info:
        ds1.diff(ds2, on=["id"], right_on=["id", "name"])

    assert str(exc_info.value) == "'on' and 'right_on' must be have the same length"


def test_diff_right_compare_wrong_length(test_session):
    ds1 = dc.read_values(id=[1, 2, 4], session=test_session)
    ds2 = dc.read_values(id=[1, 2, 4], session=test_session)

    with pytest.raises(ValueError) as exc_info:
        ds1.diff(ds2, on=["id"], compare=["name"], right_compare=["name", "city"])

    assert str(exc_info.value) == (
        "'compare' and 'right_compare' must have the same length"
    )


def test_diff_right_compare_defined_but_not_compare(test_session):
    ds1 = dc.read_values(id=[1, 2, 4], session=test_session)
    ds2 = dc.read_values(id=[1, 2, 4], session=test_session)

    with pytest.raises(ValueError) as exc_info:
        ds1.diff(ds2, on=["id"], right_compare=["name"])

    assert str(exc_info.value) == (
        "'compare' must be defined if 'right_compare' is defined"
    )


@pytest.mark.parametrize("status_col", ("diff", None))
@pytest.mark.parametrize("right_on", ("file2", None))
def test_file_diff(test_session, str_default, int_default, status_col, right_on):
    fs1 = File(source="s1", path="p1", version="2", etag="e2")
    fs1_updated = File(source="s1", path="p1", version="1", etag="e1")
    fs2 = File(source="s2", path="p2", version="1", etag="e1")
    fs3 = File(source="s3", path="p3", version="1", etag="e1")
    fs4 = File(source="s4", path="p4", version="1", etag="e1")

    ds1 = dc.read_values(
        file1=[fs1_updated, fs2, fs4], score=[1, 2, 4], session=test_session
    )

    if right_on:
        ds2 = dc.read_values(
            file2=[fs1, fs3, fs4], score=[1, 3, 4], session=test_session
        )
    else:
        ds2 = dc.read_values(
            file1=[fs1, fs3, fs4], score=[1, 3, 4], session=test_session
        )

    diff = ds1.file_diff(
        ds2,
        added=True,
        deleted=True,
        modified=True,
        same=True,
        on="file1",
        right_on=right_on,
        status_col=status_col,
    )

    expected = [
        (CompareStatus.MODIFIED, "s1", "p1", "1", "e1", 1),
        (CompareStatus.ADDED, "s2", "p2", "1", "e1", 2),
        (CompareStatus.DELETED, "s3", "p3", str_default, str_default, int_default),
        (CompareStatus.SAME, "s4", "p4", "1", "e1", 4),
    ]

    collect_fields = [
        "diff",
        "file1.source",
        "file1.path",
        "file1.version",
        "file1.etag",
        "score",
    ]
    if not status_col:
        expected = [row[1:] for row in expected]
        collect_fields = collect_fields[1:]

    assert diff.order_by("file1.source").to_list(*collect_fields) == expected


@pytest.mark.parametrize("status_col", ("diff", None))
def test_file_diff_nested(test_session, str_default, int_default, status_col):
    class Nested(BaseModel):
        file: File

    fs1 = Nested(file=File(source="s1", path="p1", version="2", etag="e2"))
    fs1_updated = Nested(file=File(source="s1", path="p1", version="1", etag="e1"))
    fs2 = Nested(file=File(source="s2", path="p2", version="1", etag="e1"))
    fs3 = Nested(file=File(source="s3", path="p3", version="1", etag="e1"))
    fs4 = Nested(file=File(source="s4", path="p4", version="1", etag="e1"))

    ds1 = dc.read_values(
        nested=[fs1_updated, fs2, fs4], score=[1, 2, 4], session=test_session
    )
    ds2 = dc.read_values(nested=[fs1, fs3, fs4], score=[1, 3, 4], session=test_session)

    diff = ds1.file_diff(
        ds2,
        added=True,
        deleted=True,
        modified=True,
        same=True,
        on="nested.file",
        status_col=status_col,
    )

    expected = [
        (CompareStatus.MODIFIED, fs1_updated, 1),
        (CompareStatus.ADDED, fs2, 2),
        (CompareStatus.DELETED, fs3, 3),
        (CompareStatus.SAME, fs4, 4),
    ]

    expected = [
        (CompareStatus.MODIFIED, "s1", "p1", "1", "e1", 1),
        (CompareStatus.ADDED, "s2", "p2", "1", "e1", 2),
        (CompareStatus.DELETED, "s3", "p3", str_default, str_default, int_default),
        (CompareStatus.SAME, "s4", "p4", "1", "e1", 4),
    ]

    collect_fields = [
        "diff",
        "nested.file.source",
        "nested.file.path",
        "nested.file.version",
        "nested.file.etag",
        "score",
    ]
    if not status_col:
        expected = [row[1:] for row in expected]
        collect_fields = collect_fields[1:]

    assert diff.order_by("nested.file.source").to_list(*collect_fields) == expected


def test_multiple_diffs(test_session):
    ds1 = dc.read_values(
        id=[1, 2, 3, 4, 5],
        name=["John", "Doe", "Andy", "Matt", "Rick"],
        session=test_session,
    )
    ds2 = dc.read_values(
        id=[1, 2, 3],
        name=["John", "Doe", "Andy"],
        session=test_session,
    )
    ds3 = dc.read_values(
        id=[1, 2],
        name=["John", "Doe"],
        session=test_session,
    )

    diff = ds1.diff(
        ds2,
        added=True,
        deleted=False,
        modified=False,
        same=False,
        on=["id"],
        status_col="diff",
    ).diff(
        ds3,
        added=True,
        deleted=False,
        modified=False,
        same=False,
        on=["id"],
        status_col="diff",
    )

    assert diff.order_by("id").to_list(*["diff", "id", "name"]) == [
        ("A", 4, "Matt"),
        ("A", 5, "Rick"),
    ]
