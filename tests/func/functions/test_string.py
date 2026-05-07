import datachain as dc
from datachain import func


def test_string_length(test_session):
    class Data(dc.DataModel):
        s1: str

    ds = (
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(s1="hello"),
                Data(s1="Hello, world!"),
                Data(s1=""),
            ),
            s2=("Hello, world!", "hello", ""),
            session=test_session,
        )
        .mutate(
            t1=func.string.length("data.s1"),
            t2=func.string.length(dc.C("data.s1")),
            t3=func.string.length(dc.C("s2")),
            t4=func.string.length("s2"),
            t5=func.string.length(dc.func.literal("hello")),
            t6=func.string.length(dc.func.literal("Hello, World!")),
            t7=func.string.length(dc.func.literal("")),
        )
        .order_by("id")
    ).to_list("t1", "t2", "t3", "t4", "t5", "t6", "t7")

    assert ds == [
        (5, 5, 13, 13, 5, 13, 0),
        (13, 13, 5, 5, 5, 13, 0),
        (0, 0, 0, 0, 5, 13, 0),
    ]


def test_string_split(test_session):
    class Data(dc.DataModel):
        s1: str

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(s1="a/b/c"),
                Data(s1="hello world"),
                Data(s1=""),
            ),
            s2=("x,y,z", "foo", ""),
            session=test_session,
        )
        .mutate(
            t1=func.string.split("data.s1", "/"),
            t2=func.string.split(dc.C("data.s1"), " "),
            t3=func.string.split(dc.C("s2"), ","),
            t4=func.string.split("s2", ","),
            t5=func.string.split(dc.func.literal("a b c"), " "),
            t6=func.string.split(dc.func.literal("a,b,c"), ",", 1),
            t7=func.string.split(dc.func.literal(""), ","),
            t8=func.string.split("data.s1", "/", limit=1),
        )
        .order_by("id")
        .to_list("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8")
    )

    assert ds == [
        (
            ["a", "b", "c"],
            ["a/b/c"],
            ["x", "y", "z"],
            ["x", "y", "z"],
            ["a", "b", "c"],
            ["a", "b,c"],
            [""],
            ["a", "b/c"],
        ),
        (
            ["hello world"],
            ["hello", "world"],
            ["foo"],
            ["foo"],
            ["a", "b", "c"],
            ["a", "b,c"],
            [""],
            ["hello world"],
        ),
        (
            [""],
            [""],
            [""],
            [""],
            ["a", "b", "c"],
            ["a", "b,c"],
            [""],
            [""],
        ),
    ]


def test_string_replace(test_session):
    class Data(dc.DataModel):
        s1: str

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(s1="foo bar foo"),
                Data(s1="hello world"),
                Data(s1=""),
            ),
            s2=("foo", "bar", ""),
            session=test_session,
        )
        .mutate(
            t1=func.string.replace("data.s1", "foo", "baz"),
            t2=func.string.replace(dc.C("data.s1"), "world", "earth"),
            t3=func.string.replace(dc.C("s2"), "foo", "baz"),
            t4=func.string.replace("s2", "bar", "baz"),
            t5=func.string.replace(dc.func.literal("foo bar foo"), "foo", "baz"),
            t6=func.string.replace(dc.func.literal("hello world"), "world", "earth"),
            t7=func.string.replace(dc.func.literal(""), "foo", "baz"),
        )
        .order_by("id")
        .to_list("t1", "t2", "t3", "t4", "t5", "t6", "t7")
    )

    assert ds == [
        ("baz bar baz", "foo bar foo", "baz", "foo", "baz bar baz", "hello earth", ""),
        ("hello world", "hello earth", "bar", "baz", "baz bar baz", "hello earth", ""),
        ("", "", "", "", "baz bar baz", "hello earth", ""),
    ]


def test_string_regexp_replace(test_session):
    class Data(dc.DataModel):
        s1: str

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(s1="abc123def"),
                Data(s1="hello 456 world"),
                Data(s1=""),
            ),
            s2=("foo123bar", "bar456baz", ""),
            session=test_session,
        )
        .mutate(
            t1=func.string.regexp_replace("data.s1", r"\d+", "X"),
            t2=func.string.regexp_replace(dc.C("data.s1"), r"[a-z]+", "Y"),
            t3=func.string.regexp_replace(dc.C("s2"), r"\d+", "Z"),
            t4=func.string.regexp_replace("s2", r"bar", "Q"),
            t5=func.string.regexp_replace(dc.func.literal("foo123bar"), r"\d+", "Z"),
            t6=func.string.regexp_replace(
                dc.func.literal("hello 456 world"), r"[a-z]+", "Y"
            ),
            t7=func.string.regexp_replace(dc.func.literal(""), r"foo", "bar"),
        )
        .order_by("id")
        .to_list("t1", "t2", "t3", "t4", "t5", "t6", "t7")
    )

    assert ds == [
        ("abcXdef", "Y123Y", "fooZbar", "foo123Q", "fooZbar", "Y 456 Y", ""),
        ("hello X world", "Y 456 Y", "barZbaz", "Q456baz", "fooZbar", "Y 456 Y", ""),
        ("", "", "", "", "fooZbar", "Y 456 Y", ""),
    ]


def test_string_regexp_replace_case_insensitive(test_session):
    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            s=("Foo bar FOO", "no match here", "FOObarfoo"),
            session=test_session,
        )
        .mutate(
            # case-sensitive (default): only lowercase "foo" replaced
            cs=func.string.regexp_replace("s", r"foo", "X"),
            # inline (?i) flag enables case-insensitive matching
            ci=func.string.regexp_replace("s", r"(?i)foo", "X"),
        )
        .order_by("id")
        .to_list("cs", "ci")
    )

    assert ds == [
        ("Foo bar FOO", "X bar X"),
        ("no match here", "no match here"),
        ("FOObarX", "XbarX"),
    ]


def test_string_byte_hamming_distance(test_session):
    class Data(dc.DataModel):
        s1: str
        s2: str

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(s1="hello", s2="world"),
                Data(s1="hello", s2="hello"),
                Data(s1="", s2=""),
            ),
            s1=("hello", "hello", ""),
            s2=("world", "hello", ""),
            session=test_session,
        )
        .mutate(
            t1=func.byte_hamming_distance("data.s1", "data.s2"),
            t2=func.byte_hamming_distance(dc.C("data.s1"), "data.s2"),
            t3=func.byte_hamming_distance("data.s1", dc.C("data.s2")),
            t4=func.byte_hamming_distance(dc.C("data.s1"), dc.C("data.s2")),
            t5=func.byte_hamming_distance(dc.C("data.s1"), dc.func.literal("world")),
            t6=func.byte_hamming_distance("s1", "s2"),
            t7=func.byte_hamming_distance(dc.C("s1"), dc.func.literal("world")),
            t8=func.byte_hamming_distance(dc.func.literal("hello"), dc.C("s2")),
            t9=func.byte_hamming_distance(
                dc.func.literal("hello"), dc.func.literal("world")
            ),
        )
        .order_by("id")
        .to_list("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9")
    )

    assert ds == [
        (4, 4, 4, 4, 4, 4, 4, 4, 4),
        (0, 0, 0, 0, 4, 0, 4, 0, 4),
        (0, 0, 0, 0, 5, 0, 5, 5, 4),
    ]
