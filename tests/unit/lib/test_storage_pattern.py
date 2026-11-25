import pytest

from datachain.lib.dc.storage_pattern import (
    expand_brace_pattern,
    list_scope,
    split_uri_pattern,
    storage_sources,
)
from datachain.lib.listing import ListScope


def test_split_uri_pattern_no_pattern():
    assert split_uri_pattern("s3://bucket/dir") == ("s3://bucket/dir", None)
    assert split_uri_pattern("s3://bucket/dir/") == ("s3://bucket/dir/", None)
    assert split_uri_pattern("file:///home/user/data") == (
        "file:///home/user/data",
        None,
    )
    assert split_uri_pattern("/local/path") == ("/local/path", None)


def test_split_uri_pattern_wildcard():
    assert split_uri_pattern("s3://bucket/dir/*.mp3") == (
        "s3://bucket/dir/",
        "*.mp3",
    )
    assert split_uri_pattern("s3://bucket/*.txt") == ("s3://bucket/", "*.txt")
    assert split_uri_pattern("file:///data/*.json") == ("file:///data/", "*.json")
    assert split_uri_pattern("/local/path/*.csv") == ("/local/path/", "*.csv")


def test_split_uri_pattern_globstar():
    assert split_uri_pattern("s3://bucket/**/*.mp3") == ("s3://bucket/", "**/*.mp3")
    assert split_uri_pattern("s3://bucket/dir/**/*.txt") == (
        "s3://bucket/dir/",
        "**/*.txt",
    )
    assert split_uri_pattern("file:///data/**/test/*.json") == (
        "file:///data/",
        "**/test/*.json",
    )


def test_split_uri_pattern_brace_expansion():
    assert split_uri_pattern("s3://bucket/*.{mp3,wav}") == (
        "s3://bucket/",
        "*.{mp3,wav}",
    )
    assert split_uri_pattern("s3://bucket/dir/*.{jpg,png,gif}") == (
        "s3://bucket/dir/",
        "*.{jpg,png,gif}",
    )
    assert split_uri_pattern("file:///data/**/*.{json,jsonl}") == (
        "file:///data/",
        "**/*.{json,jsonl}",
    )


def test_split_uri_pattern_question_mark():
    assert split_uri_pattern("s3://bucket/file?.txt") == (
        "s3://bucket/",
        "file?.txt",
    )
    assert split_uri_pattern("s3://bucket/dir/doc??.pdf") == (
        "s3://bucket/dir/",
        "doc??.pdf",
    )
    assert split_uri_pattern("file:///data/test?/file.txt") == (
        "file:///data/",
        "test?/file.txt",
    )


def test_split_uri_pattern_combined():
    assert split_uri_pattern("s3://bucket/**/test?.{mp3,wav}") == (
        "s3://bucket/",
        "**/test?.{mp3,wav}",
    )
    assert split_uri_pattern("s3://bucket/*/dir/**/*.txt") == (
        "s3://bucket/",
        "*/dir/**/*.txt",
    )


def test_split_uri_pattern_edge_cases():
    assert split_uri_pattern("s3://bucket/*") == ("s3://bucket/", "*")
    assert split_uri_pattern("s3://bucket/**") == ("s3://bucket/", "**")

    assert split_uri_pattern("s3://bucket/dir/subdir/*.mp3") == (
        "s3://bucket/dir/subdir/",
        "*.mp3",
    )

    assert split_uri_pattern("s3://bucket/*/*.mp3") == ("s3://bucket/", "*/*.mp3")


@pytest.mark.parametrize(
    "pattern,expanded",
    [
        ("*.txt", ["*.txt"]),
        ("*.{mp3,wav,flac}", ["*.mp3", "*.wav", "*.flac"]),
        ("**/*.{json,jsonl}", ["**/*.json", "**/*.jsonl"]),
        ("*.{jpg, png}", ["*.jpg", "*.png"]),
        ("file{1..3}.txt", ["file1.txt", "file2.txt", "file3.txt"]),
        ("f{3..1}", ["f3", "f2", "f1"]),
        ("f{03..01}", ["f03", "f02", "f01"]),
        ("f{0..10}", [f"f{i}" for i in range(11)]),
        ("f{00..3}", ["f00", "f01", "f02", "f03"]),
        ("f{01..10}", [f"f{i:02}" for i in range(1, 11)]),
        ("f{09..011}", ["f009", "f010", "f011"]),
        ("f{01..100}", [f"f{i:03}" for i in range(1, 101)]),
        ("f{a..c}", ["fa", "fb", "fc"]),
        ("f{C..A}", ["fC", "fB", "fA"]),
        ("{a..b}/f{1..2}", ["a/f1", "a/f2", "b/f1", "b/f2"]),
        ("{a,{b,c}}", ["a", "b", "c"]),
        ("{a,b{1..2}}", ["a", "b1", "b2"]),
        ("x{a{b,c}}y", ["x{ab}y", "x{ac}y"]),
        ("f{a,}g", ["fag", "fg"]),
        ("f{a}", ["f{a}"]),
        ("f{1..a}", ["f{1..a}"]),
        ("f{-1..1}", ["f{-1..1}"]),
        ("f{1..9..2}", ["f{1..9..2}"]),
        ("f{aa..zz}", ["f{aa..zz}"]),
        ("f{\u00b2..\u00b3}", ["f{\u00b2..\u00b3}"]),
        ("f{\u0660..\u0663}", ["f{\u0660..\u0663}"]),
        ("f{a..\u00e9}", ["f{a..\u00e9}"]),
        ("f{abc", ["f{abc"]),
        ("f{a,b", ["f{a,b"]),
        ("x}{a,b}", ["x}a", "x}b"]),
    ],
)
def test_expand_brace_pattern(pattern, expanded):
    assert expand_brace_pattern(pattern) == expanded


@pytest.mark.parametrize(
    "uris,sources",
    [
        (["s3://b/d/x.jpg"], [("s3://b/d/x.jpg", ())]),
        (["s3://b/d/*.{jpg,png}"], [("s3://b/d/", ("*.jpg", "*.png"))]),
        (["s3://b/d/{,*/x.json}"], [("s3://b/d/", ("**", "*/x.json"))]),
        (["s3://b/d/*/"], [("s3://b/d/", ("*/**",))]),
        (["s3://b/d/*.{json,json}"], [("s3://b/d/", ("*.json",))]),
        (
            ["s3://b/{a,c}/*.jpg"],
            [("s3://b/a/", ("*.jpg",)), ("s3://b/c/", ("*.jpg",))],
        ),
        (["s3://b/d/*.jpg"] * 2, [("s3://b/d/", ("*.jpg",))] * 2),
    ],
)
def test_storage_sources_share_a_base_within_one_uri(uris, sources):
    assert storage_sources(uris, True) == sources


@pytest.mark.parametrize(
    "list_path,patterns,recursive,scope",
    [
        ("dir/", (), True, None),
        ("dir/", (), False, ListScope("dir/", "dir/", True)),
        ("dir//", (), False, ListScope("dir//", "dir//", True)),
        ("", (), False, ListScope("", "", True)),
        ("dir", ("*.jpg",), True, ListScope("dir/", "dir/", True)),
        ("dir//", ("*.jpg",), True, ListScope("dir//", "dir//", True)),
        ("dir///", ("*.jpg",), True, ListScope("dir///", "dir///", True)),
        ("dir", ("img_0*.jpg",), True, ListScope("dir/", "dir/img_0", True)),
        ("logs", ("2024-*/**",), True, ListScope("logs/", "logs/2024-", False)),
        ("logs", ("**/*.json",), True, None),
        ("logs", ("**/*.json",), False, None),
        ("logs", ("2024/**/*.json",), False, ListScope("logs/", "logs/2024/", False)),
        ("", ("d*/x",), True, ListScope("", "d", False)),
        ("dir", ("a[0-9]/*",), True, ListScope("dir/", "dir/a", False)),
        ("dir", ("ab*.jpg", "ac*.png"), True, ListScope("dir/", "dir/a", True)),
        ("dir", ("a/*", "*.jpg"), True, None),
    ],
)
def test_list_scope(list_path, patterns, recursive, scope):
    assert list_scope(list_path, patterns, recursive) == scope


@pytest.mark.parametrize(
    "uri,base",
    [
        ("s3://bucket/photos//*.jpg", "s3://bucket/photos//"),
        ("s3://bucket/photos///*.jpg", "s3://bucket/photos///"),
        ("s3://bucket//*.jpg", "s3://bucket//"),
        ("file:///*.jpg", "file:///"),
        ("/*.jpg", "/"),
    ],
)
def test_glob_base_keeps_directory_boundary(uri, base):
    # URI splitting only, not filtering of keys with empty segments.
    assert split_uri_pattern(uri) == (base, "*.jpg")
