import pytest

from datachain.lib.dc.storage_pattern import (
    _expand_range,
    expand_brace_pattern,
    list_scope,
    split_uri_pattern,
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


def test_expand_brace_pattern():
    assert expand_brace_pattern("*.{mp3,wav}") == ["*.mp3", "*.wav"]
    assert expand_brace_pattern("*.{jpg,png,gif}") == ["*.jpg", "*.png", "*.gif"]
    assert expand_brace_pattern("**/*.{json,jsonl}") == ["**/*.json", "**/*.jsonl"]
    assert expand_brace_pattern("*.txt") == ["*.txt"]

    # Complex pattern
    assert expand_brace_pattern("dir/*.{mp3,wav,flac}") == [
        "dir/*.mp3",
        "dir/*.wav",
        "dir/*.flac",
    ]


def test_expand_brace_pattern_numeric_ranges():
    assert expand_brace_pattern("file{1..3}.txt") == [
        "file1.txt",
        "file2.txt",
        "file3.txt",
    ]

    assert expand_brace_pattern("file{10..13}") == [
        "file10",
        "file11",
        "file12",
        "file13",
    ]

    assert expand_brace_pattern("file{01..03}.txt") == [
        "file01.txt",
        "file02.txt",
        "file03.txt",
    ]

    assert expand_brace_pattern("file{01..10}.txt") == [
        "file01.txt",
        "file02.txt",
        "file03.txt",
        "file04.txt",
        "file05.txt",
        "file06.txt",
        "file07.txt",
        "file08.txt",
        "file09.txt",
        "file10.txt",
    ]

    assert expand_brace_pattern("file{3..1}.txt") == [
        "file3.txt",
        "file2.txt",
        "file1.txt",
    ]


def test_expand_brace_pattern_character_ranges():
    assert expand_brace_pattern("file{a..c}.txt") == [
        "filea.txt",
        "fileb.txt",
        "filec.txt",
    ]

    assert expand_brace_pattern("file{A..C}") == ["fileA", "fileB", "fileC"]
    assert expand_brace_pattern("file{c..a}") == ["filec", "fileb", "filea"]


def test_expand_brace_pattern_complex():
    assert expand_brace_pattern("{a..b}/file{1..2}.txt") == [
        "a/file1.txt",
        "a/file2.txt",
        "b/file1.txt",
        "b/file2.txt",
    ]

    assert expand_brace_pattern("dir{1..2}/subdir/file.txt") == [
        "dir1/subdir/file.txt",
        "dir2/subdir/file.txt",
    ]

    result = expand_brace_pattern("file{1..2}.{txt,log}")
    expected = ["file1.txt", "file1.log", "file2.txt", "file2.log"]
    assert sorted(result) == sorted(expected)


def test_expand_brace_pattern_edge_cases():
    assert _expand_range("abc") == ["abc"]
    assert _expand_range("1..2..3") == ["1..2..3"]

    # reverse numeric with zero-padding
    assert expand_brace_pattern("f{03..01}") == ["f03", "f02", "f01"]

    # unknown range format
    assert _expand_range("aa..zz") == ["aa..zz"]

    # malformed braces - no closing brace
    assert expand_brace_pattern("f{abc") == ["f{abc"]


@pytest.mark.parametrize(
    "list_path,pattern,recursive,scope",
    [
        ("dir/", None, True, None),
        ("dir/", None, False, ListScope("dir/", "dir/", True)),
        ("", None, False, ListScope("", "", True)),
        ("dir", "*.jpg", True, ListScope("dir/", "dir/", True)),
        ("dir", "img_0*.jpg", True, ListScope("dir/", "dir/img_0", True)),
        ("logs", "2024-*/**", True, ListScope("logs/", "logs/2024-", False)),
        ("logs", "**/*.json", True, None),
        ("", "d*/x", True, ListScope("", "d", False)),
        ("dir", "a[0-9]/*", True, ListScope("dir/", "dir/a", False)),
    ],
)
def test_list_scope(list_path, pattern, recursive, scope):
    assert list_scope(list_path, pattern, recursive) == scope


@pytest.mark.parametrize(
    "uri,base",
    [
        ("s3://bucket/photos//*.jpg", "s3://bucket/photos//"),
        ("file:///*.jpg", "file:///"),
        ("/*.jpg", "/"),
    ],
)
def test_glob_base_keeps_directory_boundary(uri, base):
    # URI splitting only, not filtering of keys with empty segments.
    assert split_uri_pattern(uri) == (base, "*.jpg")
