import pytest

import datachain as dc
from tests.utils import instantiate_tree

DEEP_TREE = {
    "deep": {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "data.json": '{"key": "value"}',
                        "info.txt": "information",
                        "config.yaml": "setting: true",
                    },
                    "audio.mp3": "audio data",
                    "video.mp4": "video data",
                },
                "documents": {
                    "report.pdf": "pdf content",
                    "notes.md": "# Notes",
                },
                "images": {
                    "photo1.jpg": "jpeg data",
                    "photo2.png": "png data",
                },
            },
            "backup": {
                "old": {
                    "archive.zip": "zip data",
                },
                "new": {
                    "current.tar": "tar data",
                },
            },
            "temp": {
                "log1.log": "log 1",
                "logfile": "log 2",
                "temp1.tmp": "temp data 1",
                "temp2.tmp": "temp data 2",
            },
        },
        "media": {
            "music": {
                "rock": {
                    "song1.mp3": "rock song 1",
                    "song2.flac": "rock song 2",
                },
                "jazz": {
                    "track1.wav": "jazz track 1",
                    "track2.mp3": "jazz track 2",
                },
            },
            "videos": {
                "movie1.mp4": "movie 1",
                "movie2.avi": "movie 2",
            },
        },
    }
}


def count_dict_items(d):
    if not isinstance(d, dict):
        return 1

    count = 0
    for value in d.values():
        count += count_dict_items(value)
    return count


@pytest.fixture
def tmp_dir(tmp_path):
    instantiate_tree(tmp_path, DEEP_TREE)
    return tmp_path


def test_simple_wildcard(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/level1/temp/*.tmp")
    names = {f.name for f in result.to_values("file")}
    assert names == {"temp1.tmp", "temp2.tmp"}


def test_recursive(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/**/*.mp3")
    files = {f.name for f in result.to_values("file")}
    assert files == {"audio.mp3", "song1.mp3", "track2.mp3"}


def test_recursive_patterns_ext(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/level1/**/level4/*")
    files = {f.name for f in result.to_values("file")}
    assert files == {"config.yaml", "data.json", "info.txt"}


def test_recursive_double(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/**/music/**/*.mp3")
    files = {f.name for f in result.to_values("file")}
    assert files == {"song1.mp3", "track2.mp3"}


def test_question_mark_patterns(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/media/videos/movie?.mp4")
    files = {f.name for f in result.to_values("file")}
    assert files == {"movie1.mp4"}

    result = dc.read_storage(f"{tmp_dir}/deep/level1/temp/temp?.tmp")
    files = {f.name for f in result.to_values("file")}
    assert files == {"temp1.tmp", "temp2.tmp"}

    result = dc.read_storage(f"{tmp_dir}/deep/media/videos/movie?.avi")
    files = {f.name for f in result.to_values("file")}
    assert files == {"movie2.avi"}


def test_brace_file_extension(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/**/*.{{mp3,wav,flac}}")
    files = {f.name for f in result.to_values("file")}
    assert files == {"audio.mp3", "song1.mp3", "song2.flac", "track1.wav", "track2.mp3"}


def test_brace_dir(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/level1/level2/{{documents,images}}/*")
    files = {f.name for f in result.to_values("file")}
    assert files == {"notes.md", "photo1.jpg", "photo2.png", "report.pdf"}


def test_brace_file_extension_no_globstar(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/media/videos/*.{{mp4,avi,mkv}}")
    files = {f.name for f in result.to_values("file")}
    assert files == {"movie1.mp4", "movie2.avi"}


def test_combined_complex_patterns_file(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/**/level?/**/*.{{json,yaml,txt}}")
    files = {f.name for f in result.to_values("file")}
    assert files == {"config.yaml", "data.json", "info.txt"}


def test_combined_complex_patterns_dir(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/level1/{{backup,temp}}/**/*.*")
    files = {f.name for f in result.to_values("file")}
    assert files == {"archive.zip", "current.tar", "temp1.tmp", "temp2.tmp", "log1.log"}


def test_directory_boundary_respect(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/media/music/**/*.mp4")
    assert [f.name for f in result.to_values("file")] == []

    result = dc.read_storage(f"{tmp_dir}/deep/level1/temp/*.zip")
    assert [f.name for f in result.to_values("file")] == []

    result = dc.read_storage(f"{tmp_dir}/deep/level1/level2/level3/*.txt")
    assert [f.name for f in result.to_values("file")] == []


def test_very_deep_nesting_patterns(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/level1/level2/level3/level4/*.json")
    files = {f.name for f in result.to_values("file")}
    assert files == {"data.json"}

    result = dc.read_storage(f"{tmp_dir}/deep/level1/level2/level3/level4/*")
    files = {f.name for f in result.to_values("file")}
    assert files == {"config.yaml", "data.json", "info.txt"}

    result = dc.read_storage(f"{tmp_dir}/deep/**/level4/**/*")
    files = {f.name for f in result.to_values("file")}
    assert files == {"config.yaml", "data.json", "info.txt"}


def test_edge_cases_and_empty_results(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/**/*.nonexistent")
    assert [f.name for f in result.to_values("file")] == []

    result = dc.read_storage(f"{tmp_dir}/deep/nonexistent/*.txt")
    assert [f.name for f in result.to_values("file")] == []

    result = dc.read_storage(f"{tmp_dir}/deep/media/*.mp3")
    assert [f.name for f in result.to_values("file")] == []

    result = dc.read_storage(f"{tmp_dir}/deep/**/level5/*.txt")
    assert [f.name for f in result.to_values("file")] == []


def test_pattern_performance_with_large_structure(tmp_dir):
    assert dc.read_storage(f"{tmp_dir}/deep/**/*").count() == count_dict_items(
        DEEP_TREE
    )

    result = dc.read_storage(f"{tmp_dir}/deep/**/*.txt")
    assert [f.name for f in result.to_values("file")] == ["info.txt"]


def test_no_pattern_behavior_unchanged(tmp_dir):
    result = dc.read_storage(f"{tmp_dir}/deep/media/videos")
    files = {f.name for f in result.to_values("file")}
    assert files == {"movie1.mp4", "movie2.avi"}

    result = dc.read_storage(f"{tmp_dir}/deep/level1/temp")
    files = {f.name for f in result.to_values("file")}
    assert files == {"temp1.tmp", "temp2.tmp", "log1.log", "logfile"}


def test_mixed_pattern_types(tmp_dir):
    # Mix wildcard and brace expansion
    result = dc.read_storage(f"{tmp_dir}/deep/media/*/rock/*.{{mp3,flac}}")
    files = {f.name for f in result.to_values("file")}
    assert files == {"song1.mp3", "song2.flac"}

    # Mix question mark with globstar
    result = dc.read_storage(f"{tmp_dir}/deep/**/temp?.*")
    files = {f.name for f in result.to_values("file")}
    assert files == {"temp1.tmp", "temp2.tmp"}

    # Complex pattern with all types
    result = dc.read_storage(
        f"{tmp_dir}/deep/**/level?/{{backup,temp}}/**/*.{{tmp,zip,tar}}"
    )
    files = {f.name for f in result.to_values("file")}
    assert files == {"archive.zip", "current.tar", "temp1.tmp", "temp2.tmp"}


def test_glob_pattern_in_bucket_name_raises_error():
    with pytest.raises(
        ValueError, match=r"Glob patterns in bucket names are not supported.*bucket-\*"
    ):
        dc.read_storage("s3://bucket-*/data/file.txt")

    with pytest.raises(
        ValueError, match=r"Glob patterns in bucket names are not supported.*bucket-\?"
    ):
        dc.read_storage("s3://bucket-?/files/*.txt")

    with pytest.raises(
        ValueError,
        # Brace expansion appears literally in the message, we only need to
        # escape braces for the regex engine, not double escape like before.
        match=(
            r"Glob patterns in bucket names are not supported.*"
            r"bucket-\{dev,prod\}/logs/.*"
        ),
    ):
        dc.read_storage("s3://bucket-{dev,prod}/logs/*.log")


def test_hugging_face_glob_patterns():
    from datachain.lib.dc.storage_pattern import (
        split_uri_pattern,
        validate_cloud_bucket_name,
    )

    base, pattern = split_uri_pattern("hf://datasets/username/repo-name/data/file.txt")
    assert base == "hf://datasets/username/repo-name/data/file.txt"
    assert pattern is None

    base, pattern = split_uri_pattern("hf://datasets/username/repo-name/data/*.txt")
    assert base == "hf://datasets/username/repo-name/data"
    assert pattern == "*.txt"

    with pytest.raises(
        ValueError,
        match=r"Glob patterns in bucket names are not supported.*hf://datasets",
    ):
        validate_cloud_bucket_name("hf://datasets*/username/repo-name/data/file.txt")


def test_brace_expansion_numeric_ranges(tmp_dir):
    (tmp_dir / "deep").mkdir(exist_ok=True)

    for i in range(1, 6):
        (tmp_dir / "deep" / f"file{i}.txt").write_text(f"content {i}")

    result = dc.read_storage(f"{tmp_dir}/deep/file{{1..3}}.txt")
    files = sorted(f.name for f in result.to_values("file"))
    assert files == ["file1.txt", "file2.txt", "file3.txt"]

    for i in range(1, 10):
        (tmp_dir / "deep" / f"data{str(i).zfill(2)}.log").write_text(f"log {i}")

    result = dc.read_storage(f"{tmp_dir}/deep/data{{01..05}}.log")
    files = sorted(f.name for f in result.to_values("file"))
    assert files == [
        "data01.log",
        "data02.log",
        "data03.log",
        "data04.log",
        "data05.log",
    ]


def test_brace_expansion_character_ranges(tmp_dir):
    (tmp_dir / "deep").mkdir(exist_ok=True)
    for char in "abcde":
        dir_path = tmp_dir / "deep" / f"dir-{char}"
        dir_path.mkdir()
        (dir_path / "file.txt").write_text(f"content {char}")

    result = dc.read_storage(f"{tmp_dir}/deep/dir-{{a..c}}/file.txt")
    dirs = sorted(f.source.split("/")[-1] for f in result.to_values("file"))
    assert dirs == ["dir-a", "dir-b", "dir-c"]


def test_brace_expansion_combined_patterns(tmp_dir):
    (tmp_dir / "deep").mkdir(exist_ok=True)
    for year in ["2005"]:
        for month in range(1, 13):
            filename = f"data-{year}-{str(month).zfill(2)}.csv"
            (tmp_dir / "deep" / filename).write_text(f"data {year}-{month}")

    result = dc.read_storage(f"{tmp_dir}/deep/data-2005-{{01..03}}.csv")
    files = sorted([f.name for f in result.to_values("file")])
    assert files == ["data-2005-01.csv", "data-2005-02.csv", "data-2005-03.csv"]

    result = dc.read_storage(f"{tmp_dir}/deep/data-*-{{10..12}}.csv")
    files = sorted(f.name for f in result.to_values("file"))
    assert files == ["data-2005-10.csv", "data-2005-11.csv", "data-2005-12.csv"]


def test_prefix_with_special_chars_reuses_broader_listing(tmp_path):
    # https://github.com/datachain-ai/datachain/issues/1891
    # the relative subpath is recovered from the sanitized listing dataset
    # name and must be decoded back into the real path
    (tmp_path / "some.dir").mkdir()
    (tmp_path / "some.dir" / "a.txt").write_text("a")
    (tmp_path / "some.dir" / "b.txt").write_text("b")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "c.txt").write_text("c")

    assert dc.read_storage(f"{tmp_path}/").count() == 3
    assert dc.read_storage(f"{tmp_path}/some.dir/").count() == 2
    assert dc.read_storage(f"{tmp_path}/some.dir/*.txt").count() == 2
