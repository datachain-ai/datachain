import datetime
import json
import uuid

import pytest

import datachain as dc
from datachain.lib.file import File


def _parse_isoformat(value: str) -> datetime.datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.datetime.fromisoformat(value)


def test_to_json_storage_files(tmp_dir, test_session):
    a_path = tmp_dir / "a.txt"
    a_path.write_text("a")
    b_path = tmp_dir / "b.txt"
    b_path.write_text("bb")

    chain = dc.read_storage(tmp_dir.as_uri(), session=test_session).filter(
        dc.C("file.path").glob("*.txt")
    )
    path = tmp_dir / "files.json"
    stored = chain.order_by("file.path").to_json(path)

    assert isinstance(stored, File)
    assert stored.path == path.name
    assert stored.source.startswith("file://")
    assert stored.size == path.stat().st_size

    with open(path) as f:
        values = json.load(f)

    assert [row["file"]["path"] for row in values] == ["a.txt", "b.txt"]

    expected_sizes = {
        a_path.name: a_path.stat().st_size,
        b_path.name: b_path.stat().st_size,
    }
    assert {
        row["file"]["path"]: row["file"].get("size") for row in values
    } == expected_sizes

    expected_last_modified = {
        a_path.name: datetime.datetime.fromtimestamp(
            a_path.stat().st_mtime, datetime.timezone.utc
        ),
        b_path.name: datetime.datetime.fromtimestamp(
            b_path.stat().st_mtime, datetime.timezone.utc
        ),
    }

    for row in values:
        file_info = row["file"]
        path = file_info.get("path")
        assert path in {a_path.name, b_path.name}
        assert isinstance(path, str) and path

        last_modified = file_info.get("last_modified")
        assert last_modified is not None
        parsed_last_modified = datetime.datetime.fromisoformat(
            last_modified.replace("Z", "+00:00")
        )
        assert parsed_last_modified == expected_last_modified[path]

        restored = File(**file_info)
        assert restored.path == path
        assert restored.size == expected_sizes[path]
        assert restored.last_modified == expected_last_modified[path]


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
def test_to_json_returns_file_with_version(cloud_test_catalog_upload, cloud_type):
    ctc = cloud_test_catalog_upload

    chain = dc.read_values(value=[1, 2], session=ctc.session)
    dest = f"{ctc.src_uri}/to-json-{uuid.uuid4().hex}.json"

    stored = chain.to_json(dest)

    assert isinstance(stored, File)
    assert stored.version
    assert stored.source == ctc.src_uri
    assert stored.path.endswith(".json")

    parsed = json.loads(stored.read_text())
    assert sorted([row["value"] for row in parsed]) == [1, 2]
