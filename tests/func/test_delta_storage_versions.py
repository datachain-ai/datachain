from uuid import uuid4

import pytest

import datachain as dc
from datachain import C, File


@pytest.mark.parametrize("cloud_type", ["s3", "gs"], indirect=True)
@pytest.mark.parametrize("version_aware", [True], indirect=True)
@pytest.mark.parametrize("versions", [False, True])
@pytest.mark.parametrize("explicit_key", [False, True])
@pytest.mark.parametrize("operation", ["map", "gen"])
def test_delta_refresh_on_versioned_storage(
    cloud_test_catalog_upload, versions, explicit_key, operation
):
    ctc = cloud_test_catalog_upload
    prefix = "delta-" + uuid4().hex[:8]
    uri = f"{ctc.src_uri}/{prefix}/"
    File.upload(b"2", uri + "a.txt", ctc.catalog)
    File.upload(b"3", uri + "b.txt", ctc.catalog)
    processed = []

    def read_number(file: File) -> int:
        processed.append(file.path)
        return int(file.read_text())

    def generate(file: File):
        value = read_number(file)
        for index in range(value):
            yield file, value, index

    def build(delta):
        kwargs = (
            {"delta_on": "file.path", "delta_compare": ["file.etag", "file.version"]}
            if explicit_key
            else {}
        )
        chain = dc.read_storage(
            uri,
            update=True,
            delta=delta,
            versions=versions,
            session=ctc.session,
            **kwargs,
        )
        if operation == "gen":
            chain = chain.gen(
                generate, output={"file": File, "value": int, "index": int}
            )
        else:
            chain = chain.map(value=read_number).map(index=lambda: 0, output=int)
        return chain.filter(C.value < 5)

    def rows(chain):
        return sorted(chain.to_iter("file.path", "value", "index"))

    first = build(True).save("versioned_result")
    initial = rows(first)
    File.upload(b"9", uri + "a.txt", ctc.catalog)
    (ctc.src / prefix / "b.txt").unlink()
    File.upload(b"4", uri + "c.txt", ctc.catalog)
    processed.clear()
    result = build(True).save("versioned_result")
    assert sorted(processed) == [prefix + "/a.txt", prefix + "/c.txt"]
    assert (
        rows(result)
        == rows(build(False))
        == [
            (prefix + "/c.txt", 4, index)
            for index in (range(4) if operation == "gen" else [0])
        ]
    )
    assert rows(first) == initial
    processed.clear()
    assert build(True).save("versioned_result").version == result.version
    assert processed == []
    (ctc.src / prefix / "c.txt").unlink()
    emptied = build(True).save("versioned_result")
    assert emptied.count() == 0
    assert processed == []
    assert build(True).save("versioned_result").version == emptied.version
