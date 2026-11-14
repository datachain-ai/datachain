import pytest

import datachain as dc

pytestmark = pytest.mark.parametrize(
    "cloud_type,version_aware", [("s3", False)], indirect=True
)


def read_text(file: dc.File) -> str:
    return file.read_text()


def test_union_with_chain_of_other_config_reads_each_source_with_its_config(
    cloud_server, cloud_server_credentials, tmp_path, test_session
):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    (local_dir / "note.txt").write_text("local")
    elsewhere = {"aws_endpoint_url": "http://127.0.0.1:9"}
    local = dc.read_storage(
        local_dir.as_uri(), client_config=elsewhere, session=test_session
    )
    cats = dc.read_storage(
        f"{cloud_server.src_uri}/cats/",
        client_config=cloud_server.client_config,
        session=test_session,
    )

    texts = local.union(cats).map(text=read_text).to_values("text")

    assert sorted(texts) == ["local", "meow", "mrow"]


def test_read_dataset_after_save_reads_with_listing_config_in_workers(
    cloud_server, cloud_server_credentials, test_session_tmpfile
):
    dc.read_storage(
        f"{cloud_server.src_uri}/cats/",
        client_config=cloud_server.client_config,
        session=test_session_tmpfile,
    ).save("cats")

    texts = (
        dc.read_dataset("cats", session=test_session_tmpfile)
        .settings(parallel=2)
        .map(text=read_text)
        .to_values("text")
    )

    assert sorted(texts) == ["meow", "mrow"]
