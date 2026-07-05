import pytest

from datachain.client.azure import AzureClient
from datachain.client.gcs import GCSClient
from datachain.client.local import FileClient
from datachain.client.s3 import ClientS3
from datachain.client.writeconfig import WriteConfig

FULL = WriteConfig(
    content_type="application/pdf",
    content_disposition="attachment",
    cache_control="max-age=3600",
    content_encoding="gzip",
    metadata={"a": "b"},
    extra={"ACL": "public-read"},
)
# Same normalized fields, without the raw escape hatch (GCS/Azure reject extra).
NORMALIZED = WriteConfig(
    content_type="application/pdf",
    content_disposition="attachment",
    cache_control="max-age=3600",
    content_encoding="gzip",
    metadata={"a": "b"},
)


def _wk(cls, cfg, *, streaming):
    # _write_kwargs does not touch instance state, so bypass __init__.
    return object.__new__(cls)._write_kwargs(cfg, streaming=streaming)


def test_write_config_empty():
    assert WriteConfig().is_empty()
    assert not WriteConfig().has_content_settings()
    assert not WriteConfig(content_type="x").is_empty()
    assert WriteConfig(content_type="x").has_content_settings()
    assert not WriteConfig(metadata={"a": "b"}).has_content_settings()
    assert not WriteConfig(metadata={"a": "b"}).is_empty()


def test_s3_write_kwargs():
    for streaming in (True, False):
        assert _wk(ClientS3, FULL, streaming=streaming) == {
            "ContentType": "application/pdf",
            "ContentDisposition": "attachment",
            "CacheControl": "max-age=3600",
            "ContentEncoding": "gzip",
            "Metadata": {"a": "b"},
            "ACL": "public-read",
        }


def test_gcs_write_kwargs():
    for streaming in (True, False):
        assert _wk(GCSClient, NORMALIZED, streaming=streaming) == {
            "content_type": "application/pdf",
            "metadata": {"a": "b"},
            "fixed_key_metadata": {
                "content_disposition": "attachment",
                "cache_control": "max-age=3600",
                "content_encoding": "gzip",
            },
        }


def test_azure_write_kwargs_metadata_only_inline():
    # Azure carries only metadata inline (with the is_directory marker
    # preserved); content settings are applied post-write, not here.
    assert _wk(AzureClient, NORMALIZED, streaming=True) == {
        "metadata": {"is_directory": "false", "a": "b"},
    }


def test_azure_never_pipes():
    assert object.__new__(AzureClient)._can_pipe_upload() is False


def test_gcs_and_azure_reject_write_options():
    # gcsfs/adlfs have no raw write-kwargs passthrough, so the escape hatch
    # must raise rather than crash cryptically or silently drop.
    for cls in (GCSClient, AzureClient):
        for streaming in (True, False):
            with pytest.raises(NotImplementedError, match="write_options"):
                _wk(cls, FULL, streaming=streaming)


def test_local_ignores_all_write_kwargs():
    assert _wk(FileClient, FULL, streaming=True) == {}
    assert _wk(FileClient, FULL, streaming=False) == {}


def test_empty_config_produces_no_kwargs():
    for cls in (ClientS3, GCSClient, AzureClient, FileClient):
        assert _wk(cls, WriteConfig(), streaming=True) == {}
        assert _wk(cls, WriteConfig(), streaming=False) == {}
