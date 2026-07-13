import pytest

from datachain.client.writeconfig import WriteConfig


def test_write_config_empty():
    assert WriteConfig().is_empty()
    assert not WriteConfig().has_content_settings()
    assert not WriteConfig(content_type="x").is_empty()
    assert WriteConfig(content_type="x").has_content_settings()
    assert not WriteConfig(metadata={"a": "b"}).has_content_settings()
    assert not WriteConfig(metadata={"a": "b"}).is_empty()


def test_reject_write_options():
    # No-op when the escape hatch is unset; otherwise raises and names the backend.
    WriteConfig().reject_write_options("Azure")
    cfg = WriteConfig(write_options={"ACL": "public-read"})
    for backend in ("GCS", "Azure"):
        with pytest.raises(NotImplementedError, match=backend):
            cfg.reject_write_options(backend)
