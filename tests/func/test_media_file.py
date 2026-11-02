from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import datachain as dc


@pytest.fixture(params=["video", "audio", "image"])
def media_file(request, tmp_path):
    kind = request.param
    data = Path(__file__).parent / "data"
    if kind == "audio":
        path = tmp_path / "sample.wav"
        sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    else:
        source = data / (
            "Big_Buck_Bunny_360_10s_1MB.mp4" if kind == "video" else "lena.jpg"
        )
        path = tmp_path / source.name
        path.write_bytes(source.read_bytes())
    return kind, path


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("prefetch", [None, 0])
@pytest.mark.parametrize("operation", ["map", "gen"])
def test_media_file_metadata_respects_prefetch(
    test_session, media_file, cache, prefetch, operation, mocker
):
    kind, path = media_file
    download = mocker.spy(dc.File, "ensure_cached")
    chain = dc.read_storage(path.as_uri(), type=kind, session=test_session)
    settings = {"cache": cache}
    if prefetch is not None:
        settings["prefetch"] = prefetch
    chain = chain.settings(**settings)

    def metadata(file) -> dict:
        info = file.get_info().model_dump()
        assert file._catalog.cache.contains(file) == (prefetch is None)
        return info

    if operation == "gen":

        def generate(file):
            yield metadata(file)

        result = chain.gen(info=generate, output=dict).to_list("info")
    else:
        result = chain.map(info=metadata).to_list("info")
    assert len(result) == 1
    assert result[0][0]
    download.assert_not_called()


@pytest.mark.parametrize("cache", [False, True])
def test_media_file_metadata_then_content_in_one_udf_stage(
    test_session, media_file, cache
):
    kind, path = media_file

    def content(file) -> int:
        assert file.get_info().model_dump()
        assert not file._catalog.cache.contains(file)
        with file.open() as stream:
            value = stream.read()
        assert file._catalog.cache.contains(file) is cache
        return len(value)

    result = (
        dc.read_storage(path.as_uri(), type=kind, session=test_session)
        .settings(cache=cache, prefetch=0)
        .map(size=content)
        .to_list("size")
    )
    assert result == [(path.stat().st_size,)]


def test_media_file_metadata_reuses_persistent_cache(test_session, media_file, mocker):
    from datachain.client.fsspec import Client

    kind, path = media_file
    chain = dc.read_storage(path.as_uri(), type=kind, session=test_session)
    chain.settings(cache=True).map(
        size=lambda file: len(file.read_bytes()), output=int
    ).to_list("size")
    remote = mocker.patch.object(
        Client, "_open_remote", side_effect=AssertionError("uncached open")
    )
    assert chain.map(
        info=lambda file: file.get_info().model_dump(), output=dict
    ).to_list("info")
    remote.assert_not_called()


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("length", [-1, 0, 4])
def test_read_bytes_preserves_full_read_cache_policy(
    test_session, tmp_path, cache, length
):
    path = tmp_path / "content.bin"
    path.write_bytes(b"hello world")

    def read(file: dc.File) -> bytes:
        value = file.read_bytes(length)
        assert file._catalog.cache.contains(file) == (cache and length < 0)
        return value

    result = (
        dc.read_storage(path, session=test_session)
        .settings(cache=cache, prefetch=0)
        .map(value=read)
        .to_values("value")
    )
    assert result == [b"hello world" if length < 0 else b"hello world"[:length]]
