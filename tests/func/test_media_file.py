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


def test_media_file_metadata_reuses_persistent_cache(test_session, media_file):
    kind, path = media_file
    chain = dc.read_storage(path.as_uri(), type=kind, session=test_session)

    def cache_and_describe(file) -> dict:
        file.read_bytes()
        return file.get_info().model_dump()

    expected = chain.settings(cache=True).map(info=cache_and_describe).to_values("info")
    path.unlink()
    assert (
        chain.settings(prefetch=0)
        .map(info=lambda file: file.get_info().model_dump(), output=dict)
        .to_values("info")
        == expected
    )


def test_get_info_with_prefetch_0_leaves_cache_empty_until_open(
    test_session, media_file
):
    kind, path = media_file

    def content(file) -> int:
        assert file.get_info().model_dump()
        assert file.get_local_path() is None
        with file.open() as stream:
            value = stream.read()
        assert file.get_local_path()
        return len(value)

    result = (
        dc.read_storage(path.as_uri(), type=kind, session=test_session)
        .settings(cache=True, prefetch=0)
        .map(size=content)
        .to_list("size")
    )
    assert result == [(path.stat().st_size,)]
