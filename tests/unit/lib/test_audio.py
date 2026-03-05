import io
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from datachain.lib.audio import (
    audio_info,
    audio_to_bytes,
    audio_to_np,
    save_audio,
)
from datachain.lib.file import Audio, AudioFile, FileError


def generate_test_wav(
    duration: float = 1.0, sample_rate: int = 16000, frequency: float = 440.0
) -> bytes:
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="wav")
    return buffer.getvalue()


@pytest.fixture
def audio_file(tmp_path, catalog):
    audio_data = generate_test_wav(duration=2.0, sample_rate=16000)
    audio_path = tmp_path / "test_audio.wav"
    audio_path.write_bytes(audio_data)

    file = AudioFile(path=audio_path.name, source=f"file://{tmp_path}")
    file._set_stream(catalog, caching_enabled=False)
    return file


@pytest.fixture
def stereo_audio_file(tmp_path, catalog):
    duration = 1.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, False)

    left_channel = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right_channel = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    stereo_data = np.column_stack([left_channel, right_channel])

    audio_path = tmp_path / "stereo_test.wav"
    sf.write(audio_path, stereo_data, sample_rate, format="wav")

    file = AudioFile(path=audio_path.name, source=f"file://{tmp_path}")
    file._set_stream(catalog, caching_enabled=False)
    return file


def test_audio_info(audio_file):
    result = audio_info(audio_file)

    assert isinstance(result, Audio)
    assert result.sample_rate == 16000
    assert result.channels == 1
    assert abs(result.duration - 2.0) < 0.1
    assert result.samples == 32000


def test_audio_to_np_full(audio_file):
    audio_np, sr = audio_to_np(audio_file)

    assert isinstance(audio_np, np.ndarray)
    assert sr == 16000
    assert len(audio_np.shape) == 1
    assert len(audio_np) == 32000


def test_audio_to_np_partial(audio_file):
    audio_np, sr = audio_to_np(audio_file, start=0.5, duration=0.5)

    assert isinstance(audio_np, np.ndarray)
    assert sr == 16000
    assert len(audio_np) == 8000


def test_audio_to_np_validation(audio_file):
    with pytest.raises(ValueError, match="start must be a non-negative float"):
        audio_to_np(audio_file, start=-1.0)

    with pytest.raises(ValueError, match="duration must be a positive float"):
        audio_to_np(audio_file, duration=0.0)


def test_audio_to_np_with_audiofile(audio_file):
    audio_np, sr = audio_to_np(audio_file)

    assert isinstance(audio_np, np.ndarray)
    assert sr == 16000


def test_audio_to_np_stereo(stereo_audio_file):
    audio_np, sr = audio_to_np(stereo_audio_file)

    assert audio_np.shape == (16000, 2)
    assert sr == 16000


def test_audio_to_bytes(audio_file):
    audio_bytes = audio_to_bytes(audio_file, "wav", 0.0, 1.0)

    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0

    buffer = io.BytesIO(audio_bytes)
    data, sr = sf.read(buffer)
    assert sr == 16000
    assert len(data) == 16000


def test_audio_to_bytes_flac(audio_file):
    audio_bytes = audio_to_bytes(audio_file, "flac")

    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0


def test_save_audio(audio_file, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = save_audio(
        audio_file, destination=str(output_dir), format="wav", start=0.5, end=1.5
    )

    assert isinstance(result, AudioFile)
    expected = output_dir / "test_audio_000500_001500.wav"
    assert expected.exists()
    data, sr = sf.read(expected)
    assert sr == 16000
    assert len(data) == 16000


def test_save_audio_validation(audio_file, tmp_path):
    with pytest.raises(ValueError, match="start time must be non-negative"):
        save_audio(audio_file, destination=str(tmp_path), start=-1.0, end=1.0)

    with pytest.raises(ValueError, match=r"Can't save audio.*invalid time range"):
        save_audio(audio_file, destination=str(tmp_path), start=2.0, end=1.0)


def test_save_audio_full_file(audio_file, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = save_audio(audio_file, destination=str(output_dir), format="wav")

    assert isinstance(result, AudioFile)
    expected = output_dir / "test_audio.wav"
    assert expected.exists()
    data, sr = sf.read(expected)
    assert sr == 16000
    assert len(data) == 32000


def test_save_audio_start_to_end(audio_file, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = save_audio(
        audio_file, destination=str(output_dir), format="wav", start=0.5
    )

    assert isinstance(result, AudioFile)
    expected = output_dir / "test_audio_000500_end.wav"
    assert expected.exists()
    data, sr = sf.read(expected)
    assert sr == 16000
    assert len(data) == 24000


def test_audiofile_save(audio_file, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = audio_file.save(
        destination=str(output_dir), format="wav", start=1.0, end=2.0
    )

    assert isinstance(result, AudioFile)
    expected = output_dir / "test_audio_001000_002000.wav"
    assert expected.exists()
    data, sr = sf.read(expected)
    assert sr == 16000
    assert len(data) == 16000


def test_save_audio_auto_format(tmp_path, catalog):
    audio_data = generate_test_wav(duration=1.0, sample_rate=16000)
    audio_path = tmp_path / "test_audio.flac"
    buffer = io.BytesIO(audio_data)
    temp_data, sr = sf.read(buffer)
    sf.write(audio_path, temp_data, sr, format="flac")

    audio_file = AudioFile(path=audio_path.name, source=f"file://{tmp_path}")
    audio_file._set_stream(catalog, caching_enabled=False)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = save_audio(audio_file, destination=str(output_dir), start=0.0, end=1.0)

    assert isinstance(result, AudioFile)
    expected = output_dir / "test_audio_000000_001000.flac"
    assert expected.exists()


def test_audiofile_save_forwards_client_config(audio_file, tmp_path):
    cfg = {"endpoint_url": "http://custom:9000"}
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    original_get_client = audio_file._catalog.get_client

    def spy_get_client(uri, **kwargs):
        spy_get_client.calls.append((uri, kwargs))
        return original_get_client(uri)

    spy_get_client.calls = []

    with patch.object(audio_file._catalog, "get_client", side_effect=spy_get_client):
        audio_file.save(destination=str(output_dir), format="wav", client_config=cfg)

    config_calls = [kw for _, kw in spy_get_client.calls if kw.get("endpoint_url")]
    assert len(config_calls) == 1
    assert config_calls[0]["endpoint_url"] == "http://custom:9000"


def test_audio_info_file_error(audio_file):
    with patch("datachain.lib.audio.sf.info", side_effect=Exception("Test error")):
        with pytest.raises(
            FileError, match="unable to extract metadata from audio file"
        ):
            audio_info(audio_file)


def test_audio_to_np_file_error(audio_file):
    with patch("datachain.lib.audio.sf.info", side_effect=Exception("Test error")):
        with pytest.raises(FileError, match="unable to read audio fragment"):
            audio_to_np(audio_file)


def test_save_audio_file_error(audio_file, tmp_path):
    with patch(
        "datachain.lib.audio.audio_to_bytes", side_effect=Exception("Test error")
    ):
        with pytest.raises(FileError, match="unable to save audio fragment"):
            save_audio(audio_file, destination=str(tmp_path), start=0.0, end=1.0)


@pytest.mark.parametrize("start,duration", [(0.0, 1.0), (0.5, 0.5), (1.0, 1.0)])
def test_audio_to_np_different_durations(audio_file, start, duration):
    audio_np, sr = audio_to_np(audio_file, start=start, duration=duration)

    assert isinstance(audio_np, np.ndarray)
    assert sr == 16000
    expected_samples = int(duration * 16000)
    assert len(audio_np) == expected_samples


@pytest.mark.parametrize("format_type", ["wav", "flac", "ogg"])
def test_audio_to_bytes_formats(audio_file, format_type):
    audio_bytes = audio_to_bytes(audio_file, format_type)

    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0


@pytest.mark.parametrize(
    "format_str,subtype,file_ext,expected_format,expected_bit_rate",
    [
        # Direct format mappings from soundfile
        ("WAV", "PCM_16", "wav", "wav", 16 * 16000),
        ("FLAC", "PCM_16", "flac", "flac", 16 * 16000),
        ("OGG", "VORBIS", "ogg", "ogg", -1),
        ("AIFF", "PCM_24", "aiff", "aiff", 24 * 16000),
        # Format fallback to file extension when subtype is PCM
        (None, "PCM_16", "wav", "wav", 16 * 16000),
        (None, "PCM_24", "aiff", "aiff", 24 * 16000),
        (None, "PCM_S16LE", "au", "au", 16 * 16000),
        (None, "PCM_F32LE", "wav", "wav", 32 * 16000),
        # Unknown format with extension falls back to extension
        (None, "UNKNOWN_CODEC", "mp3", "mp3", -1),
        ("", "UNKNOWN_CODEC", "flac", "flac", -1),
        # Files without extension should fall back to "unknown"
        (None, "PCM_16", "", "unknown", 16 * 16000),
        ("", "UNKNOWN_CODEC", "", "unknown", -1),
    ],
)
def test_audio_info_format_detection(
    tmp_path, catalog, format_str, subtype, file_ext, expected_format, expected_bit_rate
):
    filename = f"test_audio.{file_ext}" if file_ext else "test_audio"
    audio_data = generate_test_wav(duration=0.1, sample_rate=16000)
    audio_path = tmp_path / filename
    audio_path.write_bytes(audio_data)

    audio_file = AudioFile(path=audio_path.name, source=f"file://{tmp_path}")
    audio_file._set_stream(catalog, caching_enabled=False)

    with patch("datachain.lib.audio.sf.info") as mock_info:
        mock_info.return_value.samplerate = 16000
        mock_info.return_value.channels = 1
        mock_info.return_value.frames = 1600
        mock_info.return_value.duration = 0.1
        mock_info.return_value.format = format_str
        mock_info.return_value.subtype = subtype

        result = audio_info(audio_file)

        assert result.format == expected_format
        assert result.codec == subtype
        assert result.bit_rate == expected_bit_rate


def test_audio_info_stereo(stereo_audio_file):
    result = audio_info(stereo_audio_file)

    assert isinstance(result, Audio)
    assert result.sample_rate == 16000
    assert result.channels == 2
    assert result.samples == 16000
