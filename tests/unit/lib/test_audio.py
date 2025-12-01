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
    """Generate a simple sine wave WAV file as bytes."""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="wav")
    return buffer.getvalue()


@pytest.fixture
def audio_file(tmp_path, catalog):
    """Create a temporary audio file for testing."""
    audio_data = generate_test_wav(duration=2.0, sample_rate=16000)
    audio_path = tmp_path / "test_audio.wav"
    audio_path.write_bytes(audio_data)

    file = AudioFile(path=str(audio_path), source="file://")
    # _set_stream is required to enable file.open() operations
    file._set_stream(catalog, caching_enabled=False)
    return file


@pytest.fixture
def stereo_audio_file(tmp_path, catalog):
    """Create a temporary stereo audio file for testing."""
    duration = 1.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, False)

    # Create stereo signal (left and right channels)
    left_channel = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right_channel = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    stereo_data = np.column_stack([left_channel, right_channel])

    audio_path = tmp_path / "stereo_test.wav"
    sf.write(audio_path, stereo_data, sample_rate, format="wav")

    # Create a real AudioFile object
    file = AudioFile(path=str(audio_path), source="file://")
    # _set_stream is required to enable file.open() operations
    file._set_stream(catalog, caching_enabled=False)
    return file


def test_audio_info(audio_file):
    """Test audio metadata extraction."""
    result = audio_info(audio_file)

    assert isinstance(result, Audio)
    assert result.sample_rate == 16000
    assert result.channels == 1
    assert abs(result.duration - 2.0) < 0.1  # Allow small tolerance
    assert result.samples == 32000


def test_audio_fragment_np_full(audio_file):
    """Test loading full audio fragment as numpy array."""
    audio_np, sr = audio_to_np(audio_file)

    assert isinstance(audio_np, np.ndarray)
    assert sr == 16000
    assert len(audio_np.shape) == 1  # Mono audio should be 1D
    assert len(audio_np) == 32000  # 2 seconds at 16kHz


def test_audio_fragment_np_partial(audio_file):
    """Test loading partial audio fragment."""
    # Load 0.5 seconds starting from 0.5 seconds
    audio_np, sr = audio_to_np(audio_file, start=0.5, duration=0.5)

    assert isinstance(audio_np, np.ndarray)
    assert sr == 16000
    assert len(audio_np) == 8000  # 0.5 seconds at 16kHz


def test_audio_fragment_np_validation(audio_file):
    """Test input validation for audio_fragment_np."""
    with pytest.raises(ValueError, match="start must be a non-negative float"):
        audio_to_np(audio_file, start=-1.0)

    with pytest.raises(ValueError, match="duration must be a positive float"):
        audio_to_np(audio_file, duration=0.0)


def test_audio_fragment_np_with_file_interface(audio_file):
    """Test audio_fragment_np with File interface."""
    # Test that the audio_file fixture (which is already an AudioFile) works
    audio_np, sr = audio_to_np(audio_file)

    assert isinstance(audio_np, np.ndarray)
    assert sr == 16000


def test_audio_fragment_np_multichannel(stereo_audio_file):
    """Test multichannel audio handling."""
    audio_np, sr = audio_to_np(stereo_audio_file)

    # Should be transposed to (samples, channels)
    assert audio_np.shape == (16000, 2)  # 1 second at 16kHz, 2 channels
    assert sr == 16000


def test_audio_to_bytes(audio_file):
    """Test converting audio fragment to bytes."""
    audio_bytes = audio_to_bytes(audio_file, "wav", 0.0, 1.0)

    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0

    # Verify it's a valid WAV file by loading it back
    buffer = io.BytesIO(audio_bytes)
    data, sr = sf.read(buffer)
    assert sr == 16000
    assert len(data) == 16000  # 1 second at 16kHz


def test_audio_to_bytes_custom_format(audio_file):
    """Test converting audio fragment to different format."""
    audio_bytes = audio_to_bytes(audio_file, "flac")

    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0


def test_save_audio(audio_file, tmp_path):
    """Test saving audio (fragment extraction mode)."""
    with patch("datachain.lib.file.AudioFile.upload") as mock_upload:
        # Mock the upload to return a new AudioFile
        mock_uploaded_file = AudioFile(
            path="test_audio_000500_001500.wav", source="file://"
        )
        mock_upload.return_value = mock_uploaded_file

        result = save_audio(
            audio_file, output=str(tmp_path), format="wav", start=0.5, end=1.5
        )

        # Verify AudioFile.upload was called
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args

        # Check the uploaded data is bytes
        assert isinstance(call_args[0][0], bytes)

        # Check the filename pattern
        filename = call_args[0][1]
        assert "test_audio_000500_001500.wav" in filename

        # Check catalog is passed
        assert call_args[1]["catalog"] == audio_file._catalog

        assert result == mock_uploaded_file


def test_save_audio_validation(audio_file, tmp_path):
    """Test input validation for save_audio."""
    with pytest.raises(ValueError, match="start time must be non-negative"):
        save_audio(audio_file, output=str(tmp_path), start=-1.0, end=1.0)

    with pytest.raises(ValueError, match=r"Can't save audio.*invalid time range"):
        save_audio(audio_file, output=str(tmp_path), start=2.0, end=1.0)


def test_save_audio_full_file_conversion(audio_file, tmp_path):
    """Test saving audio with full file conversion (end=None)."""
    with patch("datachain.lib.file.AudioFile.upload") as mock_upload:
        mock_uploaded_file = AudioFile(path="test_audio.wav", source="file://")
        mock_upload.return_value = mock_uploaded_file

        result = save_audio(audio_file, output=str(tmp_path), format="wav")

        # Verify AudioFile.upload was called
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args

        # Check that the audio bytes were generated
        assert isinstance(call_args[0][0], bytes)

        # Check that the output file has no timestamps (full file conversion)
        output_file = call_args[0][1]
        assert output_file == str(tmp_path) + "/test_audio.wav"

        assert result == mock_uploaded_file


def test_save_audio_start_to_end(audio_file, tmp_path):
    """Test saving audio from start time to end of file (end=None, start>0)."""
    with patch("datachain.lib.file.AudioFile.upload") as mock_upload:
        mock_uploaded_file = AudioFile(
            path="test_audio_000500_end.wav", source="file://"
        )
        mock_upload.return_value = mock_uploaded_file

        result = save_audio(audio_file, output=str(tmp_path), format="wav", start=0.5)

        # Verify AudioFile.upload was called
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args

        # Check that the audio bytes were generated
        assert isinstance(call_args[0][0], bytes)

        # Check that the output file has timestamp and "_end" suffix
        output_file = call_args[0][1]
        assert output_file == str(tmp_path) + "/test_audio_000500_end.wav"

        assert result == mock_uploaded_file


def test_audiofile_save(audio_file, tmp_path):
    with patch("datachain.lib.file.AudioFile.upload") as mock_upload:
        mock_uploaded_file = AudioFile(path="test_audio.mp3", source="file://")
        mock_upload.return_value = mock_uploaded_file

        result = audio_file.save(output=str(tmp_path), format="mp3", start=1.0, end=2.0)

        # Verify AudioFile.upload was called
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args

        # Check that the audio bytes were generated
        assert isinstance(call_args[0][0], bytes)

        # Check that the output file has correct format and timestamps
        output_file = call_args[0][1]
        assert output_file == str(tmp_path) + "/test_audio_001000_002000.mp3"

        assert result == mock_uploaded_file


def test_save_audio_auto_format(tmp_path, catalog):
    """Test saving audio with auto-detected format."""
    audio_data = generate_test_wav(duration=1.0, sample_rate=16000)
    audio_path = tmp_path / "test_audio.flac"
    buffer = io.BytesIO(audio_data)
    temp_data, sr = sf.read(buffer)
    sf.write(audio_path, temp_data, sr, format="flac")

    audio_file = AudioFile(path=str(audio_path), source="file://")
    # _set_stream is required to enable file.open() operations
    audio_file._set_stream(catalog, caching_enabled=False)

    with patch("datachain.lib.file.AudioFile.upload") as mock_upload:
        mock_upload.return_value = AudioFile(path="test_output.flac", source="file://")

        save_audio(audio_file, output=str(tmp_path), start=0.0, end=1.0)

        # Should use format from file extension
        call_args = mock_upload.call_args
        filename = call_args[0][1]
        assert filename.endswith(".flac")


def test_audio_info_file_error(audio_file):
    """Test audio_info handles file errors properly."""
    with patch("datachain.lib.audio.sf.info", side_effect=Exception("Test error")):
        with pytest.raises(
            FileError, match="unable to extract metadata from audio file"
        ):
            audio_info(audio_file)


def test_audio_fragment_np_file_error(audio_file):
    """Test audio_fragment_np handles file errors properly."""
    with patch("datachain.lib.audio.sf.info", side_effect=Exception("Test error")):
        with pytest.raises(FileError, match="unable to read audio fragment"):
            audio_to_np(audio_file)


def test_save_audio_file_error(audio_file, tmp_path):
    """Test save_audio handles errors properly."""
    with patch(
        "datachain.lib.audio.audio_to_bytes", side_effect=Exception("Test error")
    ):
        with pytest.raises(FileError, match="unable to save audio fragment"):
            save_audio(audio_file, output=str(tmp_path), start=0.0, end=1.0)


@pytest.mark.parametrize("start,duration", [(0.0, 1.0), (0.5, 0.5), (1.0, 1.0)])
def test_audio_fragment_np_different_durations(audio_file, start, duration):
    """Test audio loading with different start times and durations."""
    audio_np, sr = audio_to_np(audio_file, start=start, duration=duration)

    assert isinstance(audio_np, np.ndarray)
    assert sr == 16000
    expected_samples = int(duration * 16000)
    assert len(audio_np) == expected_samples


@pytest.mark.parametrize("format_type", ["wav", "flac", "ogg"])
def test_audio_to_bytes_formats(audio_file, format_type):
    """Test audio conversion to different formats."""
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
    """Test audio format detection for different file extensions and formats."""
    # Create a test audio file with the specified extension
    filename = f"test_audio.{file_ext}" if file_ext else "test_audio"
    audio_data = generate_test_wav(duration=0.1, sample_rate=16000)
    audio_path = tmp_path / filename
    audio_path.write_bytes(audio_data)

    audio_file = AudioFile(path=str(audio_path), source="file://")
    audio_file._set_stream(catalog, caching_enabled=False)

    # Mock soundfile.info to return controlled format
    with patch("datachain.lib.audio.sf.info") as mock_info:
        mock_info.return_value.samplerate = 16000
        mock_info.return_value.channels = 1
        mock_info.return_value.frames = 1600  # 0.1 seconds
        mock_info.return_value.duration = 0.1
        mock_info.return_value.format = format_str
        mock_info.return_value.subtype = subtype

        result = audio_info(audio_file)

        assert result.format == expected_format
        assert result.codec == subtype
        assert result.bit_rate == expected_bit_rate


def test_audio_info_stereo(stereo_audio_file):
    """Test audio info extraction for stereo files."""
    result = audio_info(stereo_audio_file)

    # Verify the returned Audio object has expected metadata
    assert isinstance(result, Audio)
    assert result.sample_rate == 16000
    assert result.channels == 2  # Stereo
    assert result.samples == 16000
