import io
from collections.abc import Iterator

import numpy as np
import soundfile as sf

import datachain as dc
from datachain.lib.audio import audio_to_np
from datachain.lib.file import Audio, AudioFile, AudioFragment


def generate_test_wav(
    duration: float = 1.0, sample_rate: int = 16000, frequency: float = 440.0
) -> bytes:
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="wav")
    return buffer.getvalue()


def extract_audio_info(file: AudioFile) -> Audio:
    return file.get_info()


def generate_fragments(file: AudioFile) -> Iterator[AudioFragment]:
    fragment_duration = 0.5
    yield from file.get_fragments(duration=fragment_duration)


def test_audio_datachain_workflow(test_session, tmp_path):
    audio_files = []
    for i, (duration, freq) in enumerate([(2.0, 440.0), (3.0, 880.0)]):
        audio_data = generate_test_wav(
            duration=duration, sample_rate=16000, frequency=freq
        )
        audio_path = tmp_path / f"test_audio_{i}.wav"
        audio_path.write_bytes(audio_data)
        audio_files.append(str(audio_path))

    # Use DataChain to read the audio files from filesystem
    chain = dc.read_storage(tmp_path.as_uri(), session=test_session, type="audio")

    # Extract Audio metadata
    chain_with_info = chain.map(
        info=extract_audio_info,
        params=["file"],
        output={"info": Audio},
    )

    # Verify we have the expected files and metadata
    results = list(chain_with_info.to_iter("file", "info"))
    assert len(results) == 2

    # Check that all files have expected audio metadata
    for file, info in results:
        assert isinstance(file, AudioFile)
        assert isinstance(info, Audio)
        assert info.sample_rate == 16000
        assert info.channels == 1
        assert info.duration > 0
        assert info.samples > 0
        assert info.format == "wav"

    # Generate audio fragments
    fragments_chain = chain.gen(
        fragment=generate_fragments,
        params=["file"],
        output={"fragment": AudioFragment},
    )

    # Get all fragments and verify their properties
    fragments = list(fragments_chain.to_values("fragment"))

    # We should have multiple fragments (2s file creates 4, 3s file creates 6)
    expected_total_fragments = 4 + 6  # 10 total fragments
    assert len(fragments) == expected_total_fragments

    # Verify each fragment has correct properties
    for fragment in fragments:
        assert isinstance(fragment, AudioFragment)
        assert isinstance(fragment.audio, AudioFile)
        assert fragment.start >= 0
        assert fragment.end > fragment.start
        assert fragment.end - fragment.start <= 0.5  # fragment duration

    # Test saving fragments
    temp_output_dir = tmp_path / "fragments"
    temp_output_dir.mkdir()

    # Save first fragment as a test
    first_fragment = fragments[0]
    saved_file = first_fragment.save(str(temp_output_dir))

    assert isinstance(saved_file, AudioFile)
    assert saved_file.path.startswith("test_audio_")
    assert saved_file.path.endswith(".wav")

    # Verify the saved fragment has expected content
    saved_info = saved_file.get_info()
    expected_duration = first_fragment.end - first_fragment.start
    # Allow small tolerance
    assert abs(saved_info.duration - expected_duration) < 0.1
    assert saved_info.sample_rate == 16000
    assert saved_info.channels == 1

    # Verify the audio content matches by comparing numpy arrays
    original_np, original_sr = first_fragment.get_np()
    saved_np, saved_sr = audio_to_np(saved_file)

    assert original_sr == saved_sr == 16000
    assert len(original_np) == len(saved_np)
    # Allow some tolerance for audio processing differences
    assert np.allclose(original_np, saved_np, atol=1e-3)
