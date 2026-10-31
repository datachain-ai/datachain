from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from datachain.lib.video import (
    _display_matrix_rotation,
    _frame_to_ndarray,
    _set_decode_threads,
    _video_stream_rotation,
)


class _FakeFrame:
    def __init__(self, array, rotation=0):
        self._array = array
        # PyAV exposes counterclockwise rotation in degrees.
        self.rotation = -rotation

    def to_ndarray(self, format):
        assert format == "rgb24"
        return self._array


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_display_matrix_rotation(rotation):
    frame = _FakeFrame(np.zeros((2, 3, 3), dtype=np.uint8), rotation)
    assert _display_matrix_rotation(frame) == rotation


def test_frame_to_ndarray_no_rotation_returns_decoded_array():
    array = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    frame = _FakeFrame(array)
    np.testing.assert_array_equal(_frame_to_ndarray(frame), array)


@pytest.mark.parametrize(
    "rotation,expected_k",
    [
        # Clockwise 90/270 == np.rot90 k=3/k=1 (verified against FFmpeg).
        (90, 3),
        (180, 2),
        (270, 1),
    ],
)
def test_frame_to_ndarray_applies_clockwise_rotation(rotation, expected_k):
    array = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    frame = _FakeFrame(array, rotation)

    result = _frame_to_ndarray(frame)

    np.testing.assert_array_equal(result, np.rot90(array, k=expected_k))
    assert result.flags["C_CONTIGUOUS"]
    # 90/270 swap width and height; 180 preserves the shape.
    if rotation in (90, 270):
        assert result.shape == (3, 2, 3)
    else:
        assert result.shape == array.shape


def test_video_stream_rotation_no_frames():
    container = SimpleNamespace(decode=lambda _stream: iter(()))
    assert _video_stream_rotation(container, object()) == 0


def test_video_stream_rotation_decode_error():
    container = SimpleNamespace(decode=Mock(side_effect=RuntimeError))
    assert _video_stream_rotation(container, object()) == 0


@pytest.mark.parametrize(
    "budget,thread_type,thread_count",
    [(None, "AUTO", 2), ("1", "SLICE", 1), ("3", "AUTO", 3)],
)
def test_decode_threads_follow_the_worker_budget(
    budget, thread_type, thread_count, monkeypatch
):
    if budget is None:
        monkeypatch.delenv("DATACHAIN_WORKER_THREADS", raising=False)
    else:
        monkeypatch.setenv("DATACHAIN_WORKER_THREADS", budget)
    monkeypatch.setattr("datachain.lib.video.usable_cpu_count", lambda: 2)
    stream = SimpleNamespace(thread_type="AUTO", thread_count=None)

    _set_decode_threads(stream)

    assert (stream.thread_type, stream.thread_count) == (thread_type, thread_count)
