from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from datachain.lib.video import (
    _display_matrix_rotation,
    _frame_to_ndarray,
    _video_stream_rotation,
)

# Valid DISPLAYMATRIX affine matrices per angle (end-to-end direction is checked
# against FFmpeg in tests/func/test_video.py).
_FP = 1 << 16
_W = 1 << 30
DISPLAY_MATRICES = {
    0: [_FP, 0, 0, 0, _FP, 0, 0, 0, _W],
    90: [0, _FP, 0, -_FP, 0, 0, 0, 0, _W],
    180: [-_FP, 0, 0, 0, -_FP, 0, 0, 0, _W],
    270: [0, -_FP, 0, _FP, 0, 0, 0, 0, _W],
}


class _FakeSideData:
    def __init__(self, matrix: list[int]):
        self._bytes = np.asarray(matrix, dtype="<i4").tobytes()

    def __bytes__(self) -> bytes:
        return self._bytes


class _RawSideData:
    """Side data wrapping arbitrary bytes (for malformed-buffer tests)."""

    def __init__(self, raw: bytes):
        self._bytes = raw

    def __bytes__(self) -> bytes:
        return self._bytes


class _FakeFrame:
    """Minimal stand-in for av.VideoFrame for rotation tests."""

    def __init__(self, array: np.ndarray, side_data=None):
        from datachain.lib.video import _DISPLAYMATRIX

        self._array = array
        self.side_data = {} if side_data is None else {_DISPLAYMATRIX: side_data}

    def to_ndarray(self, format: str) -> np.ndarray:
        assert format == "rgb24"
        return self._array


def _frame(array: np.ndarray, matrix: list[int] | None = None) -> _FakeFrame:
    return _FakeFrame(array, None if matrix is None else _FakeSideData(matrix))


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_display_matrix_rotation(rotation):
    frame = _frame(np.zeros((2, 3, 3), dtype=np.uint8), DISPLAY_MATRICES[rotation])
    assert _display_matrix_rotation(frame) == rotation


def test_display_matrix_rotation_no_side_data():
    frame = _frame(np.zeros((2, 3, 3), dtype=np.uint8))
    assert _display_matrix_rotation(frame) == 0


def test_display_matrix_rotation_degenerate_matrix():
    frame = _frame(np.zeros((2, 3, 3), dtype=np.uint8), [0] * 9)
    assert _display_matrix_rotation(frame) == 0


@pytest.mark.parametrize(
    "raw",
    [
        b"",  # empty
        b"\x00" * 10,  # length not a multiple of int32 size
        b"\x01" * 35,  # one byte short of a full matrix
        b"\x00" * 16,  # well-formed int32 stream but fewer than 9 entries
    ],
)
def test_display_matrix_rotation_malformed_buffer(raw):
    # A truncated/odd-length side-data buffer must never crash get_np()/get_info().
    frame = _FakeFrame(np.zeros((2, 3, 3), dtype=np.uint8), _RawSideData(raw))
    assert _display_matrix_rotation(frame) == 0


def test_frame_to_ndarray_no_rotation_returns_decoded_array():
    array = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    frame = _frame(array, DISPLAY_MATRICES[0])
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
    frame = _frame(array, DISPLAY_MATRICES[rotation])

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
