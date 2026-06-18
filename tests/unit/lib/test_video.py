import numpy as np
import pytest

from datachain.lib.video import _display_matrix_rotation, _frame_to_ndarray

# DISPLAYMATRIX side data (9 int32, 16.16 fixed point) captured from real files
# rewrapped by FFmpeg. The last entry is in 2.30 fixed point and is unused here.
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


class _FakeFrame:
    """Minimal stand-in for av.VideoFrame for rotation tests."""

    def __init__(self, array: np.ndarray, matrix: list[int] | None = None):
        from datachain.lib.video import _DISPLAYMATRIX

        self._array = array
        self.side_data = (
            {} if matrix is None else {_DISPLAYMATRIX: _FakeSideData(matrix)}
        )

    def to_ndarray(self, format: str) -> np.ndarray:
        assert format == "rgb24"
        return self._array


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_display_matrix_rotation(rotation):
    frame = _FakeFrame(np.zeros((2, 3, 3), dtype=np.uint8), DISPLAY_MATRICES[rotation])
    assert _display_matrix_rotation(frame) == rotation


def test_display_matrix_rotation_no_side_data():
    frame = _FakeFrame(np.zeros((2, 3, 3), dtype=np.uint8))
    assert _display_matrix_rotation(frame) == 0


def test_display_matrix_rotation_degenerate_matrix():
    # A zero matrix has no defined rotation; degrade to no rotation.
    frame = _FakeFrame(np.zeros((2, 3, 3), dtype=np.uint8), [0] * 9)
    assert _display_matrix_rotation(frame) == 0


def test_frame_to_ndarray_no_rotation_returns_decoded_array():
    array = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    frame = _FakeFrame(array, DISPLAY_MATRICES[0])
    np.testing.assert_array_equal(_frame_to_ndarray(frame), array)


@pytest.mark.parametrize(
    "rotation,expected_k",
    [
        # np.rot90 rotates counter-clockwise; a clockwise display rotation of
        # 90/270 degrees corresponds to k=3/k=1 (verified against FFmpeg).
        (90, 3),
        (180, 2),
        (270, 1),
    ],
)
def test_frame_to_ndarray_applies_clockwise_rotation(rotation, expected_k):
    array = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    frame = _FakeFrame(array, DISPLAY_MATRICES[rotation])

    result = _frame_to_ndarray(frame)

    np.testing.assert_array_equal(result, np.rot90(array, k=expected_k))
    assert result.flags["C_CONTIGUOUS"]
    # 90/270 swap width and height; 180 preserves the shape.
    if rotation in (90, 270):
        assert result.shape == (3, 2, 3)
    else:
        assert result.shape == array.shape
