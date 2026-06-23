import os
import posixpath
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from math import atan2, ceil, degrees, floor, hypot
from typing import Any, cast

import numpy as np
from fsspec.utils import stringify_path
from numpy import ndarray
from PIL import Image as PilImage

from datachain.lib.file import (
    File,
    FileError,
    VFileRegistry,
    Video,
    VideoFile,
    VideoFrame,
)

try:
    import av
    import ffmpeg
    from av.sidedata.sidedata import Type as _SideDataType
except ImportError as exc:
    raise ImportError(
        "Missing dependencies for processing video.\n"
        "To install run:\n\n"
        "  pip install 'datachain[video]'\n"
    ) from exc

_DISPLAYMATRIX = _SideDataType.DISPLAYMATRIX

VIDEO_TIME_BASE = 1_000_000
# DISPLAYMATRIX rotation entries are 16.16 fixed point (FFmpeg libavutil/display.h).
DISPLAY_MATRIX_FIXED_POINT = 1 << 16
FFMPEG_FRAGMENT_TIMEOUT_MIN = 60.0
FFMPEG_FRAGMENT_TIMEOUT_FACTOR = 10.0
FFMPEG_FRAGMENT_EXECUTABLE = "ffmpeg"
FFMPEG_FRAGMENT_ARGS = ("-nostdin", "-hide_banner", "-loglevel", "error")
FRAME_INDEX_EPSILON = 1e-9


class _SeekLandedAfterStartError(Exception):
    pass


def _display_matrix_rotation(frame) -> int:
    """
    Clockwise display rotation (0/90/180/270) from a frame's DISPLAYMATRIX,
    matching what FFmpeg/OpenCV apply on playback (PyAV does not autorotate).

    Only right-angle rotation is applied: scale is normalized out and any
    flipped or off-axis matrix is snapped to the nearest 0/90/180/270, as
    OpenCV does. Returns 0 when there is no rotation metadata.
    """
    try:
        side_data = frame.side_data.get(_DISPLAYMATRIX)
        if side_data is None:
            return 0
        matrix = np.frombuffer(bytes(side_data), dtype=np.int32)  # native int32
    except Exception:  # noqa: BLE001  # malformed side data -> no rotation
        return 0
    if matrix.size < 9:
        return 0

    # Rotation angle, normalizing out scale (FFmpeg av_display_rotation_get).
    a = matrix[0] / DISPLAY_MATRIX_FIXED_POINT
    b = matrix[1] / DISPLAY_MATRIX_FIXED_POINT
    c = matrix[3] / DISPLAY_MATRIX_FIXED_POINT
    d = matrix[4] / DISPLAY_MATRIX_FIXED_POINT
    scale_x = hypot(a, c)
    scale_y = hypot(b, d)
    if scale_x == 0 or scale_y == 0:
        return 0

    theta = degrees(atan2(b / scale_y, a / scale_x))
    return round(theta / 90) % 4 * 90


def _frame_to_ndarray(frame) -> ndarray:
    """Decode a frame to an RGB array, applying display-matrix rotation."""
    array = frame.to_ndarray(format="rgb24")
    rotation = _display_matrix_rotation(frame)
    if rotation == 0:
        return array
    # np.rot90 is counter-clockwise; negate k for clockwise.
    return np.ascontiguousarray(np.rot90(array, k=-(rotation // 90) % 4))


def _video_stream_rotation(container, video_stream) -> int:
    """Stream display rotation. Rotation is a per-stream property, but PyAV
    exposes the DISPLAYMATRIX only on frames, so the first frame is decoded.
    Decode failure or absence -> 0, never blocks metadata."""
    try:
        frame = next(container.decode(video_stream), None)
    except Exception:  # noqa: BLE001
        return 0
    if frame is None:
        return 0
    return _display_matrix_rotation(frame)


def video_info(file: File | VideoFile, video_stream_index: int = 0) -> Video:
    """
    Returns video file information.

    Args:
        file (VideoFile): Video file object.
        video_stream_index: Zero-based index among video streams.

    Returns:
        Video: Video file information.
    """
    file = file.as_video_file()
    _validate_video_stream_index(video_stream_index)

    try:
        with file.open() as f:
            with av.open(f) as container:
                video_stream = _video_stream(
                    container,
                    video_stream_index,
                    file.source,
                    file.path,
                    "unable to extract metadata from video file",
                )
                fps = _video_fps(video_stream)
                duration = _video_duration(container, video_stream)

                width = int(video_stream.width or 0)
                height = int(video_stream.height or 0)
                # 90/270 rotation swaps the display dimensions, as get_np() does.
                if _video_stream_rotation(container, video_stream) in (90, 270):
                    width, height = height, width
                frames = int(video_stream.frames or 0)
                if frames <= 0 and duration > 0 and fps > 0:
                    frames = ceil(duration * fps)

                format_name = container.format.name or ""
                codec_name = video_stream.codec_context.name or ""
    except FileError:
        raise
    except Exception as exc:
        raise FileError(
            "unable to extract metadata from video file", file.source, file.path
        ) from exc

    return Video(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        frames=frames,
        format=format_name,
        codec=codec_name,
    )


def _validate_video_stream_index(video_stream_index: int) -> None:
    if video_stream_index < 0:
        raise ValueError("video_stream_index must be a non-negative integer")


def _video_stream(
    container, video_stream_index: int, source: str, path: str, error: str
):
    video_streams = list(container.streams.video)
    if video_stream_index >= len(video_streams):
        raise FileError(
            f"{error}: video_stream_index {video_stream_index} is out of range",
            source,
            path,
        )
    return video_streams[video_stream_index]


def _video_fps(video_stream) -> float:
    for rate in (
        video_stream.average_rate,
        video_stream.base_rate,
        video_stream.guessed_rate,
    ):
        if rate:
            return float(rate)
    return -1.0


def _video_duration(container, video_stream) -> float:
    if container.duration is not None:
        return float(container.duration / VIDEO_TIME_BASE)
    if video_stream.duration is not None and video_stream.time_base is not None:
        return float(video_stream.duration * video_stream.time_base)
    return -1.0


def _has_constant_frame_rate(video_stream) -> bool:
    average_rate = video_stream.average_rate
    base_rate = video_stream.base_rate
    guessed_rate = video_stream.guessed_rate
    if not average_rate or not base_rate or average_rate != base_rate:
        return False
    return not guessed_rate or guessed_rate == average_rate


def _stream_start_time(video_stream) -> float:
    if video_stream.start_time is None or video_stream.time_base is None:
        return 0.0
    return float(video_stream.start_time * video_stream.time_base)


def _seekable(file_obj) -> bool:
    seekable = getattr(file_obj, "seekable", None)
    if seekable is not None:
        try:
            return bool(seekable())
        except OSError:
            return False
    return hasattr(file_obj, "seek") and hasattr(file_obj, "tell")


def _seek_to_frame(container, file_obj, video_stream, frame: int, fps: float) -> bool:
    if (
        frame <= 0
        or fps <= 0
        or video_stream.start_time not in (None, 0)
        or video_stream.time_base is None
        or not _seekable(file_obj)
        or not _has_constant_frame_rate(video_stream)
    ):
        return False

    offset = (video_stream.start_time or 0) + int(
        (frame / fps) / video_stream.time_base
    )
    try:
        container.seek(offset, backward=True, any_frame=False, stream=video_stream)
    except (av.error.FFmpegError, OSError, ValueError):
        return False
    return True


def _decoded_frame_index(decoded_frame, fps: float, video_stream) -> int:
    if decoded_frame.pts is not None and decoded_frame.time_base is not None:
        timestamp = float(decoded_frame.pts * decoded_frame.time_base)
    elif decoded_frame.time is not None:
        timestamp = float(decoded_frame.time)
    else:
        return 0

    frame_position = (timestamp - _stream_start_time(video_stream)) * fps
    return max(0, floor(frame_position + FRAME_INDEX_EPSILON))


def _find_decoded_frame(container, video_stream, frame: int, fps: float, seeked: bool):
    start_frame_index = 0
    for decoded_offset, decoded_frame in enumerate(container.decode(video_stream)):
        if decoded_offset == 0 and seeked:
            start_frame_index = _decoded_frame_index(decoded_frame, fps, video_stream)
            if start_frame_index > frame:
                return None

        frame_index = start_frame_index + decoded_offset
        if frame_index == frame:
            return decoded_frame
        if frame_index > frame:
            return None

    return None


def _decoded_frames(container, video_stream, start: int, fps: float, seeked: bool):
    start_frame_index = 0
    for decoded_offset, decoded_frame in enumerate(container.decode(video_stream)):
        if decoded_offset == 0 and seeked:
            start_frame_index = _decoded_frame_index(decoded_frame, fps, video_stream)
            if start_frame_index > start:
                raise _SeekLandedAfterStartError

        yield start_frame_index + decoded_offset, decoded_frame


def _video_frames_from_decoded(
    video: VideoFile,
    decoded_frames,
    start: int,
    end: int,
    step: int,
    fps: float,
    video_stream_index: int,
) -> Iterator[VideoFrame]:
    for frame_index, frame in decoded_frames:
        if frame_index >= end:
            break
        if frame_index < start or (frame_index - start) % step:
            continue

        video_frame = VideoFrame(
            video=video,
            frame=frame_index,
            video_stream_index=video_stream_index,
            timestamp=_frame_timestamp(frame, frame_index, fps),
        )
        video_frame._decoded = frame
        yield video_frame


def video_frame_np(
    video: VideoFile, frame: int, video_stream_index: int = 0
) -> ndarray:
    """
    Reads video frame from a file and returns as numpy array.

    Display rotation is applied, matching FFmpeg/OpenCV; channels are RGB.

    Args:
        video (VideoFile): Video file object.
        frame (int): Frame index.
        video_stream_index: Zero-based index among video streams.

    Returns:
        ndarray: Video frame.
    """
    if frame < 0:
        raise ValueError("frame must be a non-negative integer")

    decoded_frame, _ = _decode_video_frame(video, frame, video_stream_index)
    return _frame_to_ndarray(decoded_frame)


def validate_frame_range(
    video: VideoFile,
    start: int = 0,
    end: int | None = None,
    step: int = 1,
    video_stream_index: int = 0,
) -> tuple[int, int, int]:
    """
    Validates frame range for a video file.

    Args:
        video (VideoFile): Video file object.
        start (int): Start frame index (default: 0).
        end (int, optional): End frame index (default: None).
        step (int): Step between frames (default: 1).
        video_stream_index: Zero-based index among video streams.

    Returns:
        tuple[int, int, int]: Start frame index, end frame index, and step.
    """
    if start < 0:
        raise ValueError("start_frame must be a non-negative integer.")
    if step < 1:
        raise ValueError("step must be a positive integer.")
    _validate_video_stream_index(video_stream_index)

    if end is None:
        end = video_info(video, video_stream_index=video_stream_index).frames

    if end < 0:
        raise ValueError("end_frame must be a non-negative integer.")
    if start > end:
        raise ValueError("start_frame must be less than or equal to end_frame.")

    return start, end, step


def _frame_timestamp(frame, frame_index: int, fps: float) -> float:
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)

    if frame.time is not None:
        return float(frame.time)

    if fps > 0:
        return frame_index / fps

    raise ValueError("unable to determine frame timestamp")


def video_frames(
    video: VideoFile,
    start: int,
    end: int,
    step: int,
    video_stream_index: int = 0,
) -> Iterator[VideoFrame]:
    """Yield video frames with decoded timestamps when available."""
    _validate_video_stream_index(video_stream_index)

    try:
        with video.open() as f:
            with av.open(f) as container:
                input_container: Any = container
                video_stream = _video_stream(
                    input_container,
                    video_stream_index,
                    video.source,
                    video.path,
                    "unable to read video frames",
                )
                fps = _video_fps(video_stream)
                seeked = _seek_to_frame(input_container, f, video_stream, start, fps)

                try:
                    yield from _video_frames_from_decoded(
                        video=video,
                        decoded_frames=_decoded_frames(
                            input_container, video_stream, start, fps, seeked
                        ),
                        start=start,
                        end=end,
                        step=step,
                        fps=fps,
                        video_stream_index=video_stream_index,
                    )
                except _SeekLandedAfterStartError:
                    if not seeked:
                        raise
                    input_container.seek(
                        video_stream.start_time or 0,
                        backward=True,
                        any_frame=False,
                        stream=video_stream,
                    )
                    yield from _video_frames_from_decoded(
                        video=video,
                        decoded_frames=_decoded_frames(
                            input_container, video_stream, start, fps, False
                        ),
                        start=start,
                        end=end,
                        step=step,
                        fps=fps,
                        video_stream_index=video_stream_index,
                    )
    except FileError:
        raise
    except Exception as exc:
        raise FileError(
            "unable to read video frames", video.source, video.path
        ) from exc


def video_frame(
    video: VideoFile, frame: int, video_stream_index: int = 0
) -> VideoFrame:
    """Return one video frame reference with an FPS-derived timestamp."""
    _validate_video_stream_index(video_stream_index)
    info = video.get_info(video_stream_index=video_stream_index)
    timestamp = frame / info.fps if info.fps > 0 else -1.0
    return VideoFrame(
        video=video,
        frame=frame,
        video_stream_index=video_stream_index,
        timestamp=timestamp,
    )


def _decode_video_frame(video: VideoFile, frame: int, video_stream_index: int = 0):
    _validate_video_stream_index(video_stream_index)

    try:
        with video.open() as f:
            with av.open(f) as container:
                input_container: Any = container
                video_stream = _video_stream(
                    input_container,
                    video_stream_index,
                    video.source,
                    video.path,
                    "unable to read video frame",
                )
                fps = _video_fps(video_stream)
                seeked = _seek_to_frame(input_container, f, video_stream, frame, fps)

                decoded_frame = _find_decoded_frame(
                    input_container, video_stream, frame, fps, seeked
                )
                if decoded_frame is not None:
                    return decoded_frame, fps

                if seeked:
                    input_container.seek(
                        video_stream.start_time or 0,
                        backward=True,
                        any_frame=False,
                        stream=video_stream,
                    )
                    decoded_frame = _find_decoded_frame(
                        input_container, video_stream, frame, fps, False
                    )
                    if decoded_frame is not None:
                        return decoded_frame, fps
    except FileError:
        raise
    except Exception as exc:
        raise FileError("unable to read video frame", video.source, video.path) from exc

    raise FileError("unable to read video frame", video.source, video.path)


def _image_format(format: str) -> str:
    extension = format.lower()
    if not extension.startswith("."):
        extension = f".{extension}"

    image_format = PilImage.registered_extensions().get(extension)
    if image_format is not None:
        return image_format

    return format.upper()


def _ffmpeg_output_options(format: str) -> dict[str, str]:
    options = {"format": format}
    if format.lower() in {"m4v", "mov", "mp4"}:
        options["movflags"] = "frag_keyframe+empty_moov"
    return options


def _ffmpeg_fragment_timeout(
    start: float, end: float, timeout: float | None
) -> float | None:
    if timeout is not None:
        if timeout < 0:
            raise ValueError("timeout must be a non-negative float")
        if timeout == 0:
            return None
        return timeout

    return max(
        FFMPEG_FRAGMENT_TIMEOUT_MIN,
        (end - start) * FFMPEG_FRAGMENT_TIMEOUT_FACTOR,
    )


def _ffmpeg_fragment_cmd(video: VideoFile) -> tuple[str, ...]:
    ffmpeg_path = shutil.which(FFMPEG_FRAGMENT_EXECUTABLE)
    if ffmpeg_path is None:
        raise FileError("ffmpeg executable not found in PATH", video.source, video.path)
    return (ffmpeg_path, *FFMPEG_FRAGMENT_ARGS)


def _run_ffmpeg(stream_spec: Any, video: VideoFile, timeout: float | None) -> None:
    process = subprocess.Popen(  # noqa: S603
        stream_spec.compile(cmd=_ffmpeg_fragment_cmd(video)),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise TimeoutError(f"ffmpeg timed out after {timeout:.3f} seconds") from exc

    if process.poll():
        raise ffmpeg.Error("ffmpeg", b"", err or b"")


def _video_fragment_format(video: VideoFile, format: str | None) -> str:
    if format is None:
        format = video.get_file_ext()
    format = format.removeprefix(".").lower()
    if not format:
        raise ValueError(
            f"Can't save video fragment for '{video.path}', "
            "output format must be specified when source has no extension"
        )
    return format


@contextmanager
def _local_video_input(video: VideoFile) -> Iterator[str]:
    if cached_path := video.get_local_path():
        yield cached_path
        return

    if not video.location:
        if not video.source or video.source.startswith("file://"):
            yield video.get_fs_path()
            return

        if video._caching_enabled:
            video.ensure_cached()
            if cached_path := video.get_local_path():
                yield cached_path
                return
    elif video._caching_enabled:
        location = cast("list[dict]", video.location)
        VFileRegistry.parent(video, location).ensure_cached()

    with tempfile.TemporaryDirectory(prefix="datachain-video-") as temp_dir:
        temp_input_name = os.path.join(temp_dir, video.name or "video")
        video.save(temp_input_name)
        yield temp_input_name


def save_video_fragment(
    video: VideoFile,
    start: float,
    end: float,
    destination: str | os.PathLike[str],
    format: str | None = None,
    client_config: dict | None = None,
    timeout: float | None = None,
) -> VideoFile:
    """
    Saves video interval as a new video file. If ``destination`` is a remote
    path, the video will be uploaded to remote storage.

    Args:
        video: Video file object.
        start: Start time in seconds.
        end: End time in seconds.
        destination: Output directory path or URI (e.g. ``s3://…``, ``gs://…``).
        format: Output format. If None, inferred from the file extension.
        client_config: Optional client configuration (e.g. credentials).
        timeout: FFmpeg subprocess timeout in seconds. If None, a timeout is
            computed from the fragment duration. Set to 0 to disable.

    Returns:
        VideoFile: Video fragment model.
    """
    catalog = video._catalog
    if catalog is None:
        raise RuntimeError("Cannot save video fragment: catalog is not set")

    destination = stringify_path(destination)

    if start < 0 or end < 0 or start >= end:
        raise ValueError(
            f"Can't save video fragment for '{video.path}', "
            f"invalid time range: ({start:.3f}, {end:.3f})"
        )

    format = _video_fragment_format(video, format)
    timeout = _ffmpeg_fragment_timeout(start, end, timeout)

    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    output_file = posixpath.join(
        destination, f"{video.get_file_stem()}_{start_ms:06d}_{end_ms:06d}.{format}"
    )

    client, rel_path = video._resolve_destination(output_file, client_config)
    with _local_video_input(video) as input_file:
        with tempfile.TemporaryDirectory(
            prefix="datachain-video-fragment-"
        ) as temp_dir:
            temp_output = os.path.join(temp_dir, f"fragment.{format}")
            try:
                _run_ffmpeg(
                    ffmpeg.input(input_file, ss=start, to=end).output(
                        temp_output, **_ffmpeg_output_options(format)
                    ),
                    video,
                    timeout,
                )
            except TimeoutError as exc:
                raise FileError(
                    "ffmpeg timed out while saving video fragment",
                    video.source,
                    video.path,
                ) from exc

            with open(temp_output, "rb") as data:
                result = client.upload(data, rel_path)
    vf = VideoFile(**result.model_dump())
    vf._set_stream(catalog)
    return vf
