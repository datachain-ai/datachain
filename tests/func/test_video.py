import io
import os
import shutil
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import av
import ffmpeg
import numpy as np
import pytest
from numpy import ndarray
from PIL import Image

from datachain import VideoFragment, VideoFrame
from datachain.lib.file import File, FileError, ImageFile, VideoFile
from datachain.lib.tar import process_tar
from datachain.lib.video import save_video_fragment, video_frame_np

requires_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg"), reason="ffmpeg not installed"
)
requires_posix = pytest.mark.skipif(
    os.name == "nt", reason="fake ffmpeg executable uses a POSIX shell script"
)


@dataclass(frozen=True)
class GeneratedVideo:
    path: Path
    file: VideoFile


class NonSeekableBytesIO(io.BytesIO):
    def seekable(self):
        return False


class SeekTellReader:
    def __init__(self, data: bytes):
        self._buffer = io.BytesIO(data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self._buffer.read(size)

    def seek(self, offset, whence=0):
        return self._buffer.seek(offset, whence)

    def tell(self):
        return self._buffer.tell()


class SeekAheadContainer:
    def __init__(self, container, start_frame: int):
        self._container = container
        self._start_frame = start_frame
        self._seek_landed_after_start = False
        self.streams = container.streams

    def __enter__(self):
        self._container.__enter__()
        return self

    def __exit__(self, *args):
        return self._container.__exit__(*args)

    def seek(self, offset, *, backward=True, any_frame=False, stream=None):
        result = self._container.seek(
            offset,
            backward=backward,
            any_frame=any_frame,
            stream=stream,
        )
        self._seek_landed_after_start = bool(
            stream is not None and offset != (stream.start_time or 0)
        )
        return result

    def decode(self, stream):
        frames = self._container.decode(stream)
        if self._seek_landed_after_start:
            self._seek_landed_after_start = False
            yield self._frame_after_start(stream)
            return
        yield from frames

    def _frame_after_start(self, stream):
        fps = float(stream.average_rate or stream.base_rate or stream.guessed_rate)
        timestamp = (self._start_frame + 1) / fps
        pts = (stream.start_time or 0) + int(timestamp / stream.time_base)
        return SimpleNamespace(pts=pts, time_base=stream.time_base, time=timestamp)


def _install_fake_ffmpeg(tmp_path, monkeypatch, script: str) -> None:
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text(script)
    fake_ffmpeg.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")


def _fake_ffmpeg_write_output_script(record_args: bool = False) -> str:
    record = 'printf \'%s\n\' "$0" "$@" > "$FFMPEG_ARGS_FILE"\n' if record_args else ""
    return (
        "#!/bin/sh\n"
        f"{record}"
        'out=""\n'
        'for arg do out="$arg"; done\n'
        'printf fake-video > "$out"\n'
    )


def _add_video_stream(
    container,
    codec: str,
    fps: int,
    *,
    width: int = 16,
    height: int = 16,
    set_time_base: bool = True,
):
    stream = cast("av.VideoStream", container.add_stream(codec, rate=fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    if set_time_base:
        stream.time_base = Fraction(1, fps)
    return stream


def _mux_video_frame(container, stream, pixel_value: int, pts: int | None, fps: int):
    image = np.full((stream.height, stream.width, 3), pixel_value, dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
    if pts is not None:
        frame.pts = pts
        frame.time_base = Fraction(1, fps)
    for packet in stream.encode(frame):
        container.mux(packet)


def _flush_video_stream(container, stream) -> None:
    for packet in stream.encode():
        container.mux(packet)


def _default_pixel_value(frame_index: int) -> int:
    return frame_index * 40


def _write_single_stream_video(
    path: Path,
    *,
    codec: str,
    fps: int,
    frame_count: int,
    container_format: str | None = None,
    set_stream_time_base: bool = True,
    set_frame_timestamps: bool = True,
    pts: list[int] | None = None,
    pixel_value: Callable[[int], int] | None = None,
) -> None:
    if container_format:
        container = av.open(str(path), "w", format=container_format)
    else:
        container = av.open(str(path), "w")
    stream = _add_video_stream(
        container,
        codec,
        fps,
        set_time_base=set_stream_time_base,
    )
    if pixel_value is None:
        pixel_value = _default_pixel_value

    try:
        for frame_index in range(frame_count):
            if pts is not None:
                frame_pts = pts[frame_index]
            elif set_frame_timestamps:
                frame_pts = frame_index
            else:
                frame_pts = None
            _mux_video_frame(
                container,
                stream,
                pixel_value(frame_index),
                frame_pts,
                fps,
            )

        _flush_video_stream(container, stream)
    finally:
        container.close()


def _generated_video(path: Path) -> GeneratedVideo:
    return GeneratedVideo(
        path=path,
        file=VideoFile.upload(path.read_bytes(), path.name),
    )


@pytest.fixture
def variable_timestamp_video(tmp_path) -> GeneratedVideo:
    path = tmp_path / "variable_timestamp.mp4"
    frame_pts = [0, 1, 5, 6]
    _write_single_stream_video(
        path,
        codec="mpeg4",
        fps=30,
        frame_count=len(frame_pts),
        pts=frame_pts,
    )
    return _generated_video(path)


@pytest.fixture
def raw_h264_video(tmp_path) -> GeneratedVideo:
    path = tmp_path / "raw.h264"
    _write_single_stream_video(
        path,
        codec="h264",
        fps=30,
        frame_count=4,
        container_format="h264",
        set_stream_time_base=False,
        set_frame_timestamps=False,
        pts=None,
    )
    return _generated_video(path)


@pytest.fixture
def mkv_video_without_frame_count(tmp_path) -> GeneratedVideo:
    path = tmp_path / "inferred_frames.mkv"
    _write_single_stream_video(
        path,
        codec="mpeg4",
        fps=3,
        frame_count=7,
        pixel_value=lambda frame_index: frame_index * 20,
    )
    return _generated_video(path)


@pytest.fixture
def mpegts_video(tmp_path) -> GeneratedVideo:
    path = tmp_path / "start_time.ts"
    _write_single_stream_video(
        path,
        codec="mpeg2video",
        fps=24,
        frame_count=10,
        container_format="mpegts",
        pixel_value=lambda frame_index: frame_index * 20,
    )
    return _generated_video(path)


@pytest.fixture
def nut_video_without_average_rate(tmp_path) -> GeneratedVideo:
    path = tmp_path / "no_average_rate.nut"
    _write_single_stream_video(
        path,
        codec="mpeg4",
        fps=5,
        frame_count=6,
        container_format="nut",
        pixel_value=lambda frame_index: frame_index * 20,
    )
    return _generated_video(path)


@pytest.fixture
def multi_stream_video(tmp_path) -> GeneratedVideo:
    path = tmp_path / "multi_stream.mp4"
    container = av.open(str(path), "w")
    fps = 30
    streams = [
        _add_video_stream(container, "mpeg4", fps, width=16, height=16),
        _add_video_stream(container, "mpeg4", fps, width=32, height=24),
    ]

    try:
        for frame_index in range(2):
            for video_stream_index, stream in enumerate(streams):
                _mux_video_frame(
                    container,
                    stream,
                    video_stream_index * 80 + frame_index * 20,
                    frame_index,
                    fps,
                )

        for stream in streams:
            _flush_video_stream(container, stream)
    finally:
        container.close()

    return _generated_video(path)


@pytest.fixture
def audio_first_video(tmp_path) -> GeneratedVideo:
    path = tmp_path / "audio_first.mp4"
    container = av.open(str(path), "w")
    sample_rate = 44100
    audio_stream = container.add_stream("aac", rate=sample_rate)
    audio_stream.layout = "mono"

    fps = 2
    video_stream = _add_video_stream(container, "mpeg4", fps)

    try:
        audio_frame_samples = 1024
        for sample_offset in range(0, sample_rate, audio_frame_samples):
            samples = min(audio_frame_samples, sample_rate - sample_offset)
            timeline = np.arange(sample_offset, sample_offset + samples)
            tone = (
                np.sin(2 * np.pi * 1000 * timeline / sample_rate)
                * 0.2
                * np.iinfo(np.int16).max
            ).astype(np.int16)
            audio_frame = av.AudioFrame(format="s16", layout="mono", samples=samples)
            audio_frame.sample_rate = sample_rate
            audio_frame.pts = sample_offset
            audio_frame.time_base = Fraction(1, sample_rate)
            audio_frame.planes[0].update(tone.tobytes())
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

        for packet in audio_stream.encode():
            container.mux(packet)

        for frame_index in range(2):
            _mux_video_frame(
                container,
                video_stream,
                frame_index * 40,
                frame_index,
                fps,
            )

        _flush_video_stream(container, video_stream)
    finally:
        container.close()

    return _generated_video(path)


@pytest.fixture(autouse=True)
def video_file(catalog) -> File:
    data_path = os.path.join(os.path.dirname(__file__), "data")
    file_name = "Big_Buck_Bunny_360_10s_1MB.mp4"

    with open(os.path.join(data_path, file_name), "rb") as f:
        return File.upload(f.read(), file_name)


@pytest.fixture
def make_tar_member_file(tmp_path, test_session):
    def make_tar_member_file(
        member_name: str,
        contents: bytes | str | os.PathLike[str],
        *,
        caching_enabled: bool = False,
    ) -> tuple[File, File]:
        archive_path = tmp_path / "archive.tar"

        with tarfile.open(archive_path, mode="w") as archive:
            if isinstance(contents, bytes):
                info = tarfile.TarInfo(member_name)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))
            else:
                archive.add(str(contents), arcname=member_name)

        archive_file = File.at(archive_path, session=test_session)
        member_file = next(process_tar(archive_file))
        member_file._set_stream(test_session.catalog, caching_enabled=caching_enabled)
        return archive_file, member_file

    return make_tar_member_file


@requires_ffmpeg
def test_get_info(video_file):
    info = video_file.as_video_file().get_info()
    assert info.model_dump() == {
        "width": 640,
        "height": 360,
        "fps": 30.0,
        "duration": 10.0,
        "frames": 300,
        "format": "mov,mp4,m4a,3gp,3g2,mj2",
        "codec": "h264",
    }


def test_get_info_error():
    # upload current Python file as video file to get an error while getting video meta
    with open(__file__, "rb") as f:
        file = VideoFile.upload(f.read(), "test.mp4")

    with pytest.raises(FileError):
        file.get_info()


def test_get_info_handles_raw_video_without_duration(raw_h264_video):
    info = raw_h264_video.file.get_info()

    assert info.fps > 0
    assert info.duration == -1.0
    assert info.frames == 0


def test_get_frame_uses_raw_video_fps_for_timestamp(raw_h264_video):
    info = raw_h264_video.file.get_info()
    frame = raw_h264_video.file.get_frame(3)

    assert frame.timestamp == pytest.approx(3 / info.fps)


def test_get_info_ceil_inferred_frame_count(mkv_video_without_frame_count):
    info = mkv_video_without_frame_count.file.get_info()

    assert info.frames == 7


def test_get_info_uses_base_rate_when_average_rate_is_missing(
    nut_video_without_average_rate,
):
    with av.open(str(nut_video_without_average_rate.path)) as container:
        stream = container.streams.video[0]
        assert stream.average_rate is None
        assert stream.base_rate == 5

    info = nut_video_without_average_rate.file.get_info()

    assert info.fps == 5.0


def test_get_frame(video_file):
    frame = video_file.as_video_file().get_frame(37)
    assert isinstance(frame, VideoFrame)
    assert frame.frame == 37
    assert frame.video_stream_index == 0
    assert frame.timestamp == pytest.approx(37 / 30)


def test_get_frames_uses_frame_index_when_timestamps_are_missing(raw_h264_video):
    file = raw_h264_video.file
    info = file.get_info()

    frames = list(file.get_frames(0, 4))

    assert [frame.timestamp for frame in frames] == pytest.approx(
        [frame.frame / info.fps for frame in frames]
    )


def test_get_frame_np_handles_raw_video_without_timestamps(raw_h264_video):
    frame = raw_h264_video.file.get_frame(3).get_np()

    assert frame.shape == (16, 16, 3)


def test_get_frame_error(video_file):
    with pytest.raises(ValueError):
        video_file.as_video_file().get_frame(-1)


def test_get_frame_returns_reference_without_decoding(video_file):
    video = video_file.as_video_file()

    frame = video.get_frame(10_000)

    assert frame.frame == 10_000
    assert frame.timestamp == pytest.approx(10_000 / 30)


def test_get_frame_reuses_metadata(video_file, monkeypatch):
    video = video_file.as_video_file()
    frame = video.get_frame(37)
    assert frame.timestamp == pytest.approx(37 / 30)

    def fail_open(self, *args, **kwargs):
        raise AssertionError("get_frame should reuse cached video metadata")

    monkeypatch.setattr(VideoFile, "open", fail_open)

    frame = video.get_frame(38)

    assert frame.timestamp == pytest.approx(38 / 30)


def test_get_frame_np(video_file):
    frame = video_file.as_video_file().get_frame(0).get_np()
    assert isinstance(frame, ndarray)
    assert frame.shape == (360, 640, 3)


def test_get_frame_np_matches_sequential_decode_after_seek(video_file):
    frame_index = 250
    image = video_file.as_video_file().get_frame(frame_index).get_np()

    data_path = os.path.join(os.path.dirname(__file__), "data")
    video_path = os.path.join(data_path, "Big_Buck_Bunny_360_10s_1MB.mp4")
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        expected = next(
            frame.to_ndarray(format="rgb24")
            for index, frame in enumerate(container.decode(stream))
            if index == frame_index
        )

    np.testing.assert_array_equal(image, expected)


def test_get_frame_np_decodes_non_seekable_stream(video_file, monkeypatch):
    data = video_file.read()

    def open_non_seekable(self, *args, **kwargs):
        return NonSeekableBytesIO(data)

    monkeypatch.setattr(VideoFile, "open", open_non_seekable)

    image = video_file.as_video_file().get_frame(3).get_np()

    assert image.shape == (360, 640, 3)


def test_get_frame_np_decodes_seek_tell_stream(video_file, monkeypatch):
    data = video_file.read()

    def open_seek_tell(self, *args, **kwargs):
        return SeekTellReader(data)

    monkeypatch.setattr(VideoFile, "open", open_seek_tell)

    image = video_file.as_video_file().get_frame(3).get_np()

    assert image.shape == (360, 640, 3)


def test_get_frame_np_handles_stream_start_time_without_seek(mpegts_video):
    video_path = mpegts_video.path
    file = mpegts_video.file
    frame_index = 5

    image = file.get_frame(frame_index).get_np()

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        assert stream.start_time not in (None, 0)
        expected = next(
            frame.to_ndarray(format="rgb24")
            for index, frame in enumerate(container.decode(stream))
            if index == frame_index
        )

    np.testing.assert_array_equal(image, expected)


def test_get_frame_np_error(video_file):
    with pytest.raises(ValueError):
        video_frame_np(video_file.as_video_file(), -1)


def test_get_frame_np_missing_frame_error(video_file):
    with pytest.raises(FileError, match="unable to read video frame"):
        video_frame_np(video_file.as_video_file(), 10_000)


def test_get_frame_np_missing_raw_frame_error(raw_h264_video):
    with pytest.raises(FileError, match="unable to read video frame"):
        video_frame_np(raw_h264_video.file, 10_000)


def test_get_frame_np_video_stream_index_error(video_file):
    with pytest.raises(FileError, match="video_stream_index 1 is out of range"):
        video_frame_np(video_file.as_video_file(), 0, video_stream_index=1)


def test_get_frame_np_wraps_decode_errors():
    file = VideoFile.upload(b"not a video", "broken.mp4")

    with pytest.raises(FileError, match="unable to read video frame"):
        video_frame_np(file, 0)


@pytest.mark.parametrize(
    "format,img_format,header",
    [
        ("jpg", "JPEG", [b"\xff\xd8\xff\xe0"]),
        (".jpg", "JPEG", [b"\xff\xd8\xff\xe0"]),
        ("png", "PNG", [b"\x89PNG\r\n\x1a\n"]),
        ("gif", "GIF", [b"GIF87a", b"GIF89a"]),
        ("tif", "TIFF", [b"II*\x00", b"MM\x00*"]),
        ("JPEG2000", "JPEG2000", [b"\x00\x00\x00\x0cjP"]),
    ],
)
def test_get_frame_bytes(video_file, format, img_format, header):
    frame = video_file.as_video_file().get_frame(0).read_bytes(format)
    assert isinstance(frame, bytes)
    assert any(frame.startswith(h) for h in header)

    with Image.open(io.BytesIO(frame)) as img:
        assert img.format == img_format
        assert img.size == (640, 360)


@pytest.mark.parametrize("format", [None, "jpg", ".jpg"])
def test_save_frame(tmp_path, video_file, format):
    frame = video_file.as_video_file().get_frame(3)
    if format is not None:
        frame_file = frame.save(str(tmp_path), format=format)
    else:
        frame_file = frame.save(str(tmp_path))
    assert isinstance(frame_file, ImageFile)
    assert frame_file.path.endswith("_0003.jpg")
    assert "..jpg" not in frame_file.path

    frame_file.ensure_cached()
    frame_path = frame_file.get_local_path()
    assert frame_path is not None
    with Image.open(frame_path) as img:
        assert img.format == "JPEG"
        assert img.size == (640, 360)


def test_video_frame_save_requires_catalog(tmp_path):
    video = VideoFile(source="file:///tmp", path="video.mp4")
    frame = VideoFrame(video=video, frame=0, timestamp=0)

    with pytest.raises(RuntimeError, match="catalog is not set"):
        frame.save(str(tmp_path))


def test_get_frames(video_file):
    frames = list(video_file.as_video_file().get_frames(10, 200, 5))
    assert len(frames) == 38
    assert all(isinstance(frame, VideoFrame) for frame in frames)
    assert [frame.timestamp for frame in frames[:3]] == pytest.approx(
        [10 / 30, 15 / 30, 20 / 30]
    )


def test_get_frames_uses_seek_for_large_start(video_file):
    start = 250
    frames = list(video_file.as_video_file().get_frames(start, 256, 2))

    assert [frame.frame for frame in frames] == [250, 252, 254]
    assert [frame.timestamp for frame in frames] == pytest.approx(
        [250 / 30, 252 / 30, 254 / 30]
    )


def test_get_frames_retries_when_seek_lands_after_start(video_file, monkeypatch):
    start = 250
    real_av_open = av.open

    def open_with_seek_ahead(*args, **kwargs):
        return SeekAheadContainer(real_av_open(*args, **kwargs), start)

    monkeypatch.setattr(av, "open", open_with_seek_ahead)

    frames = list(video_file.as_video_file().get_frames(start, start + 3))

    assert [frame.frame for frame in frames] == [250, 251, 252]
    assert [frame.timestamp for frame in frames] == pytest.approx(
        [250 / 30, 251 / 30, 252 / 30]
    )


def test_get_frames_handles_stream_start_time_without_seek(mpegts_video):
    start = 5
    frames = list(mpegts_video.file.get_frames(start, 8))

    assert [frame.frame for frame in frames] == [5, 6, 7]

    with av.open(str(mpegts_video.path)) as container:
        stream = container.streams.video[0]
        assert stream.start_time not in (None, 0)
        expected = [
            frame.time
            for index, frame in enumerate(container.decode(stream))
            if start <= index < 8
        ]

    assert [frame.timestamp for frame in frames] == pytest.approx(expected)


def test_get_frames_uses_presentation_timestamps(variable_timestamp_video):
    file = variable_timestamp_video.file

    frames = list(file.get_frames(0, 4))
    assert [frame.frame for frame in frames] == [0, 1, 2, 3]
    assert [frame.timestamp for frame in frames] == pytest.approx(
        [0, 1 / 30, 5 / 30, 6 / 30]
    )


def test_video_stream_index_selects_video_stream(multi_stream_video):
    file = multi_stream_video.file

    info = file.get_info(video_stream_index=1)
    assert info.width == 32
    assert info.height == 24
    assert info.frames == 2

    frame = file.get_frame(0, video_stream_index=1)
    assert frame.frame == 0
    assert frame.video_stream_index == 1
    assert frame.get_np().shape == (24, 32, 3)

    frames = list(file.get_frames(0, 2, video_stream_index=1))
    assert [frame.video_stream_index for frame in frames] == [1, 1]
    assert [frame.timestamp for frame in frames] == pytest.approx([0, 1 / 30])


def test_video_stream_index_is_relative_to_video_streams(audio_first_video):
    video_path = audio_first_video.path

    with av.open(str(video_path)) as container:
        assert next(iter(container.streams)).type == "audio"
        assert container.streams.video[0].index == 1

    file = audio_first_video.file
    info = file.get_info(video_stream_index=0)
    assert info.width == 16
    assert info.height == 16


def test_video_stream_index_error(video_file):
    with pytest.raises(ValueError):
        video_file.as_video_file().get_frame(0, video_stream_index=-1)

    with pytest.raises(FileError):
        video_file.as_video_file().get_info(video_stream_index=1)

    with pytest.raises(FileError):
        list(video_file.as_video_file().get_frames(0, 1, video_stream_index=1))


def test_get_frames_wraps_decode_errors():
    file = VideoFile.upload(b"not a video", "broken.mp4")

    with pytest.raises(FileError, match="unable to read video frames"):
        list(file.get_frames(0, 1))


@requires_ffmpeg
def test_get_all_frames(video_file):
    frames = list(video_file.as_video_file().get_frames())
    assert len(frames) == 300
    assert all(isinstance(frame, VideoFrame) for frame in frames)


@pytest.mark.parametrize(
    "start,end,step",
    [
        (-1, None, 1),
        (0, -1, 1),
        (1, 0, 1),
        (0, 1, -1),
    ],
)
def test_get_frames_error(video_file, start, end, step):
    with pytest.raises(ValueError):
        list(video_file.as_video_file().get_frames(start, end, step))


def test_save_frames(tmp_path, video_file):
    frames = list(video_file.as_video_file().get_frames(10, 200, 5))
    frame_files = [frame.save(str(tmp_path), format="jpg") for frame in frames]
    assert len(frame_files) == 38

    for frame_file in frame_files:
        frame_file.ensure_cached()
        frame_path = frame_file.get_local_path()
        assert frame_path is not None
        with Image.open(frame_path) as img:
            assert img.format == "JPEG"
            assert img.size == (640, 360)


def test_get_fragment(video_file):
    fragment = video_file.as_video_file().get_fragment(2.5, 5)
    assert isinstance(fragment, VideoFragment)
    assert fragment.start == 2.5
    assert fragment.end == 5


@requires_ffmpeg
def test_get_fragments(video_file):
    fragments = list(video_file.as_video_file().get_fragments(duration=1.5))
    for i, fragment in enumerate(fragments):
        assert isinstance(fragment, VideoFragment)
        assert fragment.start == i * 1.5
        duration = 1.5 if i < 6 else 1.0
        assert fragment.end == fragment.start + duration


@pytest.mark.parametrize(
    "duration,start,end",
    [
        (-1, 0, 10),
        (1, -1, 10),
        (1, 0, -1),
        (1, 2, 1),
    ],
)
def test_get_fragments_error(video_file, duration, start, end):
    with pytest.raises(ValueError):
        list(
            video_file.as_video_file().get_fragments(
                duration=duration, start=start, end=end
            )
        )


@pytest.mark.parametrize(
    "start,end",
    [
        (-1, -1),
        (-1, 2.5),
        (5, -1),
        (5, 2.5),
        (5, 5),
    ],
)
def test_save_fragment_error(video_file, start, end):
    with pytest.raises(ValueError):
        video_file.as_video_file().get_fragment(start, end)


@requires_ffmpeg
def test_save_fragment(tmp_path, video_file):
    fragment = video_file.as_video_file().get_fragment(2.5, 5).save(str(tmp_path))

    fragment.ensure_cached()
    assert fragment.get_info().model_dump() == {
        "width": 640,
        "height": 360,
        "fps": 30.0,
        "duration": 2.5,
        "frames": 75,
        "format": "mov,mp4,m4a,3gp,3g2,mj2",
        "codec": "h264",
    }


@requires_ffmpeg
def test_save_video_fragment_uses_cached_input(tmp_path, video_file):
    video = video_file.as_video_file()
    video.ensure_cached()
    cached_path = video.get_local_path()
    source_path = video.get_fs_path()
    assert cached_path

    if source_path != cached_path:
        os.remove(source_path)

    fragment = save_video_fragment(video, 2.5, 5, str(tmp_path))

    fragment.ensure_cached()
    assert fragment.get_info().duration == 2.5


@requires_ffmpeg
def test_save_video_fragment_uses_cache_after_ensure_cached(
    tmp_path, monkeypatch, video_file
):
    video = video_file.as_video_file()
    real_cached_path = video.get_fs_path()
    video.source = "gs://bucket"
    video._caching_enabled = True
    calls = []

    def fake_get_local_path(self):
        calls.append("get_local_path")
        return real_cached_path if "ensure_cached" in calls else None

    def fake_ensure_cached(self):
        calls.append("ensure_cached")

    monkeypatch.setattr(VideoFile, "get_local_path", fake_get_local_path)
    monkeypatch.setattr(VideoFile, "ensure_cached", fake_ensure_cached)

    fragment = save_video_fragment(video, 0, 1, str(tmp_path / "out"))

    assert calls == ["get_local_path", "ensure_cached", "get_local_path"]
    fragment.ensure_cached()
    assert fragment.get_info().duration == 1


@requires_ffmpeg
@pytest.mark.parametrize("caching_enabled", [False, True])
def test_save_video_fragment_remote_input_uses_temp_file_when_cache_is_unavailable(
    tmp_path, monkeypatch, video_file, caching_enabled
):
    video = video_file.as_video_file()
    source_path = video.get_fs_path()
    video.source = "gs://bucket"
    video._caching_enabled = caching_enabled
    calls = []

    def fake_get_local_path(self):
        calls.append("get_local_path")

    def fake_ensure_cached(self):
        calls.append("ensure_cached")

    def fake_save(self, destination, client_config=None):
        calls.append("save")
        shutil.copyfile(source_path, destination)

    monkeypatch.setattr(VideoFile, "get_local_path", fake_get_local_path)
    monkeypatch.setattr(VideoFile, "ensure_cached", fake_ensure_cached)
    monkeypatch.setattr(VideoFile, "save", fake_save)

    fragment = save_video_fragment(video, 0, 1, str(tmp_path / "out"))

    if caching_enabled:
        assert calls == ["get_local_path", "ensure_cached", "get_local_path", "save"]
    else:
        assert calls == ["get_local_path", "save"]
    fragment.ensure_cached()
    assert fragment.get_info().duration == 1


@requires_ffmpeg
def test_save_video_fragment_temp_input_uses_original_name_on_error(
    tmp_path, make_tar_member_file
):
    _, video = make_tar_member_file("original video.mp4", b"not a video")
    video = video.as_video_file()

    with pytest.raises(ffmpeg.Error) as exc_info:
        save_video_fragment(video, 1, 2, str(tmp_path))

    stderr = exc_info.value.stderr.decode("utf-8", errors="ignore")
    assert "original video.mp4" in stderr


@requires_ffmpeg
def test_save_video_fragment_caches_virtual_parent(tmp_path, make_tar_member_file):
    data_path = os.path.join(os.path.dirname(__file__), "data")
    video_name = "Big_Buck_Bunny_360_10s_1MB.mp4"
    archive, video = make_tar_member_file(
        "original.mp4",
        os.path.join(data_path, video_name),
        caching_enabled=True,
    )
    assert archive.get_local_path() is None
    video = video.as_video_file()

    fragment = save_video_fragment(video, 2.5, 5, str(tmp_path / "fragments"))

    assert archive.get_local_path() is not None
    fragment.ensure_cached()
    assert fragment.get_info().duration == 2.5


@pytest.mark.parametrize(
    "start,end",
    [
        (-1, 2),
        (1, -1),
        (2, 1),
    ],
)
def test_save_video_fragment_error(video_file, start, end):
    with pytest.raises(ValueError):
        save_video_fragment(video_file.as_video_file(), start, end, ".")


def test_save_video_fragment_requires_format_without_source_extension(tmp_path):
    video = VideoFile.upload(b"not a video", "video")

    with pytest.raises(ValueError, match="output format must be specified"):
        save_video_fragment(video, 0, 1, str(tmp_path))


def test_save_video_fragment_requires_catalog(tmp_path):
    video = VideoFile(source="file:///tmp", path="video.mp4")

    with pytest.raises(RuntimeError, match="catalog is not set"):
        save_video_fragment(video, 0, 1, str(tmp_path))


def test_save_video_fragment_requires_ffmpeg_executable(
    tmp_path, monkeypatch, video_file
):
    monkeypatch.setenv("PATH", "")

    with pytest.raises(FileError, match="ffmpeg executable not found"):
        save_video_fragment(video_file.as_video_file(), 0, 1, str(tmp_path / "out"))


def test_save_video_fragment_rejects_negative_timeout(tmp_path, video_file):
    with pytest.raises(ValueError, match="non-negative"):
        save_video_fragment(video_file.as_video_file(), 0, 1, str(tmp_path), timeout=-1)


@requires_ffmpeg
def test_save_video_fragment_accepts_zero_timeout(tmp_path, video_file):
    fragment = save_video_fragment(
        video_file.as_video_file(), 0, 1, str(tmp_path / "out"), timeout=0
    )

    fragment.ensure_cached()
    assert fragment.get_info().duration == 1


@requires_ffmpeg
@pytest.mark.parametrize("format", ["avi", ".avi", ".AVI"])
def test_save_video_fragment_uses_explicit_format(tmp_path, video_file, format):
    fragment = save_video_fragment(
        video_file.as_video_file(), 0, 1, str(tmp_path / "out"), format=format
    )

    assert fragment.path.endswith(".avi")
    assert "..avi" not in fragment.path
    fragment.ensure_cached()
    assert fragment.get_info().format == "avi"


@requires_posix
def test_save_video_fragment_invokes_ffmpeg_non_interactively(
    tmp_path, monkeypatch, video_file
):
    args_file = tmp_path / "ffmpeg.args"
    monkeypatch.setenv("FFMPEG_ARGS_FILE", str(args_file))
    _install_fake_ffmpeg(
        tmp_path,
        monkeypatch,
        _fake_ffmpeg_write_output_script(record_args=True),
    )

    save_video_fragment(video_file.as_video_file(), 0, 1, str(tmp_path / "out"))

    args = args_file.read_text().splitlines()
    assert os.path.isabs(args[0])
    assert os.path.basename(args[0]) == "ffmpeg"
    assert args[1:5] == ["-nostdin", "-hide_banner", "-loglevel", "error"]
    assert "pipe:1" not in args
    assert args[-1].endswith(".mp4")


@requires_posix
def test_save_video_fragment_drops_stdout_on_ffmpeg_error(
    tmp_path, monkeypatch, video_file
):
    _install_fake_ffmpeg(
        tmp_path,
        monkeypatch,
        "#!/bin/sh\nprintf stdout-noise\nprintf stderr-detail >&2\nexit 1\n",
    )

    with pytest.raises(ffmpeg.Error) as exc_info:
        save_video_fragment(video_file.as_video_file(), 0, 1, str(tmp_path / "out"))

    assert exc_info.value.stdout == b""
    assert exc_info.value.stderr == b"stderr-detail"


@requires_posix
def test_save_video_fragment_times_out_ffmpeg(tmp_path, monkeypatch, video_file):
    _install_fake_ffmpeg(
        tmp_path,
        monkeypatch,
        "#!/bin/sh\nexec sleep 10\n",
    )

    with pytest.raises(FileError, match="ffmpeg timed out"):
        save_video_fragment(
            video_file.as_video_file(), 0, 1, str(tmp_path / "out"), timeout=0.01
        )


@requires_ffmpeg
def test_save_fragments(tmp_path, video_file):
    fragments = list(video_file.as_video_file().get_fragments(duration=1))
    fragment_files = [fragment.save(str(tmp_path)) for fragment in fragments]
    assert len(fragment_files) == 10

    for fragment in fragment_files:
        fragment.ensure_cached()
        assert fragment.get_info().model_dump() == {
            "width": 640,
            "height": 360,
            "fps": 30.0,
            "duration": 1,
            "frames": 30,
            "format": "mov,mp4,m4a,3gp,3g2,mj2",
            "codec": "h264",
        }
