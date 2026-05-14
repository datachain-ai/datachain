import io
import os
import shutil
import tarfile
from fractions import Fraction

import av
import ffmpeg
import numpy as np
import pytest
from numpy import ndarray
from PIL import Image

from datachain import VideoFragment, VideoFrame
from datachain.lib.file import File, FileError, ImageFile, VideoFile
from datachain.lib.tar import process_tar
from datachain.lib.video import save_video_fragment, video_frame, video_frame_np

requires_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg"), reason="ffmpeg not installed"
)
requires_posix = pytest.mark.skipif(
    os.name == "nt", reason="fake ffmpeg executable uses a POSIX shell script"
)


def _install_fake_ffmpeg(tmp_path, monkeypatch, script: str) -> None:
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text(script)
    fake_ffmpeg.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")


def _fake_ffmpeg_write_output_script(record_args: bool = False) -> str:
    record = 'printf \'%s\n\' "$@" > "$FFMPEG_ARGS_FILE"\n' if record_args else ""
    return (
        "#!/bin/sh\n"
        f"{record}"
        'out=""\n'
        'for arg do out="$arg"; done\n'
        'printf fake-video > "$out"\n'
    )


def _write_variable_timestamp_video(path):
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=30)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    time_base = Fraction(1, 30)
    stream.time_base = time_base

    try:
        for index, pts in enumerate([0, 1, 5, 6]):
            image = np.full((16, 16, 3), index * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = pts
            frame.time_base = time_base
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _write_raw_h264_video(path):
    container = av.open(str(path), "w", format="h264")
    stream = container.add_stream("h264", rate=30)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"

    try:
        for index in range(4):
            image = np.full((16, 16, 3), index * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _write_multi_stream_video(path):
    container = av.open(str(path), "w")
    time_base = Fraction(1, 30)
    streams = []

    for width, height in [(16, 16), (32, 24)]:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = time_base
        streams.append(stream)

    try:
        for frame_index in range(2):
            for video_stream_index, stream in enumerate(streams):
                image = np.full(
                    (stream.height, stream.width, 3),
                    video_stream_index * 80 + frame_index * 20,
                    dtype=np.uint8,
                )
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = frame_index
                frame.time_base = time_base
                for packet in stream.encode(frame):
                    container.mux(packet)

        for stream in streams:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        container.close()


def _write_audio_first_video(path):
    container = av.open(str(path), "w")
    sample_rate = 44100
    audio_stream = container.add_stream("aac", rate=sample_rate)
    audio_stream.layout = "mono"

    video_stream = container.add_stream("mpeg4", rate=2)
    video_stream.width = 16
    video_stream.height = 16
    video_stream.pix_fmt = "yuv420p"
    video_stream.time_base = Fraction(1, 2)

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
            image = np.full((16, 16, 3), frame_index * 40, dtype=np.uint8)
            video_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            video_frame.pts = frame_index
            video_frame.time_base = Fraction(1, 2)
            for packet in video_stream.encode(video_frame):
                container.mux(packet)

        for packet in video_stream.encode():
            container.mux(packet)
    finally:
        container.close()


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


def test_get_info_handles_raw_video_without_duration(tmp_path):
    video_path = tmp_path / "raw.h264"
    _write_raw_h264_video(video_path)
    file = VideoFile.upload(video_path.read_bytes(), video_path.name)

    info = file.get_info()

    assert info.fps > 0
    assert info.duration == -1.0
    assert info.frames == 0


def test_get_frame(video_file):
    frame = video_file.as_video_file().get_frame(37)
    assert isinstance(frame, VideoFrame)
    assert frame.frame == 37
    assert frame.video_stream_index == 0
    assert frame.timestamp == pytest.approx(37 / 30)


def test_get_frame_uses_frame_index_when_timestamps_are_missing(tmp_path):
    video_path = tmp_path / "raw.h264"
    _write_raw_h264_video(video_path)
    file = VideoFile.upload(video_path.read_bytes(), video_path.name)
    info = file.get_info()

    frame = file.get_frame(3)

    assert frame.timestamp == pytest.approx(frame.frame / info.fps)


def test_get_frame_error(video_file):
    with pytest.raises(ValueError):
        video_file.as_video_file().get_frame(-1)


def test_video_frame_function_rejects_negative_frame(video_file):
    with pytest.raises(ValueError):
        video_frame(video_file.as_video_file(), -1)


def test_get_frame_missing_frame_error(video_file):
    with pytest.raises(FileError, match="unable to read video frame"):
        video_file.as_video_file().get_frame(10_000)


def test_get_frame_np(video_file):
    frame = video_file.as_video_file().get_frame(0).get_np()
    assert isinstance(frame, ndarray)
    assert frame.shape == (360, 640, 3)


def test_get_frame_get_np_reuses_decoded_frame(monkeypatch, video_file):
    video = video_file.as_video_file()
    calls = 0
    original_open = VideoFile.open

    def counted_open(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(VideoFile, "open", counted_open)

    frame = video.get_frame(0)
    image = frame.get_np()

    assert image.shape == (360, 640, 3)
    assert frame.get_np() is image
    assert calls == 1


def test_get_frame_np_error(video_file):
    with pytest.raises(ValueError):
        video_frame_np(video_file.as_video_file(), -1)


def test_get_frame_np_missing_frame_error(video_file):
    with pytest.raises(FileError, match="unable to read video frame"):
        video_frame_np(video_file.as_video_file(), 10_000)


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
        ("png", "PNG", [b"\x89PNG\r\n\x1a\n"]),
        ("gif", "GIF", [b"GIF87a", b"GIF89a"]),
    ],
)
def test_get_frame_bytes(video_file, format, img_format, header):
    frame = video_file.as_video_file().get_frame(0).read_bytes(format)
    assert isinstance(frame, bytes)
    assert any(frame.startswith(h) for h in header)

    with Image.open(io.BytesIO(frame)) as img:
        assert img.format == img_format
        assert img.size == (640, 360)


@pytest.mark.parametrize("use_format", [True, False])
def test_save_frame(tmp_path, video_file, use_format):
    frame = video_file.as_video_file().get_frame(3)
    if use_format:
        frame_file = frame.save(str(tmp_path), format="jpg")
    else:
        frame_file = frame.save(str(tmp_path))
    assert isinstance(frame_file, ImageFile)

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


def test_get_frames_uses_presentation_timestamps(tmp_path):
    video_path = tmp_path / "variable_timestamp.mp4"
    _write_variable_timestamp_video(video_path)

    file = VideoFile.upload(video_path.read_bytes(), video_path.name)

    frames = list(file.get_frames(0, 4))
    assert [frame.frame for frame in frames] == [0, 1, 2, 3]
    assert [frame.timestamp for frame in frames] == pytest.approx(
        [0, 1 / 30, 5 / 30, 6 / 30]
    )


def test_video_stream_index_selects_video_stream(tmp_path):
    video_path = tmp_path / "multi_stream.mp4"
    _write_multi_stream_video(video_path)

    file = VideoFile.upload(video_path.read_bytes(), video_path.name)

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


def test_video_stream_index_is_relative_to_video_streams(tmp_path):
    video_path = tmp_path / "audio_first.mp4"
    _write_audio_first_video(video_path)

    with av.open(str(video_path)) as container:
        assert next(iter(container.streams)).type == "audio"
        assert container.streams.video[0].index == 1

    file = VideoFile.upload(video_path.read_bytes(), video_path.name)
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
def test_save_video_fragment_uses_explicit_format(tmp_path, video_file):
    fragment = save_video_fragment(
        video_file.as_video_file(), 0, 1, str(tmp_path / "out"), format="avi"
    )

    assert fragment.path.endswith(".avi")
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
    assert args[:4] == ["-nostdin", "-hide_banner", "-loglevel", "error"]
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
