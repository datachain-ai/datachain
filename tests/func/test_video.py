import io
import os
import shutil

import pytest
from numpy import ndarray
from PIL import Image

from datachain import VideoFragment, VideoFrame
from datachain.lib.file import File, FileError, ImageFile, VideoFile
from datachain.lib.video import save_video_fragment, video_frame_np

requires_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg"), reason="ffmpeg not installed"
)


@pytest.fixture(autouse=True)
def video_file(catalog) -> File:
    data_path = os.path.join(os.path.dirname(__file__), "data")
    file_name = "Big_Buck_Bunny_360_10s_1MB.mp4"

    with open(os.path.join(data_path, file_name), "rb") as f:
        file = File.upload(f.read(), file_name)

    file.ensure_cached()
    return file


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

    file.ensure_cached()
    with pytest.raises(FileError):
        file.get_info()


def test_get_frame(video_file):
    frame = video_file.as_video_file().get_frame(37)
    assert isinstance(frame, VideoFrame)
    assert frame.frame == 37


def test_get_frame_error(video_file):
    with pytest.raises(ValueError):
        video_file.as_video_file().get_frame(-1)


def test_get_frame_np(video_file):
    frame = video_file.as_video_file().get_frame(0).get_np()
    assert isinstance(frame, ndarray)
    assert frame.shape == (360, 640, 3)


def test_get_frame_np_error(video_file):
    with pytest.raises(ValueError):
        video_frame_np(video_file.as_video_file(), -1)


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
    with Image.open(frame_file.get_local_path()) as img:
        assert img.format == "JPEG"
        assert img.size == (640, 360)


def test_get_frames(video_file):
    frames = list(video_file.as_video_file().get_frames(10, 200, 5))
    assert len(frames) == 38
    assert all(isinstance(frame, VideoFrame) for frame in frames)


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
        with Image.open(frame_file.get_local_path()) as img:
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
