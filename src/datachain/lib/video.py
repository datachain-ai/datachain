import os
import posixpath
import shutil
import tempfile

from fsspec.utils import stringify_path
from numpy import ndarray

from datachain.lib.file import File, FileError, ImageFile, Video, VideoFile

try:
    import ffmpeg
    import imageio.v3 as iio
except ImportError as exc:
    raise ImportError(
        "Missing dependencies for processing video.\n"
        "To install run:\n\n"
        "  pip install 'datachain[video]'\n"
    ) from exc


def video_info(file: File | VideoFile) -> Video:
    """
    Returns video file information.

    Args:
        file (VideoFile): Video file object.

    Returns:
        Video: Video file information.
    """
    file = file.as_video_file()

    if not (file_path := file.get_local_path()):
        file.ensure_cached()
        file_path = file.get_local_path()
        if not file_path:
            raise FileError("unable to download video file", file.source, file.path)

    try:
        probe = ffmpeg.probe(file_path)
    except Exception as exc:
        raise FileError(
            "unable to extract metadata from video file", file.source, file.path
        ) from exc

    all_streams = probe.get("streams")
    video_format = probe.get("format")
    if not all_streams or not video_format:
        raise FileError(
            "unable to extract metadata from video file", file.source, file.path
        )

    video_streams = [s for s in all_streams if s["codec_type"] == "video"]
    if len(video_streams) == 0:
        raise FileError(
            "unable to extract metadata from video file", file.source, file.path
        )

    video_stream = video_streams[0]

    r_frame_rate = video_stream.get("r_frame_rate", "0")
    if "/" in r_frame_rate:
        num, denom = r_frame_rate.split("/")
        fps = float(num) / float(denom)
    else:
        fps = float(r_frame_rate)

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    duration = float(video_format.get("duration", 0))
    if "nb_frames" in video_stream:
        frames = int(video_stream.get("nb_frames", 0))
    else:
        start_time = float(video_format.get("start_time", 0))
        frames = int((duration - start_time) * fps)
    format_name = video_format.get("format_name", "")
    codec_name = video_stream.get("codec_name", "")

    return Video(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        frames=frames,
        format=format_name,
        codec=codec_name,
    )


def video_frame_np(video: VideoFile, frame: int) -> ndarray:
    """
    Reads video frame from a file and returns as numpy array.

    Args:
        video (VideoFile): Video file object.
        frame (int): Frame index.

    Returns:
        ndarray: Video frame.
    """
    if frame < 0:
        raise ValueError("frame must be a non-negative integer")

    with video.open() as f:
        return iio.imread(f, index=frame, plugin="pyav")  # type: ignore[arg-type]


def validate_frame_range(
    video: VideoFile,
    start: int = 0,
    end: int | None = None,
    step: int = 1,
) -> tuple[int, int, int]:
    """
    Validates frame range for a video file.

    Args:
        video (VideoFile): Video file object.
        start (int): Start frame index (default: 0).
        end (int, optional): End frame index (default: None).
        step (int): Step between frames (default: 1).

    Returns:
        tuple[int, int, int]: Start frame index, end frame index, and step.
    """
    if start < 0:
        raise ValueError("start_frame must be a non-negative integer.")
    if step < 1:
        raise ValueError("step must be a positive integer.")

    if end is None:
        end = video_info(video).frames

    if end < 0:
        raise ValueError("end_frame must be a non-negative integer.")
    if start > end:
        raise ValueError("start_frame must be less than or equal to end_frame.")

    return start, end, step


def video_frame_bytes(video: VideoFile, frame: int, format: str = "jpg") -> bytes:
    """
    Reads video frame from a file and returns as image bytes.

    Args:
        video (VideoFile): Video file object.
        frame (int): Frame index.
        format (str): Image format (default: 'jpg').

    Returns:
        bytes: Video frame image as bytes.
    """
    img = video_frame_np(video, frame)
    return iio.imwrite("<bytes>", img, extension=f".{format}")


def save_video_frame(
    video: VideoFile,
    frame: int,
    destination: str | os.PathLike[str],
    format: str = "jpg",
    client_config: dict | None = None,
) -> ImageFile:
    """
    Saves video frame as a new image file. If ``destination`` is a remote
    path, the image will be uploaded to remote storage.

    Args:
        video: Video file object.
        frame: Frame index.
        destination: Output directory path or URI (e.g. ``s3://…``, ``gs://…``).
        format: Image format (default: 'jpg').
        client_config: Optional client configuration (e.g. credentials).

    Returns:
        ImageFile: Image file model.
    """
    destination = stringify_path(destination)
    img = video_frame_bytes(video, frame, format=format)
    output_file = posixpath.join(
        destination, f"{video.get_file_stem()}_{frame:04d}.{format}"
    )
    client, rel_path = video._resolve_destination(output_file, client_config)
    result = client.upload(img, rel_path)
    image = ImageFile(**result.model_dump())
    image._set_stream(video._catalog)
    return image


def save_video_fragment(
    video: VideoFile,
    start: float,
    end: float,
    destination: str | os.PathLike[str],
    format: str | None = None,
    client_config: dict | None = None,
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

    Returns:
        VideoFile: Video fragment model.
    """
    destination = stringify_path(destination)

    if start < 0 or end < 0 or start >= end:
        raise ValueError(
            f"Can't save video fragment for '{video.path}', "
            f"invalid time range: ({start:.3f}, {end:.3f})"
        )

    if format is None:
        format = video.get_file_ext()

    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    output_file = posixpath.join(
        destination, f"{video.get_file_stem()}_{start_ms:06d}_{end_ms:06d}.{format}"
    )

    temp_dir = tempfile.mkdtemp()
    try:
        output_file_tmp = posixpath.join(temp_dir, posixpath.basename(output_file))
        ffmpeg.input(
            video.get_local_path(),
            ss=start,
            to=end,
        ).output(output_file_tmp).run(quiet=True)

        with open(output_file_tmp, "rb") as f:
            data = f.read()
    finally:
        shutil.rmtree(temp_dir)

    client, rel_path = video._resolve_destination(output_file, client_config)
    result = client.upload(data, rel_path)
    vf = VideoFile(**result.model_dump())
    vf._set_stream(video._catalog)
    return vf
