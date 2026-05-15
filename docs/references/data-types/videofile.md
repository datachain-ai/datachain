# VideoFile

`VideoFile` extends [`File`](file.md) and provides additional methods for working with video files.

`VideoFile` instances are created when a `DataChain` is initialized [from storage](../datachain.md#datachain.lib.dc.storage.read_storage) with the `type="video"` parameter:

```python
import datachain as dc

chain = dc.read_storage("s3://bucket-name/", type="video")
```

There are additional models for working with video files:

- `VideoFrame` - represents a single frame of a video file, including its video stream index, frame index, and timestamp.
- `VideoFragment` - represents a fragment of a video file.

`video_stream_index` arguments are zero-based indexes among video streams, matching FFmpeg `v:N` and PyAV `container.streams.video[N]` selectors.

`VideoFile.get_frame()` returns a single frame reference with a timestamp estimated from FPS metadata. Pixel access may seek to a nearby keyframe and decode forward; use `VideoFile.get_frames()` for sequential access and decoded frame timestamps when available.

These are virtual models that do not create physical files.
Instead, they are used to represent the data in the `VideoFile` these models are referring to.
If you need to save the data, you can use the `save` method of these models,
allowing you to save data locally or upload it to a storage service.

::: datachain.lib.file.VideoFile

::: datachain.lib.file.VideoFrame

::: datachain.lib.file.VideoFragment

::: datachain.lib.file.Video
