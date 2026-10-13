---
title: Files and Types
---

# Files and Types

DataChain's type system is built on Pydantic. Every chain carries a schema, every column has a type, and custom models integrate automatically.

## File Abstraction

`File` is the bridge between object storage and the data layer. Every chain begins with files: `read_storage()` produces a chain of File objects. File carries storage coordinates and provides methods to read content.

### Storage Coordinates

File tracks everything needed to locate and identify a blob: `source`, `path`, `version`, `etag`, `size`, `is_latest`, and `last_modified`. This metadata is indexed by the Dataset DB and available for chain operations without touching the actual bytes.

### Content Access

```python
import datachain as dc

file = dc.File.at("s3://bucket/path/to/file.png")

content = file.read()  # bytes
text = file.read_text()  # str

with file.open("rb") as f:  # stream large files
    chunk = f.read(4096)

file.ensure_cached()  # cache locally
local = file.get_local_path()  # local path after caching

file.export("/local/output/", placement="filename")
file.save("s3://bucket/output/result.png")
```

File inherits from DataModel (a Pydantic BaseModel subclass), so it participates in schemas, gets stored as columns, and flows through all chain operations.

## Modality Types

File has specialized subclasses for each data modality. Each adds domain-specific methods while inheriting all of File's capabilities.

```python
import datachain as dc

images = dc.read_storage("s3://bucket/images/", type="image")  # ImageFile
videos = dc.read_storage("s3://bucket/videos/", type="video")  # VideoFile
audio = dc.read_storage("s3://bucket/audio/", type="audio")  # AudioFile
```

**ImageFile**: `read()` returns a `PIL.Image.Image`. `get_info()` returns Image metadata (width, height, format). `save()` supports format conversion.

**VideoFile**: `get_frame(frame, video_stream_index=0)` returns one frame reference with a timestamp estimated from FPS metadata. Pixel access may seek to a nearby keyframe and decode forward. `get_frames(start, end, step, video_stream_index=0)` yields VideoFrame objects with frame indexes, video stream indexes, and decoded frame timestamps when available, and is preferred for sequential frame reads. `get_fragments(duration, start, end)` yields VideoFragment time slices. `get_info(video_stream_index=0)` returns Video metadata (fps, duration, codec, resolution).

**AudioFile**: `get_fragments(duration, start, end)` yields AudioFragment chunks. `get_info()` returns Audio metadata (sample_rate, channels, duration, codec). `save()` supports format conversion.

### Sub-File Units

Videos and audio tracks can be sliced into smaller units that are themselves DataModels:

```python
import datachain as dc

# Expand one video into thousands of frames
chain = (
    dc.read_storage("s3://bucket/videos/", type="video")
    .gen(frame=lambda file: file.get_frames(step=30))
    .save("video_frames")
)

# Time-based slicing
from typing import Iterator


def split_into_clips(file: dc.VideoFile) -> Iterator[dc.VideoFragment]:
    yield from file.get_fragments(duration=10.0)


chain = (
    dc.read_storage("s3://videos/", type="video")
    .gen(frag=split_into_clips)
    .save("video_clips")
)
```

## Annotation Types

Built-in DataModels for annotation primitives: BBox, OBBox, Pose, Pose3D, Segment.

```python
from datachain import model

# BBox with format conversion
bbox = model.BBox.from_coco([100, 150, 200, 300], title="car")
bbox = model.BBox.from_yolo([0.5, 0.5, 0.4, 0.6], img_size=(640, 480), title="car")

coco = bbox.to_coco()  # [x, y, w, h]
yolo = bbox.to_yolo(img_size=(640, 480))  # normalized [cx, cy, w, h]

bbox.point_inside(300, 250)  # spatial queries
```

Annotation types compose naturally into Pydantic models:

```python
from pydantic import BaseModel
from datachain import model


class YoloPose(BaseModel):
    bbox: model.BBox
    pose: model.Pose
    confidence: float
```

## DataModel and Custom Types

Use `pydantic.BaseModel` directly for custom types; DataChain accepts it natively:

```python
from pydantic import BaseModel
import datachain as dc

class AudioSegment(BaseModel):
    audio: dc.AudioFragment
    id: int
    channel: str
    rms: float

def get_segments(file: dc.AudioFile) -> Iterator[AudioSegment]:
    ...
    yield AudioSegment(audio=file, ...)

chain = (
    dc.read_storage("s3://mybucket/audio_dir", type="audio")
    .gen(segm=get_segments)
    .save("audio_segments")
)
```

For external Pydantic models (like Mistral's `ChatCompletionResponse`), register them explicitly: `dc.DataModel.register(MistralResponse)`.

## Union and Optional Fields

A field or a signal can hold one of several types. Each type gets its own columns,
plus a hidden discriminator naming the type a row carries, so values keep their type
through saving, reading and exporting. `X | None` (the same as `Optional[X]`) is the
single-type case of the same mechanism.

```python
import datachain as dc


class Cat(dc.DataModel):
    name: str
    lives: int = 9


class Dog(dc.DataModel):
    name: str
    breed: str = ""


class Sighting(dc.DataModel):
    pet: Cat | Dog  # one of two models
    weight: int | float  # one of two scalars
    note: str | None = None  # optional


chain = dc.read_values(
    id=[1, 2],
    sighting=[
        Sighting(pet=Cat(name="Tom"), weight=4),
        Sighting(pet=Dog(name="Rex", breed="lab"), weight=12.5, note="stray"),
    ],
    output={"id": int, "sighting": Sighting},
)

seen = chain.order_by("id").to_values("sighting")
seen[0].pet  # Cat(name='Tom', lives=9) -- the model type is preserved
seen[1].weight  # 12.5, a float; seen[0].weight is 4, an int
```

A union is also a signal type on its own, and a UDF can return one:

```python
chain = dc.read_values(value=[42, "hello"], output={"value": int | str})
chain.map(pet=classify, output={"pet": Cat | Dog})
```

To reference one of the types, name it in the path; the name can be dropped when a
single type has that field. Functions read the discriminator rather than a column:

```python
chain.filter(dc.C("sighting.pet.Cat.name") == "Tom")  # type-qualified path
chain.filter(dc.C("sighting.pet.breed") == "lab")  # only Dog has `breed`
chain.filter(dc.func.isnone("sighting.note"))  # rows where the field is None
```

`select()`, `distinct()` and `group_by()` carry the whole union, so a value is never
separated from the type it belongs to; for the same reason a union cannot be a key
for `diff()` or `subtract()`.

Inside a `list` or a `dict` there are no per-element columns — the collection is one
JSON value — so types that JSON stores alike are rejected when the chain is created:
models whose field names are the same or a subset of one another, `datetime | str`,
and a union as a dict key. Give such models a `kind: Literal["..."]` field to tell
them apart. A collection as one of the types (`str | list[str]`) is not supported.

## Core Classes

- **DataChain**: the core class for composing queries with 60+ methods
- **DataModel**: Pydantic base for structured types
- **Column** (aliased as `C`): references a column by name for vectorized expressions
- **File**, **ImageFile**, **VideoFile**, **AudioFile**, **TextFile**: storage-native file types
- **BBox**, **OBBox**, **Pose**, **Pose3D**, **Segment**: annotation types (in `datachain.model`)
