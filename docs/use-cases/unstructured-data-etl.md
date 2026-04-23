---
title: Unstructured Data ETL
---

# Unstructured Data ETL

DataChain provides a Pythonic framework for transforming and enriching unstructured data -- images, audio, video, text, and PDFs -- without copying files from their original storage.

## Image Processing Pipeline

```python
from transformers import pipeline
import datachain as dc

chain = (
    dc.read_storage("s3://bucket/images/", type="image")
    .settings(parallel=8, cache=True)
    .setup(pipe=lambda: pipeline("image-to-text", model="Salesforce/blip-image-captioning-large"))
    .map(caption=lambda file, pipe: pipe(file.read().convert("RGB"))[0]["generated_text"])
    .save("image_captions")
)
```

## PDF Chunking

Split PDFs into embedding-ready chunks using a generator:

```python
import datachain as dc
from pydantic import BaseModel
from typing import Iterator

class Chunk(BaseModel):
    text: str
    page: int

def split_pdf(file: dc.File) -> Iterator[Chunk]:
    for i, page in enumerate(extract_pages(file)):
        yield Chunk(text=page.text, page=i)

chain = (
    dc.read_storage("s3://docs/*.pdf")
    .gen(chunk=split_pdf)
    .save("pdf_chunks")
)
```

## Video Frame Extraction

```python
import datachain as dc

chain = (
    dc.read_storage("s3://bucket/videos/", type="video")
    .gen(frame=lambda file: file.get_frames(step=30))
    .save("video_frames")
)
```

## Audio Segmentation

```python
import datachain as dc
from pydantic import BaseModel
from typing import Iterator

class AudioSegment(BaseModel):
    audio: dc.AudioFragment
    channel: str
    rms: float

def segment_audio(file: dc.AudioFile) -> Iterator[AudioSegment]:
    for frag in file.get_fragments(duration=10.0):
        yield AudioSegment(audio=frag, channel="mono", rms=compute_rms(frag))

chain = (
    dc.read_storage("s3://audio/", type="audio")
    .gen(segm=segment_audio)
    .save("audio_segments")
)
```

## Merging Files with Metadata

The most common pattern: files in storage, annotations in a sidecar format.

```python
import datachain as dc

images = dc.read_storage("gs://bucket/images/*jpg", anon=True)
meta = dc.read_json("gs://bucket/images/*json", column="meta", anon=True)

images_id = images.map(id=lambda file: file.path.split(".")[-2])
annotated = images_id.merge(meta, on="id", right_on="meta.id")

high_conf = annotated.filter(
    (dc.C("meta.inference.confidence") > 0.93)
    & (dc.C("meta.inference.class_") == "cat")
)
high_conf.to_storage("high-confidence-cats/", signal="file")
```

## Delta Processing for Growing Datasets

Process only new and changed files on each run:

```python
import datachain as dc

chain = (
    dc.read_storage(
        "s3://bucket/incoming/",
        update=True,
        delta=True,
        delta_on="file.path",
    )
    .settings(parallel=8)
    .map(result=process_file)
    .save("processed_incoming")
)
```

Each run adds new files to the dataset while preserving all previously processed results.
