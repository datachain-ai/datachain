"""
To run this example, install:

`datachain[examples]`

It demonstrates how to use DataChain models like Audio, AudioFile,
and AudioFragment to efficiently access audio files, chunk them into
fragments, pass them to a model to get text.
"""

from collections.abc import Iterator

import torch
from transformers import Pipeline, pipeline

import datachain as dc
from datachain import Audio, AudioFile, AudioFragment, C


def info(file: AudioFile) -> Audio:
    return file.get_info()


def fragments(file: AudioFile, meta: Audio) -> Iterator[AudioFragment]:
    start = meta.duration / 3
    fragment = file.get_fragment(start, start + 10)
    yield fragment


def process(fragment: AudioFragment, pipeline: Pipeline) -> str:
    audio_array, _ = fragment.get_np()

    # Convert to mono if stereo (average the channels)
    if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
        audio_array = audio_array.mean(axis=1)

    # Ensure audio_array is 1D and float32
    if len(audio_array.shape) > 1:
        audio_array = audio_array.squeeze()
    audio_array = audio_array.astype("float32")

    # Use chunk_length_s to avoid the stride=None code path that requires num_frames
    # This is a workaround for transformers 5.0 compatibility
    result = pipeline(
        audio_array,
        chunk_length_s=30,  # Process in 30-second chunks
        generate_kwargs={"language": "en"},
    )
    return str(result["text"])


# We disable caching and prefetching to ensure that we read only bytes
# that we need for processing. Methods like `get_info` and `get_fragment`
# don't require reading the entire file.
(
    dc.read_storage("gs://datachain-demo/musdb18", type="audio", anon=True)
    .filter(C("file.path").glob("*.wav"))
    .limit(3)
    .settings(cache=False, prefetch=False, parallel=True)
    .map(meta=info)
    .gen(fragment=fragments)
    .setup(
        pipeline=lambda: pipeline(
            "automatic-speech-recognition",
            "openai/whisper-small",
            torch_dtype=torch.float32,
            device="cpu",
        )
    )
    .map(text=process)
    .show()
)
