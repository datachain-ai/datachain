"""
To install the required dependencies:

  uv pip install "datachain[examples]"

HuggingFace pipeline helper model zoo demo.
Runs various HuggingFace pipelines on images, audio, and text data.
"""

# NOTE: also need to install ffmpeg binary
import json
import os
import subprocess

import torch
from transformers import Pipeline, pipeline

import datachain as dc
from datachain import C, File

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class Helper(dc.Mapper):
    def __init__(self, model, device, **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs

    def setup(self):
        self.helper = pipeline(model=self.model, device=self.device)

    def process(self, file):
        imgs = file.read()
        result = self.helper(
            imgs,
            **self.kwargs,
        )
        return (json.dumps(result), "")


def process(file: File, pipeline: Pipeline, args: dict) -> str:
    result = pipeline(file.read(), **args)
    return json.dumps(result)


image_source = "gs://datachain-demo/dogs-and-cats/"
audio_source = "gs://datachain-demo/speech-emotion-recognition-dataset/"
text_source = "gs://datachain-demo/nlp-cnn-stories"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


if __name__ == "__main__":
    print("** HuggingFace pipeline helper model zoo demo **")
    print("\nZero-shot object detection and classification:")
    (
        dc.read_storage(image_source, anon=True, type="image")
        .filter(C("file.path").glob("*.jpg"))
        .limit(1)
        .setup(
            pipeline=lambda: pipeline(model="google/owlv2-base-patch16", device=device),
            args=lambda: {"candidate_labels": ["cat", "dog", "squirrel", "unknown"]},
        )
        .map(result=process)
        .show()
    )

    print("\nNot-safe-for-work image detection:")
    (
        dc.read_storage(image_source, anon=True, type="image")
        .filter(C("file.path").glob("*.jpg"))
        .limit(1)
        .setup(
            pipeline=lambda: pipeline(
                model="Falconsai/nsfw_image_detection", device=device
            ),
            args=dict,
        )
        .map(result=process)
        .show()
    )

    print("\nAudio emotion classification:")
    try:
        subprocess.run(["ffmpeg", "-L"], check=True)  # noqa: S607
        (
            dc.read_storage(audio_source, anon=True, type="audio")
            .filter(dc.C("file.path").glob("*.wav"))
            .limit(1)
            .setup(
                pipeline=lambda: pipeline(
                    model="Krithika-p/my_awesome_emotions_model", device=device
                ),
                args=dict,
            )
            .map(result=process)
            .show()
        )
    except FileNotFoundError:
        print("ffmpeg binary not found, skipping audio example")

    print("\nLong text summarization:")
    (
        dc.read_storage(text_source, anon=True, type="text")
        .filter(C("file.path").glob("*.story"))
        .limit(1)
        .setup(
            pipeline=lambda: pipeline(
                task="text-generation",
                model="pszemraj/led-large-book-summary",
                device=device,
            ),
            args=lambda: {"max_length": 150},
        )
        .map(result=process)
        .show()
    )
