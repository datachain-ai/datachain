"""
To run this example, install (on Python >= 3.11, which Zarr support requires):

`datachain[examples,zarr]`

It demonstrates how to use DataChain's Zarr support to turn a directory of
Zarr stores into an embedding dataset. Each store is one robot-manipulation
episode holding a stack of RGB camera frames plus aligned low-dimensional
state arrays.

`read_zarr` discovers every store under the prefix and yields one row per
store as a `ZarrStore`. We first attach the per-frame metadata read straight
from the Zarr stores (frame to embed, resolution, task instruction), then as
a separate step encode that frame with OpenCLIP and keep only the compact
embedding — no raw pixels are carried in the row. The result is saved as a
DataChain dataset that can be reused for search or clustering.
"""

import numpy as np
import open_clip
from PIL import Image

import datachain as dc
from datachain.lib.zarr import ZarrStore

# Public, anonymous bucket: one ``*.zarr`` store per episode. Each store has:
#   workspace_rgb        (T, 480, 640, 3) uint8   -- RGB camera frames
#   joint_pos_lowdim     (T, 6)           float32 -- joint positions
#   joint_action_lowdim  (T, 6)           float32 -- commanded actions
#   language_instruction (1,)             str     -- the task description
SOURCE = "gs://datachain-demo/robot-frames-zarr"


class ClipImageEncoder:
    def __init__(
        self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"
    ):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained
        )


def frame_meta(episode: ZarrStore) -> tuple[int, str, str]:
    # Read straight from the Zarr metadata/small arrays: the frame index we
    # will embed, the image resolution, and the task instruction.
    rgb = episode.get_array("workspace_rgb")
    frame_idx = rgb.shape[0] // 2
    resolution = f"{rgb.shape[2]}x{rgb.shape[1]}"
    instruction = bytes(episode.get_array("language_instruction").read()[0]).decode()
    return frame_idx, resolution, instruction


def embed_frame(
    episode: ZarrStore, frame_idx: int, encoder: ClipImageEncoder
) -> list[float]:
    # Read only the selected frame's chunk, then embed it with OpenCLIP. The
    # heavy pixel data never enters the row — only the float embedding does.
    frame = np.asarray(episode.get_array("workspace_rgb").select(frame_idx).read())
    image = encoder.preprocess(Image.fromarray(frame.astype("uint8"))).unsqueeze(0)
    return encoder.model.encode_image(image)[0].tolist()


ds = (
    dc.read_zarr(SOURCE, column="episode", anon=True)
    .limit(3)
    .map(frame_meta, output=("frame_idx", "resolution", "instruction"))
    .setup(encoder=lambda: ClipImageEncoder())  # noqa: PLW0108
    .map(embedding=embed_frame)
    .save("robot_frame_embeddings")
)

ds.select("frame_idx", "resolution", "instruction").show()
