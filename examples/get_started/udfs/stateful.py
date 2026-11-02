"""
Compute image embeddings in batches with a model loaded once per worker.

To install the required dependencies:

  pip install datachain[examples]

"""

import open_clip
import torch

import datachain as dc
from datachain import C


class ClipImageEncoder:
    def __init__(self, model_name: str, pretrained: str):
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, self.pretrained
        )
        self.model.eval()


def embeddings(
    file: list[dc.ImageFile], encoder: ClipImageEncoder
) -> list[list[float]]:
    images = torch.stack([encoder.preprocess(image.read()) for image in file])
    with torch.inference_mode():
        return encoder.model.encode_image(images).tolist()


if __name__ == "__main__":
    (
        dc.read_storage("gs://datachain-demo/dogs-and-cats/", type="image", anon=True)
        .filter(C("file.path").glob("*cat*.jpg"))
        .limit(5)
        .settings(parallel=True)
        .setup(encoder=lambda: ClipImageEncoder("ViT-B-32", "laion2b_s34b_b79k"))
        .batch_map(emb=embeddings, batch=5)
        .show()
    )
