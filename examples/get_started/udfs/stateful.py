"""
Compute image embeddings in batches with a model loaded once per process.

To install the required dependencies:

  pip install datachain[examples]

"""

import open_clip
import torch

import datachain as dc
from datachain import C


def load_encoder():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    return model, preprocess


def embeddings(file: list[dc.ImageFile], encoder) -> list[list[float]]:
    model, preprocess = encoder
    images = torch.stack([preprocess(image.read()) for image in file])
    with torch.inference_mode():
        return model.encode_image(images).tolist()


if __name__ == "__main__":
    (
        dc.read_storage("gs://datachain-demo/dogs-and-cats/", type="image", anon=True)
        .filter(C("file.path").glob("*cat*.jpg"))
        .limit(5)
        .setup(encoder=load_encoder)
        .batch_map(emb=embeddings, batch=5)
        .show()
    )
