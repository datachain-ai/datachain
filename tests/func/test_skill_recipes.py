import ast
import re
import sys
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

import datachain as dc
from datachain.cli.commands.skill import _skills_src


def test_skill_inference_function_loads_once_and_saves_each_image(
    test_session, tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    calls = {"setup": 0, "batches": []}

    class Encoder:
        def eval(self):
            return self

        def encode_image(self, images):
            calls["batches"].append(len(images))
            return images

    def preprocess(image):
        return torch.tensor(np.array(image), dtype=torch.float32).mean(dim=(0, 1))

    def load(*args, **kwargs):
        calls["setup"] += 1
        return Encoder(), None, preprocess

    open_clip = ModuleType("open_clip")
    open_clip.create_model_and_transforms = load
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)
    recipe = ModuleType("datachain_test_inference_recipe")
    monkeypatch.setitem(sys.modules, recipe.__name__, recipe)
    skill = (_skills_src() / "datachain" / "SKILL.md").read_text(encoding="utf-8")
    (block,) = [
        block
        for block in re.findall(r"```python\n(.*?)```", skill, re.DOTALL)
        if "def load_encoder(" in block
    ]
    definitions = ast.Module(
        body=[
            node
            for node in ast.parse(block).body
            if isinstance(
                node, ast.Import | ast.ImportFrom | ast.FunctionDef | ast.ClassDef
            )
        ],
        type_ignores=[],
    )
    exec(compile(definitions, "skill recipe", "exec"), recipe.__dict__)  # noqa: S102
    for index, color in enumerate(["red", "lime", "blue"]):
        Image.new("RGB", (index + 2, index + 3), color).save(tmp_path / f"{index}.png")

    previous_threads = torch.get_num_threads()
    try:
        result = (
            dc.read_storage(tmp_path.as_uri(), type="image", session=test_session)
            .setup(encoder=recipe.load_encoder)
            .batch_map(img=recipe.embed_images, batch=2)
            .save("recipe_embeddings")
        )
        rows = result.order_by("file.path").to_list(
            "img.width", "img.height", "img.embedding"
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert calls == {"setup": 1, "batches": [2, 1]}
    assert rows == [
        (2, 3, [1.0, 0.0, 0.0]),
        (3, 4, [0.0, 1.0, 0.0]),
        (4, 5, [0.0, 0.0, 1.0]),
    ]
