import ast
import os
import re
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

import datachain as dc
from datachain.cli.commands.skill import _skills_src


@pytest.mark.parametrize("document", ["datachain/SKILL.md", "core/SDK.md"])
def test_skill_inference_function_loads_once_and_saves_each_image(
    test_session, tmp_path, monkeypatch, document
):
    torch = pytest.importorskip("torch")
    calls = {"setup": 0, "batches": []}

    class Encoder:
        def eval(self):
            return self

        def encode_image(self, images):
            assert torch.is_inference_mode_enabled()
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
    skill = (_skills_src() / document).read_text(encoding="utf-8")
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
        chain = dc.read_storage(
            tmp_path.as_uri(), type="image", session=test_session
        ).setup(encoder=recipe.load_encoder)
        if document == "core/SDK.md":
            result = chain.map(emb=recipe.encode)
            expected_batches = [1, 1, 1]
            embedding = "emb"
        else:
            result = chain.batch_map(img=recipe.embed_images, batch=2)
            expected_batches = [2, 1]
            embedding = "img.embedding"
        result = result.save("recipe_embeddings").order_by("file.path")
        rows = result.to_list(embedding)
        if document == "datachain/SKILL.md":
            assert result.to_list("img.width", "img.height") == [(2, 3), (3, 4), (4, 5)]
    finally:
        torch.set_num_threads(previous_threads)
    assert calls == {"setup": 1, "batches": expected_batches}
    assert rows == [([1.0, 0.0, 0.0],), ([0.0, 1.0, 0.0],), ([0.0, 0.0, 1.0],)]


def _recipe(heading: str, uri: str, src) -> str:
    skill = (_skills_src() / "datachain" / "SKILL.md").read_text(encoding="utf-8")
    section = skill.split(f"### {heading}", 1)[1].split("\n### ", 1)[0]
    (block,) = re.findall(r"```python\n(.*?)```", section, re.DOTALL)
    return block.replace(uri, f"{src.as_uri()}/")


def test_skill_changes_recipe_reports_added_modified_deleted(
    test_session, tmp_path, capsys
):
    src = tmp_path / "data"
    src.mkdir()
    for name in ("kept", "changed", "deleted"):
        (src / name).write_text(name)
    recipe = _recipe("What changed since the last run", "s3://bucket/prefix/", src)

    def run() -> list:
        with patch(
            "datachain.lib.listing.write_listing",
            wraps=dc.lib.listing.write_listing,
        ) as write_listing:
            exec(recipe, {})  # noqa: S102
        assert write_listing.call_count == 1
        out = capsys.readouterr().out.strip().splitlines()
        return sorted(ast.literal_eval(out[-1])) if out else []

    assert run() == []
    (src / "changed").write_text("changed again")
    os.utime(src / "changed", (1, 1))
    (src / "deleted").unlink()
    (src / "added").write_text("added")
    assert run() == [("A", "added"), ("D", "deleted"), ("M", "changed")]
    assert run() == []
