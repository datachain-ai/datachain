import os
import random
from contextlib import closing

import pytest
import torch
from datasets import load_dataset
from torch import Size, Tensor
from torchvision.datasets import FakeData
from torchvision.transforms import v2

import datachain as dc
from datachain.lib.file import File
from datachain.lib.pytorch import PytorchDataset


@pytest.fixture(scope="module")
def fake_image_dir(tmp_path_factory):
    # Create fake images in labeled dirs
    data_path = tmp_path_factory.mktemp("data")
    for i, (img, label) in enumerate(FakeData(size=100)):
        label = str(label)
        (data_path / label).mkdir(parents=True, exist_ok=True)
        img.save(data_path / label / f"{i}.jpg")
    return data_path


@pytest.fixture
def fake_dataset(catalog, fake_image_dir):
    # Create dataset from images
    uri = fake_image_dir.as_uri()
    return (
        dc.read_storage(uri, type="image")
        .settings(prefetch=0, cache=False)
        .map(text=lambda file: file.parent.split("/")[-1], output=str)
        .map(label=int, output=int)
        .save("fake")
    )


def fake_tokenizer(text: str):
    return [random.randint(0, 100) for _ in text]  # noqa: S311


def test_pytorch_dataset(fake_dataset):
    transform = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((64, 64))]
    )
    pt_dataset = PytorchDataset(
        name=fake_dataset.name,
        version=fake_dataset.version,
        transform=transform,
        tokenizer=fake_tokenizer,
    )
    img, text, label = next(iter(pt_dataset))
    assert isinstance(img, Tensor)
    assert isinstance(text, Tensor)
    assert isinstance(label, int)
    assert img.size() == Size([3, 64, 64])


def test_pytorch_dataset_sample(fake_dataset):
    transform = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((64, 64))]
    )
    pt_dataset = PytorchDataset(
        name=fake_dataset.name,
        version=fake_dataset.version,
        transform=transform,
        num_samples=50,
    )
    assert len(list(pt_dataset)) == 50


def test_to_pytorch(fake_dataset):
    from torch.utils.data import IterableDataset

    transform = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((64, 64))]
    )
    pt_dataset = fake_dataset.to_pytorch(transform=transform, tokenizer=fake_tokenizer)
    assert isinstance(pt_dataset, IterableDataset)
    img, text, label = next(iter(pt_dataset))
    assert isinstance(img, Tensor)
    assert isinstance(text, Tensor)
    assert isinstance(label, int)
    assert img.size() == Size([3, 64, 64])


@pytest.mark.parametrize("use_cache", (True, False))
@pytest.mark.parametrize("prefetch", (0, 2))
@pytest.mark.parametrize("remove_prefetched", (True, False))
def test_prefetch(
    mocker, catalog, fake_dataset, use_cache, prefetch, remove_prefetched
):
    catalog.cache.clear()

    dataset = fake_dataset.limit(10)
    ds = dataset.settings(cache=use_cache, prefetch=prefetch).to_pytorch(
        remove_prefetched=remove_prefetched
    )

    iter_with_prefetch = ds._iter_with_prefetch
    cache = ds._cache

    def is_prefetched(file: File):
        assert file._catalog
        assert file._catalog.cache == cache
        return cache.contains(file)

    def check_prefetched():
        seen = []
        for row in iter_with_prefetch():
            # prefetched files should persist if `cache` is enabled, or
            # `remove_prefetched` is not set. Otherwise, the old prefetched
            # files should be removed.
            for f in seen:
                assert is_prefetched(f) == (cache and not remove_prefetched)

            files = [f for f in row if isinstance(f, File)]
            seen.extend(files)
            assert files
            files_not_in_cache = [f for f in files if not is_prefetched(f)]
            if prefetch:
                assert not files_not_in_cache, "Some files are not in cache"
            else:
                assert files == files_not_in_cache, "Some files are in cache"
            yield row

    # we peek internally with `_iter_with_prefetch` to check if the files are prefetched
    # as `__iter__` transforms them.
    m = mocker.patch.object(ds, "_iter_with_prefetch", wraps=check_prefetched)
    with closing(ds), closing(iter(ds)) as rows:
        assert next(rows)
    m.assert_called_once()
    # prefetch cache directory should be removed after `close()`
    assert os.path.exists(cache.cache_dir) == (use_cache or not prefetch)


def test_hf_to_pytorch(catalog, fake_image_dir):
    hf_ds = load_dataset("imagefolder", data_dir=fake_image_dir)
    chain = dc.read_hf(hf_ds)
    pt_ds = chain.order_by("label").to_pytorch()
    img, label = next(iter(pt_ds))
    assert isinstance(img, Tensor)
    assert label == 0
