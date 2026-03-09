import logging
import os
import weakref
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import closing
from typing import TYPE_CHECKING, Any

from PIL import Image
from torch import float32
from torch.distributed import get_rank, get_world_size
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms import v2

from datachain import Session
from datachain.cache import get_temp_cache
from datachain.catalog import Catalog, get_catalog
from datachain.lib.dc.datasets import read_dataset
from datachain.lib.settings import Settings
from datachain.lib.text import convert_text
from datachain.progress import CombinedDownloadCallback
from datachain.query.dataset import get_download_callback

if TYPE_CHECKING:
    from torchvision.transforms.v2 import Transform

    from datachain.cache import Cache


logger = logging.getLogger("datachain")


DEFAULT_TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(float32, scale=True)])


def label_to_int(value: str, classes: list) -> int:
    """Given a value and list of classes, return the index of the value's class."""
    return classes.index(value)


class PytorchDataset(IterableDataset):
    prefetch: int = 2

    def __init__(
        self,
        name: str,
        version: str | None = None,
        catalog: Catalog | None = None,
        transform: "Transform | None" = None,
        tokenizer: Callable | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        num_samples: int = 0,
        dc_settings: Settings | None = None,
        remove_prefetched: bool = False,
    ):
        """
        Pytorch IterableDataset that streams DataChain datasets.

        See Also:
            `DataChain.to_pytorch()` - convert chain to PyTorch Dataset.

        Args:
            name (str): Name of DataChain dataset to stream.
            version (str): Version of DataChain dataset to stream.
            catalog (Catalog): DataChain catalog to which dataset belongs.
            transform (Transform): Torchvision transforms to apply to the dataset.
            tokenizer (Callable): Tokenizer to use to tokenize text values.
            tokenizer_kwargs (dict): Additional kwargs to pass when calling tokenizer.
            num_samples (int): Number of random samples to draw for each epoch.
                This argument is ignored if `num_samples=0` (the default).
        """
        self.name = name
        self.version = version
        self.transform = transform or DEFAULT_TRANSFORM
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.num_samples = num_samples
        owns_catalog = catalog is None
        if catalog is None:
            catalog = get_catalog()
        self._init_catalog(catalog)

        dc_settings = dc_settings or Settings()
        self.cache = dc_settings.cache
        if (prefetch := dc_settings.prefetch) is not None:
            self.prefetch = prefetch

        self._cache = catalog.cache
        self._prefetch_cache: Cache | None = None
        self._remove_prefetched = remove_prefetched
        if prefetch and not self.cache:
            tmp_dir = catalog.cache.tmp_dir
            assert tmp_dir
            self._prefetch_cache = get_temp_cache(tmp_dir, prefix="prefetch-")
            self._cache = self._prefetch_cache
            weakref.finalize(self, self._prefetch_cache.destroy)

        # Close the catalog if we created it - we only needed it for clone params
        if owns_catalog:
            catalog.close()

    def close(self) -> None:
        if self._prefetch_cache:
            self._prefetch_cache.destroy()

    def _init_catalog(self, catalog: "Catalog"):
        # For compatibility with multiprocessing,
        # we can only store params in __init__(), as Catalog isn't picklable
        # see https://github.com/iterative/dvcx/issues/954
        self._ms_params = catalog.metastore.clone_params()
        self._wh_params = catalog.warehouse.clone_params()
        self._catalog_params = catalog.get_init_params()
        self.catalog: Catalog | None = None

    def _get_catalog(self) -> "Catalog":
        ms_cls, ms_args, ms_kwargs = self._ms_params
        ms = ms_cls(*ms_args, **ms_kwargs)
        wh_cls, wh_args, wh_kwargs = self._wh_params
        wh = wh_cls(*wh_args, **wh_kwargs)
        catalog = Catalog(ms, wh, **self._catalog_params)
        catalog.cache = self._cache
        return catalog

    def _row_iter(
        self,
        total_rank: int,
        total_workers: int,
    ) -> Generator[tuple[Any, ...], None, None]:
        catalog = self._get_catalog()
        try:
            session = Session("PyTorch", catalog=catalog)
            ds = read_dataset(
                name=self.name, version=self.version, session=session
            ).settings(cache=self.cache, prefetch=self.prefetch)

            # remove file signals from dataset
            schema = ds.signals_schema.clone_without_file_signals()
            ds = ds.select(*schema.values.keys())

            if self.num_samples > 0:
                ds = ds.sample(self.num_samples)
            ds = ds.chunk(total_rank, total_workers)
            yield from ds.to_iter()
        finally:
            catalog.close()

    def _iter_with_prefetch(self) -> Generator[tuple[Any], None, None]:
        from datachain.lib.udf import _prefetch_inputs

        total_rank, total_workers = self.get_rank_and_workers()
        download_cb = CombinedDownloadCallback()
        if os.getenv("DATACHAIN_SHOW_PREFETCH_PROGRESS"):
            download_cb = get_download_callback(
                f"{total_rank}/{total_workers}",
                position=total_rank,
                leave=True,
            )

        rows = self._row_iter(total_rank, total_workers)
        rows = _prefetch_inputs(
            rows,
            self.prefetch,
            download_cb=download_cb,
            remove_prefetched=self._remove_prefetched,
        )

        with download_cb, closing(rows):
            yield from rows

    def __iter__(self) -> Iterator[list[Any]]:
        with closing(self._iter_with_prefetch()) as rows:
            yield from map(self._process_row, rows)

    def _process_row(self, row_features: Iterable[Any]) -> list[Any]:
        row = []
        for fr in row_features:
            if hasattr(fr, "read"):
                row.append(fr.read())  # type: ignore[unreachable]
            else:
                row.append(fr)
        # Apply transforms
        if self.transform:
            try:
                if isinstance(self.transform, v2.Transform):
                    row = self.transform(row)
                for i, val in enumerate(row):
                    if isinstance(val, Image.Image):
                        row[i] = self.transform(val)
            except ValueError:
                logger.warning("Skipping transform due to unsupported data types.")
                self.transform = None
        if self.tokenizer:
            for i, val in enumerate(row):
                if isinstance(val, str) or (
                    isinstance(val, list) and isinstance(val[0], str)
                ):
                    row[i] = convert_text(
                        val, self.tokenizer, self.tokenizer_kwargs
                    ).squeeze(0)  # type: ignore[union-attr]
        return row

    @staticmethod
    def get_rank_and_workers() -> tuple[int, int]:
        """Get combined rank and number of workers across all nodes."""
        try:
            world_rank = get_rank()
            world_size = get_world_size()
        except (RuntimeError, ValueError):
            world_rank = 0
            world_size = 1
        worker_info = get_worker_info()
        if worker_info:
            worker_rank = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_rank = 0
            num_workers = 1
        total_workers = world_size * num_workers
        total_rank = world_rank * num_workers + worker_rank
        return total_rank, total_workers
