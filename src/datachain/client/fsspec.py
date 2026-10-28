import asyncio
import functools
import io
import logging
import multiprocessing
import os
import posixpath
import re
import shutil
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncIterator,
    Iterator,
    Sequence,
)
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, BinaryIO, ClassVar, Literal, NamedTuple
from urllib.parse import urlparse

from dvc_objects.fs.system import reflink
from fsspec.asyn import get_loop, sync
from fsspec.callbacks import DEFAULT_CALLBACK, Callback

from datachain.cache import Cache
from datachain.client.fileslice import FileWrapper
from datachain.fs.utils import is_win_local_path
from datachain.nodes_fetcher import NodesFetcher
from datachain.nodes_thread_pool import NodeChunk
from datachain.progress import tqdm

from .ranges import LIST_WORKERS, Page, RangeListing

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem

    from datachain.client.writeconfig import WriteConfig
    from datachain.dataset import StorageURI
    from datachain.lib.file import File


logger = logging.getLogger("datachain")

FETCH_WORKERS = 100
DELIMITER = "/"  # Path delimiter.

DATA_SOURCE_URI_PATTERN = re.compile(r"^[\w]+:\/\/.*$")
CLOUD_STORAGE_PROTOCOLS = {"s3", "gs", "az", "hf"}

ResultQueue = asyncio.Queue[Sequence[Any] | None]


def is_cloud_uri(uri: str) -> bool:
    protocol = urlparse(uri).scheme
    return protocol in CLOUD_STORAGE_PROTOCOLS


def get_cloud_schemes() -> list[str]:
    """Get list of cloud storage scheme prefixes."""
    return [f"{p}://" for p in CLOUD_STORAGE_PROTOCOLS]


class Bucket(NamedTuple):
    name: str
    uri: "StorageURI"
    created: datetime | None


class BucketStatus(NamedTuple):
    exists: bool
    access: Literal["anonymous", "authenticated", "denied"]
    error: str | None = None


class Client(ABC):
    MAX_THREADS = multiprocessing.cpu_count()
    FS_CLASS: ClassVar[type["AbstractFileSystem"]]
    PREFIX: ClassVar[str]
    protocol: ClassVar[str]
    # client_config keys this backend treats as credentials.
    CREDENTIAL_KEYS: ClassVar[frozenset[str]] = frozenset()
    LIST_PAGE_SIZE: ClassVar[int] = 1000

    @classmethod
    def has_explicit_credentials(cls, client_config: dict | None) -> bool:
        """True if client_config contains any backend-specific credential key."""
        if not client_config:
            return False
        return any(k in client_config for k in cls.CREDENTIAL_KEYS)

    def __init__(self, name: str, fs_kwargs: dict[str, Any], cache: Cache) -> None:
        self.name = name
        self.fs_kwargs = fs_kwargs
        self._fs: AbstractFileSystem | None = None
        self.cache = cache
        self.uri = self.storage_uri(self.name)

    @staticmethod
    def format_etag(etag: str) -> str:
        """Render an etag (e.g. for error messages).

        Default matches ``dc.show()`` / Studio (the stored string). Backends that
        encode extra meaning in the etag (local mtime) override this.
        """
        return str(etag)

    @staticmethod
    def get_implementation(url: str | os.PathLike[str]) -> type["Client"]:  # noqa: PLR0911
        from .azure import AzureClient
        from .gcs import GCSClient
        from .hf import HfClient
        from .http import HTTPClient, HTTPSClient
        from .local import FileClient
        from .s3 import ClientS3

        protocol = urlparse(os.fspath(url)).scheme

        if not protocol or is_win_local_path(os.fspath(url)):
            return FileClient
        if protocol == ClientS3.protocol:
            return ClientS3
        if protocol == GCSClient.protocol:
            return GCSClient
        if protocol == AzureClient.protocol:
            return AzureClient
        if protocol == FileClient.protocol:
            return FileClient
        if protocol == HfClient.protocol:
            return HfClient
        if protocol == HTTPClient.protocol:
            return HTTPClient
        if protocol == HTTPSClient.protocol:
            return HTTPSClient

        raise NotImplementedError(f"Unsupported protocol: {protocol}")

    @staticmethod
    def is_data_source_uri(name: str) -> bool:
        # Returns True if name is one of supported data sources URIs, e.g s3 bucket
        return DATA_SOURCE_URI_PATTERN.match(name) is not None

    @staticmethod
    def validate_file_path(path: str) -> None:
        """Validate a relative object path for this backend.

        Raises ``ValueError`` for paths that are empty, end with ``/``, or
        contain ``.`` / ``..`` segments.  Subclasses extend this with
        backend-specific rules (e.g. local-filesystem restrictions).
        """
        if not path:
            raise ValueError("path must not be empty")
        if path.endswith("/"):
            raise ValueError("path must not be a directory")
        parts = path.split("/")
        if any(part in (".", "..") for part in parts):
            raise ValueError("path must not contain '.' or '..'")

    @classmethod  # noqa: B027
    def validate_source(cls, source: str) -> None:
        """Validate the source URI for this backend."""

    @staticmethod
    def parse_url(source: str) -> tuple["StorageURI", str]:
        cls = Client.get_implementation(source)
        storage_name, rel_path = cls.split_url(source)
        return cls.storage_uri(storage_name), rel_path

    @staticmethod
    def get_client(source: str | os.PathLike[str], cache: Cache, **kwargs) -> "Client":
        cls = Client.get_implementation(source)
        storage_url, _ = cls.split_url(os.fspath(source))
        if os.name == "nt":
            storage_url = storage_url.removeprefix("/")
        cls.validate_source(os.fspath(source))
        return cls.from_name(storage_url, cache, kwargs)

    @classmethod
    def create_fs(cls, **kwargs) -> "AbstractFileSystem":
        kwargs.setdefault("version_aware", True)
        fs = cls.FS_CLASS(**kwargs)
        fs.invalidate_cache()
        return fs

    @classmethod
    def from_name(
        cls,
        name: str,
        cache: Cache,
        kwargs: dict[str, Any],
    ) -> "Client":
        return cls(name, kwargs, cache)

    @classmethod
    def from_source(
        cls,
        uri: "StorageURI",
        cache: Cache,
        **kwargs,
    ) -> "Client":
        return cls(cls.FS_CLASS._strip_protocol(uri), kwargs, cache)

    @classmethod
    def ls_buckets(cls, **kwargs) -> Iterator[Bucket]:
        from datachain.dataset import StorageURI

        for entry in cls.create_fs(**kwargs).ls(cls.PREFIX, detail=True):
            name = entry["name"].rstrip("/")
            yield Bucket(
                name=name,
                uri=StorageURI(f"{cls.PREFIX}{name}"),
                created=entry.get("CreationDate"),
            )

    @classmethod
    def bucket_status(cls, name: str, **kwargs) -> "BucketStatus":
        """Check bucket existence and access level without listing objects.

        Returns a BucketStatus with:
          - exists: whether the bucket/container exists
          - access: 'anonymous' | 'authenticated' | 'denied'

        Network errors propagate as exceptions.
        """
        raise NotImplementedError(f"bucket_status is not supported for {cls.__name__}")

    @classmethod
    def is_root_url(cls, url) -> bool:
        return url == cls.PREFIX

    @classmethod
    def storage_uri(cls, storage_name: str) -> "StorageURI":
        """Build a :class:`StorageURI` by prepending the protocol to *storage_name*."""
        from datachain.dataset import StorageURI

        return StorageURI(f"{cls.PREFIX}{storage_name}")

    @classmethod
    def split_url(cls, url: str) -> tuple[str, str]:
        """
        Splits the URL into two pieces:
        1. bucket name without protocol (everything up until the first /)
        2. path which is the rest of URL starting from bucket name
        e.g s3://my-bucket/animals/dogs -> (my-bucket, animals/dogs)
        """
        fill_path = url[len(cls.PREFIX) :]
        path_split = fill_path.split("/", 1)
        bucket = path_split[0]
        path = path_split[1] if len(path_split) > 1 else ""
        return bucket, path

    @property
    def fs(self) -> "AbstractFileSystem":
        if not self._fs:
            self._fs = self.create_fs(**self.fs_kwargs)
        return self._fs

    def url(
        self,
        path: str,
        expires: int = 3600,
        version_id: str | None = None,
        **kwargs,
    ) -> str:
        self.validate_file_path(path)
        kwargs.update(self._version_kwargs(version_id))
        return self.fs.sign(self.get_uri(path), expiration=expires, **kwargs)

    async def get_current_etag(self, file: "File") -> str:
        full_path = file.get_fs_path()
        info = await self.fs._info(full_path, **self._file_info_kwargs(file.version))
        return self.info_to_file(info, file.path).etag

    def get_file_info(self, path: str, version_id: str | None = None) -> "File":
        self.validate_file_path(path)
        full_path = self.get_uri(path)
        info = sync(
            get_loop(),
            self.fs._info,
            full_path,
            **self._file_info_kwargs(version_id),
        )
        return self.info_to_file(info, path)

    async def get_size(self, file: "File") -> int:
        full_path = file.get_fs_path()
        info = await self.fs._info(full_path, **self._version_kwargs(file.version))
        size = info.get("size")
        if size is None:
            raise FileNotFoundError(full_path)
        return int(size)

    async def get_file(self, lpath, rpath, callback, version_id: str | None = None):
        return await self.fs._get_file(
            lpath,
            rpath,
            callback=callback,
            **self._version_kwargs(version_id),
        )

    async def scandir(
        self, start_prefix: str, method: str = "default"
    ) -> AsyncIterator[Sequence[Any]]:
        """Pages of entries: ``FileRow`` rows from a range listing, ``File``
        objects otherwise."""
        try:
            impl = getattr(self, f"_fetch_{method}")
        except AttributeError:
            raise ValueError(f"Unknown indexing method '{method}'") from None
        result_queue: ResultQueue = asyncio.Queue(2 * LIST_WORKERS)

        async def listing() -> None:
            try:
                await impl(start_prefix, result_queue)
            except BaseException as e:
                # the reader stops at None, unless it cancelled this task itself
                task = asyncio.current_task()
                if not (isinstance(e, asyncio.CancelledError) and task.cancelling()):  # type: ignore[union-attr]
                    await result_queue.put(None)
                raise
            await result_queue.put(None)

        main_task = get_loop().create_task(listing())
        try:
            while (page := await result_queue.get()) is not None:
                yield page
            await main_task
        finally:
            main_task.cancel()
            await asyncio.gather(main_task, return_exceptions=True)

    async def _fetch_nested(self, start_prefix: str, result_queue: ResultQueue) -> None:
        progress_bar = tqdm(desc=f"Listing {self.uri}", unit=" objects", leave=False)
        loop = get_loop()

        queue: asyncio.Queue[str] = asyncio.Queue()
        queue.put_nowait(start_prefix)

        async def process(queue) -> None:
            while True:
                prefix = await queue.get()
                try:
                    subdirs = await self._fetch_dir(prefix, progress_bar, result_queue)
                    for subdir in subdirs:
                        queue.put_nowait(subdir)
                except Exception:
                    while not queue.empty():
                        queue.get_nowait()
                        queue.task_done()
                    raise

                finally:
                    queue.task_done()

        workers: list[asyncio.Task] = []
        try:
            workers = [loop.create_task(process(queue)) for _ in range(FETCH_WORKERS)]
            await queue.join()
            for worker in workers:
                if worker.done() and (exc := worker.exception()):
                    raise exc
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            progress_bar.close()

    async def _fetch_default(
        self, start_prefix: str, result_queue: ResultQueue
    ) -> None:
        await self._fetch_nested(start_prefix, result_queue)

    async def _fetch_dir(
        self, prefix: str, pbar, result_queue: ResultQueue
    ) -> set[str]:
        path = f"{self.name}/{prefix}"
        infos = await self.ls_dir(path)
        files = []
        subdirs = set()
        for info in infos:
            full_path = info["name"]
            subprefix = self.rel_path(full_path)
            if prefix.strip(DELIMITER) == subprefix.strip(DELIMITER):
                continue
            if info["type"] == "directory":
                subdirs.add(subprefix)
            else:
                files.append(self.info_to_file(info, subprefix))
        if files:
            await result_queue.put(files)
        found_count = len(subdirs) + len(files)
        pbar.update(found_count)
        return subdirs

    @staticmethod
    def _is_valid_key(key: str) -> bool:
        """
        Check if the key should be indexed.

        Folder markers (``a/``) and keys with a leading ``/`` are skipped
        (see #2040); keys with ``//`` inside (``a//b``) are kept.
        """
        return not (key.startswith("/") or key.endswith("/"))

    def _is_version_aware(self) -> bool:
        return getattr(self.fs, "version_aware", False)

    def _version_kwargs(self, version_id: str | None) -> dict[str, Any]:
        if version_id:
            return {"version_id": version_id}
        return {}

    @staticmethod
    def _block_size_kwargs(block_size: int | None) -> dict[str, Any]:
        # Hugging Face opens a stream that can't seek for block_size=0.
        if block_size == 0:
            return {"cache_type": "none"}
        return {"block_size": block_size}

    def _file_info_kwargs(self, version_id: str | None = None) -> dict[str, Any]:
        return self._version_kwargs(version_id)

    def _ls_dir_kwargs(self) -> dict[str, Any]:
        if self._is_version_aware():
            return {"versions": True}
        return {}

    async def ls_dir(self, path):
        return await self.fs._ls(path, detail=True, **self._ls_dir_kwargs())

    def rel_path(self, path: str) -> str:
        return self.fs.split_path(path)[1]

    def get_uri(self, rel_path: str) -> str:
        """Build a full URI for the given relative path within this client's storage."""
        return f"{self.PREFIX}{self.name}/{rel_path}"

    @abstractmethod
    def info_to_file(self, v: dict[str, Any], path: str) -> "File": ...

    def fetch_nodes(
        self,
        nodes,
        shared_progress_bar=None,
    ) -> None:
        fetcher = NodesFetcher(self, self.MAX_THREADS, self.cache)
        chunk_gen = NodeChunk(self.cache, self.uri, nodes)
        fetcher.run(chunk_gen, shared_progress_bar)

    def instantiate_object(
        self,
        file: "File",
        dst: str,
        progress_bar: tqdm,
        force: bool = False,
    ) -> None:
        if os.path.exists(dst):
            if force:
                os.remove(dst)
            else:
                progress_bar.close()
                raise FileExistsError(f"Path {dst} already exists")
        self.do_instantiate_object(file, dst)

    def do_instantiate_object(self, file: "File", dst: str) -> None:
        src = self.cache.get_path(file)
        assert src is not None

        try:
            reflink(src, dst)
        except OSError:
            # Default to copy if reflinks are not supported
            shutil.copy2(src, dst)

    def open_object(
        self,
        file: "File",
        use_cache: bool = True,
        cb: Callback = DEFAULT_CALLBACK,
        block_size: int | None = None,
    ) -> BinaryIO:
        """Open a file, including files in tar archives.

        Each read fetches at least `block_size` bytes (by default 50 MiB on S3),
        which suits reading a file in order. With `block_size=0`, each read
        fetches only the bytes asked for, for callers like pyarrow that read a
        few ranges.
        """
        if use_cache and (cache_path := self.cache.get_path(file)):
            return open(cache_path, mode="rb")
        assert not file.location
        kwargs = self._version_kwargs(file.version)
        kwargs |= self._block_size_kwargs(block_size)
        full_path = file.get_fs_path()
        return FileWrapper(
            self.fs.open(full_path, **kwargs),
            cb,
        )  # type: ignore[return-value]

    def upload(
        self,
        data: bytes | bytearray | memoryview | BinaryIO,
        path: str,
        *,
        write_config: "WriteConfig | None" = None,
    ) -> "File":
        """Upload *data* to *path*.

        ``data`` may be:
          - a bytes-like object (``bytes``/``bytearray``/``memoryview``) —
            written in a single ``pipe_file`` call, unless the backend opts out
            of piping (``_can_pipe_upload`` is False) or overrides
            ``_write_object`` (e.g. Azure, which uploads via the SDK).
          - a binary readable stream — copied into the destination via
            ``fs.open(path, 'wb')`` using ``shutil.copyfileobj``. The
            destination handle (an ``AbstractBufferedFile`` on cloud
            backends) flushes a multipart part every ``self.blocksize``
            bytes (e.g. 5 MiB on s3fs), so peak RAM is bounded by the
            backend's part size rather than the file size.

        ``write_config`` carries normalized object metadata (content type,
        content disposition, custom metadata, …) mapped to the backend's
        native write parameters.
        """
        from datachain.client.writeconfig import WriteConfig

        cfg = write_config or WriteConfig()

        if path.startswith(self.PREFIX):
            full_path = path
            _, rel_path = self.split_url(path)
        else:
            rel_path = path
            full_path = self.get_uri(path)

        self.validate_file_path(rel_path)

        parent = posixpath.dirname(full_path)
        self.fs.makedirs(parent, exist_ok=True)

        version = self._write_object(full_path, data, cfg)
        file_info = self.fs.info(full_path, **self._file_info_kwargs(version))
        return self.info_to_file(file_info, rel_path)

    def _write_object(
        self,
        full_path: str,
        data: "bytes | bytearray | memoryview | BinaryIO",
        cfg: "WriteConfig",
    ) -> str | None:
        """Write *data* to *full_path*, applying *cfg*.

        Returns a version id when the write itself produced one, else ``None``
        (the caller then reads the latest). Backends that set metadata through a
        different API (Azure) override this.
        """
        if isinstance(data, (bytes, bytearray, memoryview)) and self._can_pipe_upload():
            self.fs.pipe_file(
                full_path, data, **self._write_kwargs(cfg, streaming=False)
            )
        else:
            src: BinaryIO = (
                io.BytesIO(data)
                if isinstance(data, (bytes, bytearray, memoryview))
                else data
            )
            with self.fs.open(
                full_path, "wb", **self._write_kwargs(cfg, streaming=True)
            ) as dst:
                # Use a larger copy buffer than shutil's 16 KiB default to
                # reduce per-chunk overhead on multi-GB uploads. The
                # destination's blocksize (when exposed by fsspec backends)
                # matches the multipart part size it will flush at, making it
                # a natural choice; fall back to 8 MiB otherwise.
                buf_size = getattr(dst, "blocksize", None) or 8 * 1024 * 1024
                shutil.copyfileobj(src, dst, length=buf_size)
        return None

    @contextmanager
    def open_for_write(
        self, full_path: str, fs_mode: str, cfg: "WriteConfig", binary_kwargs: dict
    ) -> "Iterator[Any]":
        """Yield a writable handle for a write-mode ``File.open``.

        Consumed by :meth:`datachain.lib.file.File.open` in write modes (it is
        the streaming counterpart of :meth:`_write_object`, which handles the
        whole-payload ``upload``). The handle carries the written version on
        ``version_id`` after close; backends that set metadata through a
        different API (Azure) override this.
        """
        kwargs = {**binary_kwargs, **self._write_kwargs(cfg, streaming=True)}
        with self.fs.open(full_path, fs_mode, **kwargs) as handle:
            yield handle

    def _can_pipe_upload(self) -> bool:
        return True

    @staticmethod
    def _write_kwargs(cfg: "WriteConfig", *, streaming: bool) -> dict[str, Any]:
        """Map a :class:`WriteConfig` to native ``fs.open``/``pipe_file`` kwargs.

        Backends with metadata support override this; the base maps nothing and
        rejects the raw ``write_options`` escape hatch, which only S3 forwards.
        """
        cfg.reject_write_options()
        return {}

    def download(self, file: "File", *, callback: Callback = DEFAULT_CALLBACK) -> None:
        sync(get_loop(), functools.partial(self._download, file, callback=callback))

    async def _download(self, file: "File", *, callback: "Callback" = None) -> None:
        if self.cache.contains(file):
            # Already in cache, so there's nothing to do.
            return
        await self._put_in_cache(file, callback=callback)

    def put_in_cache(self, file: "File", *, callback: "Callback" = None) -> None:
        sync(get_loop(), functools.partial(self._put_in_cache, file, callback=callback))

    async def _put_in_cache(self, file: "File", *, callback: "Callback" = None) -> None:
        assert not file.location
        await self._check_unchanged(file)
        await self.cache.download(file, self, callback=callback)

    def check_unchanged(self, file: "File") -> None:
        """Fail if the file changed on the source since the catalog listed it."""
        sync(get_loop(), self._check_unchanged, file)

    async def _check_unchanged(self, file: "File") -> None:
        if file.etag:
            etag = await self.get_current_etag(file)
            if file.etag != etag:
                raise FileNotFoundError(
                    f"{file.source}/{file.path} changed on the source since the "
                    f"catalog was created (etag was {self.format_etag(file.etag)}, now "
                    f"{self.format_etag(etag)}). "
                    "Re-run the original dc.read_storage(...) call with "
                    "update=True to refresh the catalog."
                )


class RangeListedClient(Client):
    """A client that lists in parallel key ranges (``RangeListing``)."""

    LISTS_LEVELS: ClassVar[bool] = False  # whether _list_page takes a delimiter

    @staticmethod
    def key_order(key: str) -> Any:
        """The sort key of the order the server lists keys in."""
        return key

    async def _fetch_ranges(self, start_prefix: str, result_queue: ResultQueue) -> None:
        prefix = start_prefix.lstrip(DELIMITER) + DELIMITER if start_prefix else ""
        await self._fetch_prefix(prefix, result_queue)

    _fetch_default = _fetch_ranges

    async def _fetch_prefix(self, start_prefix: str, result_queue: ResultQueue) -> None:
        """Keys that start with ``start_prefix``, which needn't end a folder."""
        from .listing_worker import LIST_PROCESSES, Pool

        prefix = start_prefix.lstrip(DELIMITER)
        pool = Pool(self, prefix, LIST_PROCESSES) if LIST_PROCESSES else None
        await RangeListing(self, prefix, result_queue, pool).run()

    async def _fetch_level(self, start_prefix: str, result_queue: ResultQueue) -> None:
        """Keys under ``start_prefix`` with no delimiter past it, in parallel key
        ranges: nothing below the level is listed. No child processes: a level
        has few rows to parse."""
        prefix = start_prefix.lstrip(DELIMITER)
        await RangeListing(self, prefix, result_queue, None, DELIMITER).run()

    async def has_keys(self, prefix: str) -> bool:
        page = await self._list_page(prefix, "", None, page_size=1)
        return page.last is not None or page.token is not None

    def _lists_in_key_order(self) -> bool:
        return True

    async def _list_page(
        self,
        prefix: str,
        start_after: str,
        token: Any,
        hi: str | None = None,
        page_size: int | None = None,
        delimiter: str = "",
    ) -> Page:
        """One page under ``prefix``, in key order: the one after ``token``, or
        the first after ``start_after``. Keys past ``hi`` make no rows; a backend
        that can end a listing at a key ends it after ``hi``. With ``delimiter``
        (``LISTS_LEVELS``), keys with it past ``prefix`` are skipped."""
        raise NotImplementedError

    async def _probe(self, prefix: str, start_after: str) -> str | None:
        """The first key after ``start_after``, of any kind; None at the end."""
        token = None
        while True:
            page = await self._list_page(prefix, start_after, token, page_size=1)
            if page.last is not None or page.token is None:
                return page.last
            if page.token == token:
                raise RuntimeError(
                    f"Listing {self.uri}: the server returned the page token it was"
                    " given, so the listing would never end."
                )
            token = page.token
