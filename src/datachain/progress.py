from fsspec import Callback
from fsspec.callbacks import TqdmCallback
from tqdm.auto import tqdm as _tqdm


class Tqdm(_tqdm):
    """tqdm subclass that erases residual characters on close.

    When tqdm clears a bar (leave=False), it writes \\r + spaces + \\r.
    The space characters remain on the terminal line and corrupt subsequent
    output such as the Python REPL prompt or DataFrame text (#1581).
    Writing the ANSI "Erase in Line" sequence after close removes them.
    """

    def close(self):
        was_enabled = not self.disable
        super().close()
        if was_enabled:
            self.fp.write("\r\033[K")
            self.fp.flush()


# Alias for drop-in replacement of tqdm imports
tqdm = Tqdm


class CombinedDownloadCallback(Callback):
    def set_size(self, size):
        # This is a no-op to prevent fsspec's .get_file() from setting the combined
        # download size to the size of the current file.
        pass

    def increment_file_count(self, n: int = 1) -> None:
        pass


class TqdmCombinedDownloadCallback(CombinedDownloadCallback, TqdmCallback):
    def __init__(self, tqdm_kwargs=None, *args, **kwargs):
        self.files_count = 0
        super().__init__(tqdm_kwargs, *args, **kwargs)

    def increment_file_count(self, n: int = 1) -> None:
        self.files_count += n
        if self.tqdm is not None:
            self.tqdm.postfix = f"{self.files_count} files"
