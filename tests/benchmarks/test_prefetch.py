import asyncio
from contextlib import closing

import pytest

from datachain.lib.file import File
from datachain.lib.udf import _prefetch_inputs


@pytest.mark.parametrize("files_per_unit", [1, 32])
def test_prefetch_inputs(benchmark, monkeypatch, files_per_unit):
    async def prefetch(self, download_cb=None):
        await asyncio.sleep(0)
        return True

    monkeypatch.setattr(File, "_prefetch", prefetch)
    files = [File(source="s3://bucket", path=str(i)) for i in range(2048)]
    if files_per_unit == 1:
        units = [[file] for file in files]
        batch_columns = None
    else:
        units = [
            [files[i : i + files_per_unit]]
            for i in range(0, len(files), files_per_unit)
        ]
        batch_columns = {0}

    def run():
        with closing(
            _prefetch_inputs(iter(units), 8, batch_columns=batch_columns)
        ) as rows:
            return sum(1 for _ in rows)

    assert benchmark(run) == len(units)
