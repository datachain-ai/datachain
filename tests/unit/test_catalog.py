from typing import TYPE_CHECKING

import pytest

from datachain.catalog import Catalog
from datachain.catalog.catalog import _round_robin_batch

if TYPE_CHECKING:
    from datachain.data_storage import AbstractWarehouse


def test_catalog_warehouse_ready_callback(mocker, warehouse, metastore):
    spy = mocker.spy(warehouse, "is_ready")

    def callback(warehouse: "AbstractWarehouse"):
        assert warehouse.is_ready()

    catalog = Catalog(metastore, warehouse, warehouse_ready_callback=callback)

    spy.assert_not_called()

    _ = catalog.warehouse

    spy.assert_called_once()
    spy.reset_mock()

    _ = catalog.warehouse

    spy.assert_not_called()


@pytest.mark.parametrize(
    "urls, num_workers, expected",
    [
        ([], 3, [[], [], []]),
        (["a"], 3, [["a"], [], []]),
        (["a", "b", "c"], 3, [["a"], ["b"], ["c"]]),
        (
            ["a", "b", "c", "d", "e"],
            3,
            [["a", "d"], ["b", "e"], ["c"]],
        ),
        (
            ["u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7"],
            5,
            [["u0", "u5"], ["u1", "u6"], ["u2", "u7"], ["u3"], ["u4"]],
        ),
        (["a", "b"], 1, [["a", "b"]]),
    ],
)
def test_round_robin_batch(urls, num_workers, expected):
    assert _round_robin_batch(urls, num_workers) == expected
