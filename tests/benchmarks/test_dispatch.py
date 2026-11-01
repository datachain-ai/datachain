"""Measure task SQL and reads separately from process startup and model inference."""

import pytest
import sqlalchemy as sa

from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.query.dispatch import guided_tasks


@pytest.mark.parametrize("width", [1, 129])
@pytest.mark.parametrize("scheduling", ["fixed", "guided"])
def test_cheap_udf_input_reads(benchmark, sqlite_db, width, scheduling):
    warehouse = SQLiteWarehouse(sqlite_db)
    table = sa.Table(
        "task_reads",
        sqlite_db.metadata,
        sa.Column("sys__id", sa.Integer, primary_key=True),
        *(sa.Column(f"value_{i}", sa.Integer) for i in range(width - 1)),
    )
    sqlite_db.create_table(table)
    ids = list(range(1, 1001))
    warehouse.insert_rows(table, [{column.name: i for column in table.c} for i in ids])
    warehouse.insert_rows_done(table)
    query = table.select()
    tasks = (
        list(guided_tasks(ids, len(ids), n_workers=4, max_size=10_000))
        if scheduling == "guided"
        else [ids[i : i + 250] for i in range(0, len(ids), 250)]
    )
    sql_sizes = [
        len(
            sqlite_db.compile_to_args(query.where(table.c.sys__id.in_(task)))[
                0
            ].encode()
        )
        for task in tasks
    ]
    benchmark.extra_info.update(
        rows=len(ids),
        selected_columns=width,
        statements=len(tasks),
        total_sql_bytes=sum(sql_sizes),
        maximum_sql_bytes=max(sql_sizes),
        distinct_in_sizes=len({len(task) for task in tasks}),
    )

    def run():
        return sum(
            row[0] + 1
            for task in tasks
            for row in warehouse.dataset_rows_select_from_ids(query, task, False)
        )

    assert benchmark(run) == sum(i + 1 for i in ids)
