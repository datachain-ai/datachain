from typing import TYPE_CHECKING

import sqlalchemy as sa

if TYPE_CHECKING:
    from datachain.data_storage.db_engine import DatabaseEngine


def drop_all_tables(db: "DatabaseEngine") -> None:
    """Drop every table in the test database behind ``db``.

    For server backends whose engine URL names the database; SQLite engines
    connect through a creator and have no URL database to check.
    """
    database = db.engine.url.database or ""
    if "test" not in database:
        raise RuntimeError(
            f"Refusing to wipe database {database!r}: the name must contain 'test'"
        )
    metadata = sa.MetaData()
    metadata.reflect(bind=db.engine)
    quote = db.dialect.identifier_preparer.quote
    with db.engine.begin() as conn:
        for table in reversed(metadata.sorted_tables):
            conn.execute(sa.text(f"DROP TABLE IF EXISTS {quote(table.name)}"))
