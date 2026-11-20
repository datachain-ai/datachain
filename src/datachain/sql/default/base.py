from datachain.sql.types import (
    DBDefaults,
    TypeConverter,
    TypeDefaults,
    register_backend_types,
    register_db_defaults,
    register_type_defaults,
)

setup_is_complete: bool = False


def setup() -> None:
    global setup_is_complete  # noqa: PLW0603
    if setup_is_complete:
        return

    register_backend_types("default", TypeConverter())
    register_type_defaults("default", TypeDefaults())
    register_db_defaults("default", DBDefaults())

    setup_is_complete = True
