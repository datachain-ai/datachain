from sqlalchemy.dialects import postgresql

from datachain.sql.postgresql_types import PostgreSQLTypeConverter


def test_datetime_is_timezone_aware_timestamp():
    result = PostgreSQLTypeConverter().datetime()

    assert isinstance(result, postgresql.TIMESTAMP)
    assert result.timezone is True


def test_json_is_jsonb():
    assert isinstance(PostgreSQLTypeConverter().json(), postgresql.JSONB)
