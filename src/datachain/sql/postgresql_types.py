"""
PostgreSQL-specific type converter for DataChain.

Handles PostgreSQL-specific type mappings that differ from the default dialect.
"""

from datachain.sql.types import TypeConverter


class PostgreSQLTypeConverter(TypeConverter):
    """PostgreSQL-specific type converter."""

    def datetime(self):
        """PostgreSQL uses TIMESTAMP WITH TIME ZONE to preserve timezone information."""
        from sqlalchemy.dialects import postgresql

        return postgresql.TIMESTAMP(timezone=True)

    def json(self):
        """PostgreSQL uses JSONB for better performance and query capabilities."""
        from sqlalchemy.dialects import postgresql

        return postgresql.JSONB()
