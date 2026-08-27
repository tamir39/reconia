"""SQLAlchemy declarative base and shared column types.

Monetary and quantity columns use NUMERIC, never floating point (POLICY.md section 5).
Floating point rounding introduces discrepancies that the reconciliation logic would
report as genuine differences.
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, MetaData, Numeric, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Explicit naming convention so Alembic autogenerate produces stable migration names.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# Quantities carry more precision than money: unit conversions (thùng -> lon) divide.
Quantity = Numeric(20, 6)
UnitPrice = Numeric(20, 6)
Money = Numeric(20, 2)
Rate = Numeric(7, 4)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    type_annotation_map = {
        Decimal: Money,
    }


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
