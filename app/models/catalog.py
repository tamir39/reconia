"""Reference data: trading parties, the product catalogue, aliases, and UoM conversion.

Party identity is the tax code (MST) and only the tax code. Addresses are stored for
display but never used for matching: Vietnam moved to a two-tier administrative structure
on 01/07/2025, so the same supplier legitimately carries different addresses on documents
issued before and after that date.
"""

from decimal import Decimal

from sqlalchemy import ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, Quantity, TimestampMixin


class Party(Base, TimestampMixin):
    """A supplier or customer."""

    __tablename__ = "party"

    id: Mapped[int] = mapped_column(primary_key=True)
    tax_code: Mapped[str] = mapped_column(String(14), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    name_norm: Mapped[str] = mapped_column(String(255), index=True)
    address: Mapped[str | None] = mapped_column(String(500))  # display only, never matched

    aliases: Mapped[list["ProductAlias"]] = relationship(back_populates="party")


class Product(Base, TimestampMixin):
    """Internal product catalogue entry — the canonical item."""

    __tablename__ = "product"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    name_norm: Mapped[str] = mapped_column(String(255), index=True)
    base_uom: Mapped[str] = mapped_column(String(32))

    # Attributes extracted from the name, used to gate fuzzy matches.
    # "Bia Sài Gòn 330ml" vs "Bia Sài Gòn 500ml" are textually near-identical and
    # commercially different: these fields must match exactly before fuzzy score counts.
    size_value: Mapped[Decimal | None] = mapped_column(Quantity)
    size_uom: Mapped[str | None] = mapped_column(String(32))
    pack_qty: Mapped[Decimal | None] = mapped_column(Quantity)


class ProductAlias(Base, TimestampMixin):
    """Correction memory: a supplier's own wording mapped to our catalogue.

    Written back every time a human confirms a mapping in the review screen. This is
    retrieval, not machine learning — no model is retrained.
    """

    __tablename__ = "product_alias"
    __table_args__ = (
        UniqueConstraint("party_id", "alias_norm", name="uq_product_alias_party_alias"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    party_id: Mapped[int | None] = mapped_column(ForeignKey("party.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("product.id"), index=True)
    alias_raw: Mapped[str] = mapped_column(String(255))
    alias_norm: Mapped[str] = mapped_column(String(255), index=True)
    confirmed_by: Mapped[str | None] = mapped_column(String(128))

    party: Mapped["Party | None"] = relationship(back_populates="aliases")
    product: Mapped["Product"] = relationship()


class UomConversion(Base, TimestampMixin):
    """Unit conversion, e.g. 1 thùng = 24 lon.

    product_id NULL means a global rule (kg -> g). A product-specific row wins, because
    "1 thùng" is not the same count for every product.

    Converting quantity by `factor` requires dividing unit price by the same factor, or
    line totals stop reconciling.
    """

    __tablename__ = "uom_conversion"
    __table_args__ = (
        UniqueConstraint(
            "product_id", "from_uom", "to_uom", name="uq_uom_conversion_product_from_to"
        ),
        Index("ix_uom_conversion_lookup", "from_uom", "to_uom"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int | None] = mapped_column(ForeignKey("product.id"), index=True)
    from_uom: Mapped[str] = mapped_column(String(32))
    to_uom: Mapped[str] = mapped_column(String(32))
    factor: Mapped[Decimal] = mapped_column(Quantity)  # 1 from_uom = factor x to_uom
