"""Documents and their line items.

One table serves all four document kinds. `doc_type` says what it is and `direction` says
which way goods flow, so the matching engine runs unchanged over inbound and outbound —
that is what makes "chứng từ nhập xuất" one engine rather than two.

Duplicate detection (lặp chứng từ) is layered:
  L0  file_sha256          — the same file uploaded twice
  L1  business key         — UNIQUE(party, serial, number) enforced by the database
  L2  near-duplicate       — same party/date/total, different number (a misread digit);
                             detected in the rules engine, not by a constraint
"""

from datetime import date
from decimal import Decimal

from sqlalchemy import (
    Date,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.enums import DocStatus, DocType, Direction, SourceType
from app.models.base import Base, Money, Quantity, Rate, TimestampMixin, UnitPrice


class Document(Base, TimestampMixin):
    __tablename__ = "document"
    __table_args__ = (
        # L1 duplicate detection. Catch the IntegrityError and raise a DUPLICATE finding.
        UniqueConstraint(
            "party_id", "doc_type", "doc_serial", "doc_no", name="uq_document_business_key"
        ),
        Index("ix_document_period", "doc_date", "party_id", "direction"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    doc_type: Mapped[DocType] = mapped_column(SAEnum(DocType, native_enum=False), index=True)
    direction: Mapped[Direction] = mapped_column(SAEnum(Direction, native_enum=False), index=True)

    party_id: Mapped[int | None] = mapped_column(ForeignKey("party.id"), index=True)
    doc_no: Mapped[str | None] = mapped_column(String(64), index=True)
    doc_serial: Mapped[str | None] = mapped_column(String(32))  # ký hiệu
    doc_date: Mapped[date | None] = mapped_column(Date, index=True)

    # Where the data came from and how much to trust it.
    source: Mapped[SourceType] = mapped_column(SAEnum(SourceType, native_enum=False))
    source_path: Mapped[str | None] = mapped_column(String(500))
    file_sha256: Mapped[str | None] = mapped_column(String(64), index=True)  # L0 duplicates
    extraction_confidence: Mapped[Decimal | None] = mapped_column(Rate)

    subtotal: Mapped[Decimal | None] = mapped_column(Money)
    tax_amount: Mapped[Decimal | None] = mapped_column(Money)
    total: Mapped[Decimal | None] = mapped_column(Money)

    status: Mapped[DocStatus] = mapped_column(
        SAEnum(DocStatus, native_enum=False), default=DocStatus.NEW, index=True
    )

    # Tier-0 matching: the reference printed on a stock note ("theo hóa đơn số ... ngày ...").
    # When present this resolves a pairing outright, with no fuzzy matching at all.
    ref_doc_no: Mapped[str | None] = mapped_column(String(64), index=True)
    ref_doc_date: Mapped[date | None] = mapped_column(Date)

    note: Mapped[str | None] = mapped_column(Text)

    party: Mapped["object | None"] = relationship("Party")
    lines: Mapped[list["DocumentLine"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="DocumentLine.line_no"
    )
    evidence: Mapped[list["FieldEvidence"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentLine(Base):
    __tablename__ = "document_line"
    __table_args__ = (
        UniqueConstraint("document_id", "line_no", name="uq_document_line_doc_lineno"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), index=True
    )
    line_no: Mapped[int] = mapped_column(Integer)

    # What the document actually said, kept verbatim for evidence and alias learning.
    raw_name: Mapped[str] = mapped_column(String(255))
    raw_uom: Mapped[str | None] = mapped_column(String(32))
    supplier_item_code: Mapped[str | None] = mapped_column(String(64), index=True)

    # What we resolved it to. NULL until resolved; that itself is a review trigger.
    product_id: Mapped[int | None] = mapped_column(ForeignKey("product.id"), index=True)
    match_confidence: Mapped[Decimal | None] = mapped_column(Rate)

    quantity: Mapped[Decimal | None] = mapped_column(Quantity)
    base_quantity: Mapped[Decimal | None] = mapped_column(Quantity)  # normalised to base_uom
    unit_price: Mapped[Decimal | None] = mapped_column(UnitPrice)
    line_amount: Mapped[Decimal | None] = mapped_column(Money)
    tax_rate: Mapped[Decimal | None] = mapped_column(Rate)  # 0 / 0.05 / 0.08 / 0.10
    tax_amount: Mapped[Decimal | None] = mapped_column(Money)

    document: Mapped["Document"] = relationship(back_populates="lines")
    product: Mapped["object | None"] = relationship("Product")


class FieldEvidence(Base):
    """Per-field provenance, including the page region it was read from.

    Stored from the very first extraction commit on purpose: the side-by-side highlight
    view is a day's work if the boxes exist and a week if they have to be backfilled.
    """

    __tablename__ = "field_evidence"
    __table_args__ = (Index("ix_field_evidence_doc_field", "document_id", "field_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), index=True
    )
    line_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_line.id", ondelete="CASCADE")
    )

    field_name: Mapped[str] = mapped_column(String(64))
    raw_value: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[Decimal | None] = mapped_column(Rate)
    source: Mapped[SourceType] = mapped_column(SAEnum(SourceType, native_enum=False))

    page: Mapped[int | None] = mapped_column(Integer)
    bbox_x0: Mapped[Decimal | None] = mapped_column(Quantity)
    bbox_y0: Mapped[Decimal | None] = mapped_column(Quantity)
    bbox_x1: Mapped[Decimal | None] = mapped_column(Quantity)
    bbox_y1: Mapped[Decimal | None] = mapped_column(Quantity)

    document: Mapped["Document"] = relationship(back_populates="evidence")
