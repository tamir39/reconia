"""Reconciliation batches, results, rule findings, and the audit trail.

A result row is the unit the user sees. It records three quantities for the same product —
ordered, physically moved, invoiced — and the outcome computed from the deltas between
them. The `reason` string is what makes a flagged invoice actionable for an accountant;
a bare confidence score is not.
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Date,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.enums import Direction, MatchTier, Outcome, Severity
from app.models.base import Base, Money, Quantity, Rate, TimestampMixin


class ReconciliationBatch(Base, TimestampMixin):
    """One reconciliation run over a (period, party, direction) slice."""

    __tablename__ = "reconciliation_batch"

    id: Mapped[int] = mapped_column(primary_key=True)
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    party_id: Mapped[int | None] = mapped_column(ForeignKey("party.id"), index=True)
    direction: Mapped[Direction] = mapped_column(SAEnum(Direction, native_enum=False))

    results: Mapped[list["ReconciliationResult"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


class ReconciliationResult(Base):
    __tablename__ = "reconciliation_result"
    __table_args__ = (Index("ix_recon_result_batch_outcome", "batch_id", "outcome"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("reconciliation_batch.id", ondelete="CASCADE"), index=True
    )

    # Any of these may be NULL — a NULL counterpart is exactly what MISSING means.
    invoice_id: Mapped[int | None] = mapped_column(ForeignKey("document.id"), index=True)
    stock_doc_id: Mapped[int | None] = mapped_column(ForeignKey("document.id"), index=True)
    po_id: Mapped[int | None] = mapped_column(ForeignKey("document.id"), index=True)
    product_id: Mapped[int | None] = mapped_column(ForeignKey("product.id"), index=True)

    # Three-way quantities, all normalised to the product's base UoM.
    qty_ordered: Mapped[Decimal | None] = mapped_column(Quantity)
    qty_moved: Mapped[Decimal | None] = mapped_column(Quantity)  # received or issued
    qty_invoiced: Mapped[Decimal | None] = mapped_column(Quantity)
    amount_diff: Mapped[Decimal | None] = mapped_column(Money)

    outcome: Mapped[Outcome] = mapped_column(SAEnum(Outcome, native_enum=False), index=True)
    match_tier: Mapped[MatchTier | None] = mapped_column(SAEnum(MatchTier, native_enum=False))
    confidence: Mapped[Decimal | None] = mapped_column(Rate)

    # Human-readable justification, e.g.
    # "Hóa đơn ghi 120, phiếu nhập ghi 100 — thừa 20 (vượt dung sai 2%)".
    reason: Mapped[str | None] = mapped_column(Text)

    # Set when a reviewer overrides the computed outcome. The original is never destroyed.
    reviewed_outcome: Mapped[Outcome | None] = mapped_column(
        SAEnum(Outcome, native_enum=False)
    )
    reviewed_by: Mapped[str | None] = mapped_column(String(128))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    batch: Mapped["ReconciliationBatch"] = relationship(back_populates="results")
    findings: Mapped[list["Finding"]] = relationship(back_populates="result")

    @property
    def effective_outcome(self) -> Outcome:
        """What the reports count: a human override if present, else the computed value."""
        return self.reviewed_outcome or self.outcome


class Finding(Base):
    """One rule violation. Every validation rule is a pure function writing rows here."""

    __tablename__ = "finding"
    __table_args__ = (Index("ix_finding_rule", "rule_code", "severity"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), index=True
    )
    result_id: Mapped[int | None] = mapped_column(
        ForeignKey("reconciliation_result.id", ondelete="CASCADE"), index=True
    )

    rule_code: Mapped[str] = mapped_column(String(64))  # e.g. "LINE_ARITHMETIC"
    severity: Mapped[Severity] = mapped_column(SAEnum(Severity, native_enum=False))
    message: Mapped[str] = mapped_column(Text)  # Vietnamese, shown to the user
    details: Mapped[dict | None] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    result: Mapped["ReconciliationResult | None"] = relationship(back_populates="findings")


class AuditEvent(Base):
    """Append-only trail: what the agent did, what the user did, and why.

    Module 6 reads this. It is also what lets the report claim traceability, so nothing
    should ever update or delete a row here.
    """

    __tablename__ = "audit_event"
    __table_args__ = (Index("ix_audit_entity", "entity_type", "entity_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(64))  # "document", "result", ...
    entity_id: Mapped[int | None] = mapped_column(Integer)

    action: Mapped[str] = mapped_column(String(64))  # "extract", "match", "approve", ...
    actor: Mapped[str] = mapped_column(String(128))  # username, or "agent"
    tool_name: Mapped[str | None] = mapped_column(String(64))  # which tool the agent called

    input_summary: Mapped[str | None] = mapped_column(Text)
    output_summary: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[Decimal | None] = mapped_column(Rate)
    reason: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
