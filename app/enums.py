"""Domain enums.

The reconciliation outcome values are fixed by the registered project description and
must not be renamed:
    trùng khớp / trùng khớp một phần / sai hoàn toàn / thiếu chứng từ / lặp chứng từ
"""

from enum import Enum


class DocType(str, Enum):
    """Document kinds handled by the system."""

    INVOICE = "INVOICE"  # hóa đơn
    STOCK_IN = "STOCK_IN"  # phiếu nhập kho
    STOCK_OUT = "STOCK_OUT"  # phiếu xuất kho
    PURCHASE_ORDER = "PURCHASE_ORDER"  # đơn đặt hàng


class Direction(str, Enum):
    """Inbound (buying) or outbound (selling) flow.

    A single matching engine serves both; only this field differs.
    """

    IN = "IN"  # nhập — mua vào
    OUT = "OUT"  # xuất — bán ra


class SourceType(str, Enum):
    """How a document's data was obtained.

    Ordered by trust. Never run OCR on a document that has XML or a usable text layer.
    """

    XML = "XML"  # hóa đơn điện tử — exact
    PDF_TEXT = "PDF_TEXT"  # PDF text layer — exact characters
    OCR = "OCR"  # scanned image — lossy
    MANUAL = "MANUAL"  # typed by a human


class DocStatus(str, Enum):
    NEW = "NEW"
    EXTRACTED = "EXTRACTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    APPROVED = "APPROVED"


class Outcome(str, Enum):
    """The five reconciliation outcomes required by the project description.

    Computed from deltas and coverage — never from a similarity score. Similarity decides
    whether two lines are comparable; deltas decide the verdict.
    """

    MATCHED = "MATCHED"  # trùng khớp
    PARTIAL = "PARTIAL"  # trùng khớp một phần — under-delivery / under-invoicing
    MISMATCH = "MISMATCH"  # sai hoàn toàn — over-invoicing, price violation, arithmetic failure
    MISSING = "MISSING"  # thiếu chứng từ — no counterpart document
    DUPLICATE = "DUPLICATE"  # lặp chứng từ


OUTCOME_LABEL_VI: dict[Outcome, str] = {
    Outcome.MATCHED: "Trùng khớp",
    Outcome.PARTIAL: "Trùng khớp một phần",
    Outcome.MISMATCH: "Sai hoàn toàn",
    Outcome.MISSING: "Thiếu chứng từ",
    Outcome.DUPLICATE: "Lặp chứng từ",
}

DOC_TYPE_LABEL_VI: dict[DocType, str] = {
    DocType.INVOICE: "Hóa đơn",
    DocType.STOCK_IN: "Phiếu nhập kho",
    DocType.STOCK_OUT: "Phiếu xuất kho",
    DocType.PURCHASE_ORDER: "Đơn đặt hàng",
}


class MatchTier(int, Enum):
    """Which layer resolved a pairing. Recorded as evidence."""

    EXPLICIT_REFERENCE = 0  # printed "theo hóa đơn số ... ngày ..." on the stock note
    HEADER_BLOCKING = 1  # same tax code + date window
    LINE_AGGREGATE = 2  # aggregated by (product, base uom)


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class UserRole(str, Enum):
    CLERK = "CLERK"  # nhân viên nhập liệu
    APPROVER = "APPROVER"  # kế toán duyệt
