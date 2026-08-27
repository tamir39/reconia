"""Load synthetic demonstration data.

All values are fabricated for demonstration purposes. No real supplier, tax code or price
appears in this file (POLICY.md section 7).

Usage:
    python -m app.seed          load the sample set
    python -m app.seed --reset  remove existing sample rows first

The loader is idempotent: running it repeatedly does not create duplicates.
"""

import argparse
import sys
from datetime import date
from decimal import Decimal

from sqlalchemy import select

from app.db import SessionLocal
from app.enums import DocStatus, DocType, Direction, SourceType
from app.models import Document, DocumentLine, Party, Product

DEMO_TAX_CODE = "0312345678"

PRODUCTS = [
    # (code, name, base_uom, size_value, size_uom, pack_qty)
    ("SP001", "Bia Sài Gòn lon 330ml", "lon", Decimal("330"), "ml", Decimal("24")),
    ("SP002", "Nước mắm Nam Ngư 500ml", "chai", Decimal("500"), "ml", Decimal("12")),
    ("SP003", "Dầu ăn Tường An 1L", "chai", Decimal("1000"), "ml", Decimal("12")),
]

# Quantities are expressed in the base unit.
# The invoice bills 1200 lon of SP001 while the goods receipt records 1152, producing a
# visible discrepancy for demonstration.
ORDER_LINES = [("SP001", 1200, 12000), ("SP002", 240, 38000), ("SP003", 120, 52000)]
RECEIPT_LINES = [("SP001", 1152, 12000), ("SP002", 240, 38000), ("SP003", 120, 52000)]
INVOICE_LINES = [("SP001", 1200, 12000), ("SP002", 240, 38000), ("SP003", 120, 52000)]

DOCS = [
    ("PO-2026-0087", None, DocType.PURCHASE_ORDER, date(2026, 8, 18), SourceType.MANUAL,
     ORDER_LINES, DocStatus.APPROVED, None, None),
    ("PNK-2026-0142", None, DocType.STOCK_IN, date(2026, 8, 21), SourceType.MANUAL,
     RECEIPT_LINES, DocStatus.APPROVED, "00098765", date(2026, 8, 20)),
    ("00098765", "1C26TAA", DocType.INVOICE, date(2026, 8, 20), SourceType.XML,
     INVOICE_LINES, DocStatus.EXTRACTED, None, None),
]


def _upsert_party(session) -> Party:
    party = session.execute(
        select(Party).where(Party.tax_code == DEMO_TAX_CODE)
    ).scalar_one_or_none()
    if party is None:
        party = Party(
            tax_code=DEMO_TAX_CODE,
            name="Công ty TNHH Thương mại Thực phẩm Minh Long",
            name_norm="cong ty tnhh thuong mai thuc pham minh long",
            address="Quận 1, Thành phố Hồ Chí Minh",
        )
        session.add(party)
        session.flush()
    return party


def _upsert_products(session) -> dict[str, Product]:
    products: dict[str, Product] = {}
    for code, name, base_uom, size_value, size_uom, pack_qty in PRODUCTS:
        product = session.execute(
            select(Product).where(Product.code == code)
        ).scalar_one_or_none()
        if product is None:
            product = Product(
                code=code,
                name=name,
                name_norm=name.lower(),
                base_uom=base_uom,
                size_value=size_value,
                size_uom=size_uom,
                pack_qty=pack_qty,
            )
            session.add(product)
            session.flush()
        products[code] = product
    return products


def _build_document(party, products, spec) -> Document:
    doc_no, serial, doc_type, doc_date, source, lines, status, ref_no, ref_date = spec
    direction = Direction.IN
    document = Document(
        doc_type=doc_type,
        direction=direction,
        party_id=party.id,
        doc_no=doc_no,
        doc_serial=serial,
        doc_date=doc_date,
        source=source,
        status=status,
        ref_doc_no=ref_no,
        ref_doc_date=ref_date,
    )

    subtotal = Decimal("0")
    for index, (code, qty, unit_price) in enumerate(lines, start=1):
        product = products[code]
        quantity = Decimal(qty)
        price = Decimal(unit_price)
        amount = (quantity * price).quantize(Decimal("0.01"))
        subtotal += amount
        document.lines.append(
            DocumentLine(
                line_no=index,
                raw_name=product.name,
                raw_uom=product.base_uom,
                supplier_item_code=code,
                product_id=product.id,
                quantity=quantity,
                base_quantity=quantity,
                unit_price=price,
                line_amount=amount,
                tax_rate=Decimal("0.08"),
                tax_amount=(amount * Decimal("0.08")).quantize(Decimal("0.01")),
            )
        )

    # Purchase orders and goods receipts carry no tax totals.
    if doc_type is DocType.INVOICE:
        tax = (subtotal * Decimal("0.08")).quantize(Decimal("0.01"))
        document.subtotal = subtotal
        document.tax_amount = tax
        document.total = subtotal + tax
    else:
        document.subtotal = subtotal
        document.total = subtotal
    return document


def load(reset: bool = False) -> int:
    session = SessionLocal()
    try:
        if reset:
            numbers = [spec[0] for spec in DOCS]
            existing = session.execute(
                select(Document).where(Document.doc_no.in_(numbers))
            ).scalars().all()
            for document in existing:
                session.delete(document)
            session.commit()

        party = _upsert_party(session)
        products = _upsert_products(session)

        created = 0
        for spec in DOCS:
            doc_no, serial, doc_type = spec[0], spec[1], spec[2]
            exists = session.execute(
                select(Document).where(
                    Document.doc_no == doc_no, Document.doc_type == doc_type
                )
            ).scalar_one_or_none()
            if exists is not None:
                continue
            session.add(_build_document(party, products, spec))
            created += 1

        session.commit()
        return created
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Load synthetic demonstration data.")
    parser.add_argument("--reset", action="store_true", help="remove sample rows first")
    args = parser.parse_args()

    created = load(reset=args.reset)
    if created:
        print(f"Loaded {created} demonstration document(s).")
    else:
        print("Demonstration data already present; nothing to do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
