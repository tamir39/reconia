"""Document list route, exercised against an in-memory database.

Uses SQLite so the test suite runs without Docker. Production is PostgreSQL; anything
dialect-specific must be covered by a separate integration test.
"""

from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_session
from app.enums import DocStatus, DocType, Direction, SourceType
from app.main import app
from app.models import Base, Document, DocumentLine, Party


@pytest.fixture
def session():
    # TestClient serves requests on a different thread than the fixture, so the
    # connection must be shared and thread-check disabled.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False)
    with maker() as s:
        yield s
    Base.metadata.drop_all(engine)


@pytest.fixture
def client(session):
    app.dependency_overrides[get_session] = lambda: session
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def seeded(session):
    party = Party(tax_code="0301234567", name="Công ty TNHH ABC", name_norm="cong ty tnhh abc")
    session.add(party)
    session.flush()

    invoice = Document(
        doc_type=DocType.INVOICE,
        direction=Direction.IN,
        party_id=party.id,
        doc_no="00012345",
        doc_serial="1C26TAA",
        doc_date=date(2026, 8, 20),
        source=SourceType.XML,
        subtotal=Decimal("1000000.00"),
        tax_amount=Decimal("80000.00"),
        total=Decimal("1080000.00"),
        status=DocStatus.EXTRACTED,
    )
    invoice.lines.append(
        DocumentLine(
            line_no=1,
            raw_name="Bia Sài Gòn lon 330ml",
            raw_uom="thùng",
            quantity=Decimal("10"),
            unit_price=Decimal("100000"),
            line_amount=Decimal("1000000.00"),
            tax_rate=Decimal("0.08"),
        )
    )
    stock_in = Document(
        doc_type=DocType.STOCK_IN,
        direction=Direction.IN,
        party_id=party.id,
        doc_no="PNK-0091",
        doc_date=date(2026, 8, 21),
        source=SourceType.MANUAL,
        status=DocStatus.NEW,
    )
    session.add_all([invoice, stock_in])
    session.commit()
    return invoice, stock_in


def test_empty_state(client):
    response = client.get("/documents")
    assert response.status_code == 200
    assert "Chưa có chứng từ nào" in response.text


def test_lists_documents(client, seeded):
    response = client.get("/documents")
    assert response.status_code == 200
    assert "00012345" in response.text
    assert "PNK-0091" in response.text
    assert "Hóa đơn" in response.text
    assert "Phiếu nhập kho" in response.text
    assert "1,080,000" in response.text  # money formatted, not raw Decimal


def test_filter_by_doc_type(client, seeded):
    response = client.get("/documents", params={"doc_type": DocType.INVOICE.value})
    assert response.status_code == 200
    assert "00012345" in response.text
    assert "PNK-0091" not in response.text


def test_search_by_doc_no(client, seeded):
    response = client.get("/documents", params={"q": "PNK"})
    assert response.status_code == 200
    assert "PNK-0091" in response.text
    assert "00012345" not in response.text


def test_money_columns_are_decimal_not_float(client, seeded, session):
    """POLICY.md section 5: money must not round-trip through float."""
    doc = session.query(Document).filter_by(doc_no="00012345").one()
    assert isinstance(doc.total, Decimal)
    assert doc.total == Decimal("1080000.00")
