"""Tests for unified dataset adapters (DocILE, CORD-v2, SROIE)."""

import pytest

from app.data import CordAdapter, DocileAdapter, SroieAdapter
from app.graph.builder import build_graph, collect_fieldtypes
from app.graph.labels import build_label_vocabulary, label_document


def available_adapter(adapter_class):
    """External dataset tests skip only when that dataset is absent."""
    try:
        return adapter_class()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_docile_adapter_splits_and_load():
    adapter = available_adapter(DocileAdapter)
    splits = adapter.splits()
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    assert adapter.has_ground_truth("train") is True
    assert adapter.has_ground_truth("val") is True
    assert adapter.has_ground_truth("test") is True
    assert adapter.has_ground_truth("test_official") is False

    test_ids = adapter.document_ids("test")
    train_ids = adapter.document_ids("train")
    assert len(test_ids) == 500
    assert len(train_ids) == 4680
    assert len(set(train_ids) & set(test_ids)) == 0
    test_doc = adapter.load(test_ids[0])
    assert test_doc.source == "docile"
    assert len(test_doc.words) > 0
    assert len(test_doc.fields) > 0

    if "test_official" in splits:
        official_ids = adapter.document_ids("test_official")
        assert len(official_ids) == 1000
        official_doc = adapter.load(official_ids[0])
        assert official_doc.source == "docile"
        assert len(official_doc.words) > 0
        assert len(official_doc.fields) == 0

    val_ids = adapter.document_ids("val")
    assert len(val_ids) == 500
    doc = adapter.load(val_ids[0])

    assert doc.source == "docile"
    assert len(doc.words) > 0
    assert len(doc.fields) > 0

    # Ensure all word bounding boxes are normalized to [0, 1]
    for w in doc.words:
        assert 0.0 <= w.bbox.x0 <= w.bbox.x1 <= 1.0
        assert 0.0 <= w.bbox.y0 <= w.bbox.y1 <= 1.0


def test_cord_adapter_splits_and_load():
    adapter = available_adapter(CordAdapter)
    splits = adapter.splits()
    assert set(splits) == {"train", "validation", "test"}
    assert len(adapter.document_ids("train")) == 800
    assert len(adapter.document_ids("validation")) == 100
    assert len(adapter.document_ids("test")) == 100

    doc = adapter.load("train_0")
    assert doc.source == "cord"
    assert len(doc.words) > 0
    assert len(doc.fields) > 0

    # Test that line items have line_item_id assigned
    menu_fields = [f for f in doc.fields if f.fieldtype.startswith("menu.")]
    assert len(menu_fields) > 0
    assert any(f.line_item_id is not None for f in menu_fields)

    # Check normalized bounding boxes
    for w in doc.words:
        assert 0.0 <= w.bbox.x0 <= w.bbox.x1 <= 1.0
        assert 0.0 <= w.bbox.y0 <= w.bbox.y1 <= 1.0


def test_sroie_adapter_splits_and_load():
    adapter = available_adapter(SroieAdapter)
    splits = adapter.splits()
    assert set(splits) == {"train", "val", "test"}

    train_ids = adapter.document_ids("train")
    val_ids = adapter.document_ids("val")
    test_ids = adapter.document_ids("test")

    assert len(train_ids) == 500
    assert len(val_ids) == 63
    assert len(test_ids) == 63
    assert len(set(train_ids) | set(val_ids) | set(test_ids)) == 626

    # Verify that splits are unbiased and non-sequential (i.e. not simply 000..499)
    all_sorted = sorted(train_ids + val_ids + test_ids)
    assert train_ids != all_sorted[:500]
    assert "001" in val_ids  # 001 was randomly assigned to val, not train

    # Test loading a document
    doc = adapter.load(train_ids[0])
    assert doc.source == "sroie"
    assert len(doc.words) > 0
    assert len(doc.fields) > 0

    fieldtypes = {f.fieldtype for f in doc.fields}
    # Standard SROIE fields
    assert any(ft in fieldtypes for ft in ("company", "date", "address", "total"))

    for w in doc.words:
        assert 0.0 <= w.bbox.x0 <= w.bbox.x1 <= 1.0
        assert 0.0 <= w.bbox.y0 <= w.bbox.y1 <= 1.0


def test_adapters_compatible_with_graph_pipeline():
    """Verify that all three adapters produce Document objects fully compatible with GNN builder."""
    docile_doc = available_adapter(DocileAdapter).load("009bdcdec5c04cd3b2e31555")
    cord_doc = available_adapter(CordAdapter).load("train_0")
    sroie_doc = available_adapter(SroieAdapter).load("000")

    for doc in (docile_doc, cord_doc, sroie_doc):
        # 1. Labeling
        tags, stats = label_document(doc)
        assert len(tags) == len(doc.words)
        assert stats.matched_fields > 0

        # 2. Graph construction
        vocab = build_label_vocabulary(collect_fieldtypes([doc]))
        graph = build_graph(doc, vocab)

        assert graph.num_nodes == len(doc.words)
        assert graph.features.shape[0] == len(doc.words)
        assert graph.edge_index.shape[0] == 2
        assert len(graph.labels) == len(doc.words)


def test_docile_reading_order_preservation():
    """Verify that DocILE OCR words are loaded in top-to-bottom forward reading order."""
    doc = available_adapter(DocileAdapter).load("00134dd365a24343b35b78c6")
    # In forward order, the first words should be near the top of the page
    # while the last words are near the bottom of the page.
    top_words_y = [w.bbox.y0 for w in doc.words[:10]]
    bottom_words_y = [w.bbox.y0 for w in doc.words[-10:]]

    avg_top = sum(top_words_y) / len(top_words_y)
    avg_bottom = sum(bottom_words_y) / len(bottom_words_y)
    assert avg_top < avg_bottom, f"Reading order inverted! top={avg_top:.3f}, bottom={avg_bottom:.3f}"


def test_sroie_label_quality_no_bleeding():
    """Verify SROIE has no giant bounding boxes and does not mislabel item lines as address."""
    adapter = available_adapter(SroieAdapter)
    doc = adapter.load("000")
    tags, stats = label_document(doc)

    # Check that individual field bounding boxes are tight
    for f in doc.fields:
        assert f.bbox.height <= 0.35, f"Field bbox too tall ({f.bbox.height:.3f}) for {f.fieldtype}"

    # Verify that item description words and payment words are 'O', not 'I-address'
    tag_map = {w.text: t for w, t in zip(doc.words, tags)}
    for non_field_word in ("CLAY", "KIDDY", "FISH", "PRICE", "DISC", "AMOUNT", "CASH", "CHANGE"):
        if non_field_word in tag_map:
            assert tag_map[non_field_word] == "O", f"Word '{non_field_word}' should be 'O', got {tag_map[non_field_word]}"


def test_adapters_get_image_api():
    """Verify that get_image method returns a valid PIL Image across all three adapters."""
    from PIL import Image

    docile_adapter = available_adapter(DocileAdapter)
    cord_adapter = available_adapter(CordAdapter)
    sroie_adapter = available_adapter(SroieAdapter)

    img_docile = docile_adapter.get_image("00134dd365a24343b35b78c6")
    assert isinstance(img_docile, Image.Image)
    assert img_docile.width > 0 and img_docile.height > 0

    img_cord = cord_adapter.get_image("train_0")
    assert isinstance(img_cord, Image.Image)
    assert img_cord.width > 0 and img_cord.height > 0

    img_sroie = sroie_adapter.get_image("000")
    assert isinstance(img_sroie, Image.Image)
    assert img_sroie.width > 0 and img_sroie.height > 0
