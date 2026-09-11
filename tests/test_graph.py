"""Graph construction and BIO labelling.

These run on hand-built documents so they need neither the dataset nor a GPU.
"""

import numpy as np
import pytest

from app.data.schema import BBox, Document, Field, Word
from app.graph.builder import FEATURE_DIM, build_graph, collect_fieldtypes
from app.graph.edges import (
    EDGE_SAME_LINE,
    EDGE_VERTICAL,
    build_edges,
    knn_edges,
    same_line_edges,
    vertical_edges,
)
from app.graph.labels import (
    OUTSIDE,
    build_label_vocabulary,
    decode_fields,
    label_document,
    reading_order,
)


def word(text, x0, y0, x1, y1, page=0, conf=0.9):
    return Word(text=text, bbox=BBox(x0, y0, x1, y1), page=page, confidence=conf)


@pytest.fixture
def invoice():
    """Two lines: a two-word vendor name, then a single-word total on the row below."""
    words = [
        word("ACME", 0.10, 0.10, 0.20, 0.13),
        word("CORP", 0.21, 0.10, 0.31, 0.13),
        word("Total", 0.10, 0.20, 0.18, 0.23),
        word("1250.00", 0.60, 0.20, 0.75, 0.23),
    ]
    fields = [
        Field("vendor_name", BBox(0.09, 0.09, 0.32, 0.14), 0, "ACME CORP"),
        Field("amount_total", BBox(0.59, 0.19, 0.76, 0.24), 0, "1250.00"),
    ]
    return Document(doc_id="demo", words=words, fields=fields, page_count=1)


# ---------------------------------------------------------------- BBox

def test_covered_by_is_asymmetric():
    small = BBox(0.1, 0.1, 0.2, 0.2)
    large = BBox(0.0, 0.0, 1.0, 1.0)
    assert small.covered_by(large) == pytest.approx(1.0)
    assert large.covered_by(small) < 0.05


def test_degenerate_box_rejected():
    with pytest.raises(ValueError):
        BBox(0.5, 0.1, 0.4, 0.2)


# ---------------------------------------------------------------- labelling

def test_multiword_field_gets_b_then_i(invoice):
    tags, _ = label_document(invoice)
    assert tags[0] == "B-vendor_name"
    assert tags[1] == "I-vendor_name"


def test_single_word_field_gets_only_b(invoice):
    tags, _ = label_document(invoice)
    assert tags[3] == "B-amount_total"


def test_unannotated_word_is_outside(invoice):
    tags, _ = label_document(invoice)
    assert tags[2] == OUTSIDE  # the literal word "Total" is not itself annotated


def test_stats_report_coverage(invoice):
    _, stats = label_document(invoice)
    assert stats.total_fields == 2
    assert stats.matched_fields == 2
    assert stats.coverage == pytest.approx(1.0)
    assert stats.multiword_fields == 1


def test_field_with_no_overlapping_word_is_counted_unmatched():
    doc = Document(
        doc_id="x",
        words=[word("hello", 0.1, 0.1, 0.2, 0.2)],
        fields=[Field("iban", BBox(0.8, 0.8, 0.9, 0.9), 0, "XX")],
    )
    tags, stats = label_document(doc)
    assert tags == [OUTSIDE]
    assert stats.unmatched_fields == 1
    assert stats.unmatched_by_type["iban"] == 1
    assert stats.coverage == 0.0


def test_smaller_field_wins_when_two_overlap():
    """A tight annotation is a more specific description than an enclosing one."""
    doc = Document(
        doc_id="x",
        words=[word("42", 0.50, 0.50, 0.54, 0.53)],
        fields=[
            Field("line_item_region", BBox(0.1, 0.1, 0.9, 0.9), 0),
            Field("line_item_quantity", BBox(0.49, 0.49, 0.55, 0.54), 0),
        ],
    )
    tags, _ = label_document(doc)
    assert tags[0] == "B-line_item_quantity"


def test_reading_order_is_top_to_bottom_then_left_to_right():
    words = [
        word("second", 0.60, 0.10, 0.70, 0.13),
        word("first", 0.10, 0.10, 0.20, 0.13),
        word("third", 0.10, 0.30, 0.20, 0.33),
    ]
    assert [words[i].text for i in reading_order(words)] == ["first", "second", "third"]


# ---------------------------------------------------------------- decoding

def test_decode_recovers_original_fields(invoice):
    tags, _ = label_document(invoice)
    recovered = decode_fields(invoice.words, tags)
    assert {f.fieldtype for f in recovered} == {"vendor_name", "amount_total"}
    vendor = next(f for f in recovered if f.fieldtype == "vendor_name")
    assert vendor.text == "ACME CORP"


def test_decode_separates_adjacent_fields_of_same_type():
    """The reason BIO is used at all: two B- tags must not merge into one field."""
    words = [
        word("A", 0.10, 0.10, 0.14, 0.13),
        word("B", 0.15, 0.10, 0.19, 0.13),
    ]
    tags = ["B-line_item_code", "B-line_item_code"]
    assert len(decode_fields(words, tags)) == 2

    tags = ["B-line_item_code", "I-line_item_code"]
    assert len(decode_fields(words, tags)) == 1


def test_decode_rejects_length_mismatch(invoice):
    with pytest.raises(ValueError):
        decode_fields(invoice.words, ["O"])


# ---------------------------------------------------------------- edges

def test_same_line_connects_horizontal_neighbours(invoice):
    pairs = same_line_edges(invoice.words)
    assert (0, 1) in pairs           # ACME - CORP share a line and are adjacent
    assert (2, 3) not in pairs       # same line, but too far apart


def test_vertical_connects_word_below(invoice):
    pairs = vertical_edges(invoice.words)
    assert (0, 2) in pairs           # ACME sits directly above Total


def test_knn_never_returns_self_loops(invoice):
    assert all(a != b for a, b in knn_edges(invoice.words, k=3))


def test_structural_edge_type_overrides_knn(invoice):
    pairs, types = build_edges(invoice.words, k=3)
    assert types[pairs.index((0, 1))] == EDGE_SAME_LINE
    assert types[pairs.index((0, 2))] == EDGE_VERTICAL


def test_edges_are_deduplicated(invoice):
    pairs, types = build_edges(invoice.words)
    assert len(pairs) == len(set(pairs))
    assert len(pairs) == len(types)


def test_edges_handle_single_word():
    assert build_edges([word("solo", 0.1, 0.1, 0.2, 0.2)]) == ([], [])


def test_words_on_different_pages_are_not_connected():
    words = [
        word("a", 0.10, 0.10, 0.20, 0.13, page=0),
        word("b", 0.21, 0.10, 0.31, 0.13, page=1),
    ]
    assert same_line_edges(words) == []
    assert vertical_edges(words) == []


# ---------------------------------------------------------------- graph

def test_graph_shapes_are_consistent(invoice):
    vocab = build_label_vocabulary(collect_fieldtypes([invoice]))
    graph = build_graph(invoice, vocab)
    assert graph.num_nodes == 4
    assert graph.features.shape == (4, FEATURE_DIM)
    assert graph.labels.shape == (4,)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_type.shape[0] == graph.num_edges
    assert graph.features.dtype == np.float32


def test_vocabulary_reserves_zero_for_outside():
    vocab = build_label_vocabulary(["amount_total", "vendor_name"])
    assert vocab[OUTSIDE] == 0
    assert len(vocab) == 5  # O plus B/I for each of two types


def test_labels_match_tags_through_vocabulary(invoice):
    vocab = build_label_vocabulary(collect_fieldtypes([invoice]))
    graph = build_graph(invoice, vocab)
    assert [vocab[t] for t in graph.tags] == list(graph.labels)


def test_unknown_tag_falls_back_to_outside(invoice):
    """A field type absent from the training vocabulary must not crash the build."""
    graph = build_graph(invoice, build_label_vocabulary(["something_else"]))
    assert set(graph.labels.tolist()) == {0}


def test_undirected_index_doubles_edges(invoice):
    vocab = build_label_vocabulary(collect_fieldtypes([invoice]))
    graph = build_graph(invoice, vocab)
    assert graph.to_undirected_index().shape[1] == 2 * graph.num_edges


def test_empty_document_produces_empty_graph():
    graph = build_graph(Document(doc_id="empty"), build_label_vocabulary([]))
    assert graph.num_nodes == 0
    assert graph.num_edges == 0
    assert graph.features.shape == (0, FEATURE_DIM)


def test_features_stay_in_reasonable_range(invoice):
    vocab = build_label_vocabulary(collect_fieldtypes([invoice]))
    graph = build_graph(invoice, vocab)
    assert np.isfinite(graph.features).all()
    assert graph.features.min() >= 0.0
    assert graph.features.max() <= 40.0  # aspect ratio is the only unbounded feature


# ---------------------------------------------------------------- cache

def test_graph_survives_a_save_load_round_trip(invoice, tmp_path):
    from app.graph.cache import load_graph, save_graph

    vocab = build_label_vocabulary(collect_fieldtypes([invoice]))
    original = build_graph(invoice, vocab)
    path = tmp_path / "g.npz"
    save_graph(path, original)
    restored = load_graph(path)

    assert restored.doc_id == original.doc_id
    assert restored.texts == original.texts
    assert restored.tags == original.tags
    np.testing.assert_array_equal(restored.features, original.features)
    np.testing.assert_array_equal(restored.edge_index, original.edge_index)
    np.testing.assert_array_equal(restored.labels, original.labels)
    assert restored.stats.matched_fields == original.stats.matched_fields


def test_cache_reuses_the_stored_graph(invoice, tmp_path):
    from app.graph.cache import GraphCache

    vocab = build_label_vocabulary(collect_fieldtypes([invoice]))
    cache = GraphCache(tmp_path, vocab)

    first = cache.get(invoice)
    assert cache.path_for(invoice.doc_id).is_file()
    assert cache.cached_ids() == {"demo"}

    second = cache.get(invoice)
    np.testing.assert_array_equal(first.features, second.features)


def test_different_edge_settings_do_not_share_a_cache(invoice, tmp_path):
    """A cache built under other settings must never be silently reused."""
    from app.graph.cache import BuildConfig, GraphCache

    vocab = build_label_vocabulary(collect_fieldtypes([invoice]))
    full = GraphCache(tmp_path, vocab, BuildConfig())
    knn_only = GraphCache(
        tmp_path, vocab, BuildConfig(use_same_line=False, use_vertical=False)
    )

    assert full.path_for("demo") != knn_only.path_for("demo")
    full.get(invoice)
    assert knn_only.cached_ids() == set()


def test_vocabulary_round_trip(tmp_path):
    from app.graph.cache import load_vocabulary, save_vocabulary

    vocab = build_label_vocabulary(["amount_total", "vendor_name"])
    path = tmp_path / "vocab.json"
    save_vocabulary(path, vocab, ["amount_total", "vendor_name"])
    restored, fieldtypes = load_vocabulary(path)

    assert restored == vocab
    assert fieldtypes == ["amount_total", "vendor_name"]


def test_vocabulary_rejects_a_foreign_version(tmp_path):
    import json

    from app.graph.cache import load_vocabulary

    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"version": 99, "vocabulary": {}, "fieldtypes": []}))
    with pytest.raises(ValueError):
        load_vocabulary(path)


def test_vocabulary_covers_every_type_in_the_documents(invoice):
    from app.graph.cache import build_vocabulary

    vocab, fieldtypes = build_vocabulary([invoice])
    assert set(fieldtypes) == {"vendor_name", "amount_total"}
    for name in fieldtypes:
        assert f"B-{name}" in vocab and f"I-{name}" in vocab
