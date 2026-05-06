"""End-to-end label construction from Review JSON without a heavy tokenizer.

Uses the whitespace segmenter (no VnCoreNLP download required) and verifies
that per-word BIO labels line up with the annotated cause spans.
"""
from src.data.dataset import build_tagged_examples
from src.data.label_schema import (
    ASPECT_SENT_ID2LABEL,
    CAUSE_ID2LABEL,
)
from src.data.schema import load_reviews
from src.data.segmenter import WhitespaceSegmenter


def test_build_tagged_examples_sample_data():
    reviews = load_reviews("scripts/sample_data.json")
    seg = WhitespaceSegmenter()
    examples = build_tagged_examples(reviews, seg)
    assert len(examples) == 2

    # Sample 1: "Ship lâu nhưng đóng gói đẹp"
    ex0 = examples[0]
    asp_tags = [ASPECT_SENT_ID2LABEL[i] for i in ex0.aspect_sent_labels]
    cau_tags = [CAUSE_ID2LABEL[i] for i in ex0.cause_labels]
    # Words: ["Ship", "lâu", "nhưng", "đóng", "gói", "đẹp"]
    assert ex0.words == ["Ship", "lâu", "nhưng", "đóng", "gói", "đẹp"]
    # "Ship lâu" → DEL-NEG, "đóng gói đẹp" → PACK-POS.
    assert asp_tags[0] == "B-DEL-NEG"
    assert asp_tags[1] == "I-DEL-NEG"
    assert asp_tags[2] == "O"
    assert asp_tags[3] == "B-PACK-POS"
    assert asp_tags[4] == "I-PACK-POS"
    assert asp_tags[5] == "I-PACK-POS"

    # Cause head mirrors the same B/I structure.
    assert cau_tags[0] == "B-CAUSE"
    assert cau_tags[1] == "I-CAUSE"
    assert cau_tags[2] == "O"
    assert cau_tags[3] == "B-CAUSE"
    assert cau_tags[4] == "I-CAUSE"
    assert cau_tags[5] == "I-CAUSE"


def test_build_tagged_examples_three_aspects():
    # Sample 2 has 3 disjoint aspects.
    reviews = load_reviews("scripts/sample_data.json")
    examples = build_tagged_examples(reviews, WhitespaceSegmenter())
    ex1 = examples[1]
    asp_tags = [ASPECT_SENT_ID2LABEL[i] for i in ex1.aspect_sent_labels]
    # Every aspect should produce at least one B-* tag.
    b_tags = {t for t in asp_tags if t.startswith("B-")}
    assert "B-QUAL-POS" in b_tags
    assert "B-PRICE-POS" in b_tags
    assert "B-SVC-POS" in b_tags
