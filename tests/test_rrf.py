"""
Tests the RRF math in isolation — no database, no embedding model needed.
Proves the fusion logic itself is correct before trusting it wired into
either store.
"""
from src.retrieval.rrf import reciprocal_rank_fusion


def test_item_in_both_lists_scores_higher_than_item_in_one():
    vector_ranked = ["a", "b", "c"]
    keyword_ranked = ["b", "a", "c"]

    scores = reciprocal_rank_fusion([vector_ranked, keyword_ranked], k=60)

    # "b" and "a" both appear near the top of both lists; "c" is last in both.
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["c"]


def test_item_present_in_only_one_list_still_scores():
    vector_ranked = ["a", "b"]
    keyword_ranked = ["c"]  # e.g. an exact acronym match vector search missed entirely

    scores = reciprocal_rank_fusion([vector_ranked, keyword_ranked], k=60)

    assert "c" in scores
    assert scores["c"] > 0


def test_top_rank_in_both_lists_beats_top_rank_in_only_one():
    vector_ranked = ["a", "b", "c"]
    keyword_ranked = ["a", "c", "b"]

    scores = reciprocal_rank_fusion([vector_ranked, keyword_ranked], k=60)

    # "a" is rank 1 in both lists — should be the clear top fused score.
    assert scores["a"] == max(scores.values())


def test_item_absent_from_all_lists_has_no_score():
    scores = reciprocal_rank_fusion([["a", "b"]], k=60)
    assert "z" not in scores
