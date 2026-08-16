"""
Tests compute_near_misses() in isolation — no embedding model, no DB. This
is the actual selection rule: candidates just below the shortlist cutoff,
within a score margin, capped at a max count.
"""
from src.retrieval.pipeline import compute_near_misses


def test_candidate_within_margin_is_flagged():
    # Shortlist cutoff (lowest shortlisted score) is 0.60. A candidate at
    # 0.58 is within a 0.05 margin — should be flagged.
    all_scores = [("a", 0.90), ("b", 0.60), ("c", 0.58)]
    near_misses = compute_near_misses(all_scores, shortlist_size=2, margin=0.05, max_count=10)

    assert len(near_misses) == 1
    assert near_misses[0].candidate_id == "c"
    assert near_misses[0].margin_below_cutoff == 0.02


def test_candidate_outside_margin_is_not_flagged():
    all_scores = [("a", 0.90), ("b", 0.60), ("c", 0.30)]  # 0.30 is far below 0.60
    near_misses = compute_near_misses(all_scores, shortlist_size=2, margin=0.05, max_count=10)
    assert near_misses == []


def test_stops_at_first_candidate_outside_margin_even_if_later_ones_would_qualify():
    # Sorted descending, so once the gap exceeds the margin it can only grow —
    # the function should stop scanning rather than skip past a big gap and
    # keep checking. This also matters for performance on large pools.
    all_scores = [("a", 0.90), ("b", 0.30), ("c", 0.29)]  # "c" would be within
    # margin of "b" but that's irrelevant — margin is always measured against
    # the SHORTLIST cutoff (0.90 here, since shortlist_size=1), not the
    # previous candidate.
    near_misses = compute_near_misses(all_scores, shortlist_size=1, margin=0.05, max_count=10)
    assert near_misses == []


def test_max_count_caps_the_result():
    all_scores = [("shortlisted", 0.90)] + [(f"c{i}", 0.899) for i in range(20)]
    near_misses = compute_near_misses(all_scores, shortlist_size=1, margin=0.05, max_count=3)
    assert len(near_misses) == 3


def test_empty_shortlist_returns_no_near_misses():
    assert compute_near_misses([], shortlist_size=5, margin=0.05, max_count=10) == []


def test_near_misses_sorted_best_first():
    all_scores = [("a", 0.90), ("b", 0.89), ("c", 0.88), ("d", 0.87)]
    near_misses = compute_near_misses(all_scores, shortlist_size=1, margin=0.05, max_count=10)
    ids_in_order = [nm.candidate_id for nm in near_misses]
    assert ids_in_order == ["b", "c", "d"]
