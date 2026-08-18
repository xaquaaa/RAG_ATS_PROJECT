"""
Tests compute_near_misses() in isolation — pure function, no embedding
model, no DB. Proves the selection/cap/ordering logic is correct before
trusting it wired into the live pipeline (which needs network access to
verify here — see docs on this sandbox's limitation in other test files).
"""
from src.retrieval.pipeline import compute_near_misses


def _scores(*pairs):
    """pairs like ('candidate_a', 0.9) — must already be sorted descending, matching tier1.score_all()'s contract."""
    return list(pairs)


def test_candidate_just_below_cutoff_is_flagged():
    all_scores = _scores(("a", 0.90), ("b", 0.85), ("c", 0.84))  # shortlist_size=2 -> cutoff=0.85
    near_misses = compute_near_misses(all_scores, shortlist_size=2, margin=0.05, max_count=10)
    assert [nm.candidate_id for nm in near_misses] == ["c"]
    assert near_misses[0].margin_below_cutoff == 0.01


def test_candidate_far_below_cutoff_is_not_flagged():
    all_scores = _scores(("a", 0.90), ("b", 0.85), ("c", 0.10))  # way below cutoff
    near_misses = compute_near_misses(all_scores, shortlist_size=2, margin=0.05, max_count=10)
    assert near_misses == []


def test_scanning_stops_at_first_out_of_range_candidate():
    # c is within margin, d is not — even though a hypothetical e further
    # down might coincidentally be within margin again, scanning correctly
    # stops at the first break in a properly-sorted descending list.
    all_scores = _scores(("a", 0.90), ("b", 0.85), ("c", 0.82), ("d", 0.70))
    near_misses = compute_near_misses(all_scores, shortlist_size=2, margin=0.05, max_count=10)
    assert [nm.candidate_id for nm in near_misses] == ["c"]


def test_max_count_caps_a_dense_cluster_near_cutoff():
    all_scores = _scores(("a", 0.90), *[(f"near_{i}", 0.89 - i * 0.001) for i in range(20)])
    near_misses = compute_near_misses(all_scores, shortlist_size=1, margin=0.05, max_count=5)
    assert len(near_misses) == 5


def test_empty_shortlist_returns_no_near_misses():
    # shortlist_size=0 means nothing was shortlisted, so "close to the
    # cutoff" is undefined — must not crash or fabricate a cutoff.
    all_scores = _scores(("a", 0.9), ("b", 0.8))
    near_misses = compute_near_misses(all_scores, shortlist_size=0, margin=0.05, max_count=10)
    assert near_misses == []


def test_no_excluded_candidates_returns_empty():
    all_scores = _scores(("a", 0.9), ("b", 0.8))
    near_misses = compute_near_misses(all_scores, shortlist_size=5, margin=0.05, max_count=10)
    assert near_misses == []
