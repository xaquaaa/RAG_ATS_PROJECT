from scripts.eval_metrics import EvalRecord, score_records


def _r(expected, actual):
    return EvalRecord(query="q", candidate_id="c1", question="Has X?", expected=expected, actual=actual)


def test_correct_match():
    report = score_records([_r("YES", "YES")])
    assert report.correct == 1
    assert report.accuracy == 1.0


def test_hallucination_detected():
    # Expected UNKNOWN, system answered confidently — the exact failure
    # mode the evidence gate exists to prevent.
    report = score_records([_r("UNKNOWN", "YES")])
    assert report.hallucinations == 1
    assert report.correct == 0
    assert report.hallucination_rate == 1.0


def test_over_cautious_detected():
    report = score_records([_r("YES", "UNKNOWN")])
    assert report.over_cautious == 1
    assert report.hallucinations == 0  # over-caution must never be miscounted as hallucination


def test_wrong_verdict_detected():
    report = score_records([_r("YES", "NO")])
    assert report.wrong_verdict == 1
    assert report.hallucinations == 0
    assert report.over_cautious == 0


def test_not_retrieved_counted_separately_from_verdict_errors():
    report = score_records([_r("YES", None)])
    assert report.not_retrieved == 1
    assert report.correct == 0
    assert report.wrong_verdict == 0  # missing candidate is a Coverage failure, not a verdict error


def test_mixed_batch_totals():
    records = [
        _r("YES", "YES"),        # correct
        _r("UNKNOWN", "NO"),     # hallucination
        _r("NO", "UNKNOWN"),     # over-cautious
        _r("YES", "NO"),         # wrong verdict
        _r("NO", None),          # not retrieved
    ]
    report = score_records(records)
    assert report.total == 5
    assert report.correct == 1
    assert report.hallucinations == 1
    assert report.over_cautious == 1
    assert report.wrong_verdict == 1
    assert report.not_retrieved == 1
    assert report.accuracy == 0.2


def test_recovered_via_direct_counts_as_correct_not_not_retrieved():
    # /screen missed this candidate, but /screen-candidate found and
    # correctly answered them — this must count as coverage working, not
    # as a Coverage failure.
    record = EvalRecord(
        query="q", candidate_id="c1", question="Has X?",
        expected="YES", actual="YES", recovered_via_direct=True,
    )
    report = score_records([record])
    assert report.correct == 1
    assert report.not_retrieved == 0
    assert report.recovered_via_direct == 1