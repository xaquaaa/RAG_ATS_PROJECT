"""
Pure metrics computation for the evaluation harness. Deliberately separated
from the API-calling orchestration (run_evaluation.py) so the scoring logic
is unit-testable without a live server or Groq key.

The metric that matters most for THIS project isn't overall accuracy — it's
the confusion breakdown, because the two error types mean very different
things:

  - HALLUCINATION (expected UNKNOWN, got YES/NO): the evidence gate failed
    to hold — the system answered confidently with no real support. This is
    the exact failure mode the whole project exists to prevent.
  - OVER-CAUTION (expected YES/NO, got UNKNOWN): the gate was too strict —
    real evidence existed but didn't clear the threshold. Annoying, costs
    you recall, but safe.

A system that's 100% accurate on YES/NO cases but hallucinates on every
UNKNOWN case is worse for this use case than one with lower raw accuracy but
zero hallucinations. Don't just look at the top-line number.
"""
from dataclasses import dataclass, field


@dataclass
class EvalRecord:
    query: str
    candidate_id: str
    question: str
    expected: str  # "YES" | "NO" | "UNKNOWN"
    actual: str | None  # None if candidate wasn't retrieved at all (Coverage failure)
    recovered_via_direct: bool = False  # True if /screen missed them but /screen-candidate found them


@dataclass
class EvalReport:
    total: int = 0
    correct: int = 0
    hallucinations: int = 0     # expected UNKNOWN, got YES or NO
    over_cautious: int = 0      # expected YES/NO, got UNKNOWN
    wrong_verdict: int = 0      # expected YES got NO, or expected NO got YES
    not_retrieved: int = 0      # candidate missing from /screen AND /screen-candidate couldn't recover them
    recovered_via_direct: int = 0  # /screen missed them, /screen-candidate found and answered correctly/incorrectly
    by_record: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return round(self.correct / self.total, 3) if self.total else 0.0

    @property
    def hallucination_rate(self) -> float:
        # Denominator: all cases where a confident answer was possible
        # (i.e. every case, since hallucination means answering confidently
        # when the expected answer was UNKNOWN).
        return round(self.hallucinations / self.total, 3) if self.total else 0.0

    def summary(self) -> str:
        return (
            f"Total cases:            {self.total}\n"
            f"Accuracy:                {self.accuracy:.1%}\n"
            f"Hallucination rate:      {self.hallucination_rate:.1%}  "
            f"({self.hallucinations}/{self.total} — expected UNKNOWN, system answered anyway)\n"
            f"Over-cautious:           {self.over_cautious}  (expected an answer, system said UNKNOWN)\n"
            f"Wrong verdict:           {self.wrong_verdict}  (expected YES got NO, or vice versa)\n"
            f"Recovered via direct:    {self.recovered_via_direct}  "
            f"(missed by /screen's shortlist, found via /screen-candidate)\n"
            f"Not retrieved (either):  {self.not_retrieved}  (missing from BOTH /screen and /screen-candidate — check the candidate_id)\n"
        )


def score_records(records: list[EvalRecord]) -> EvalReport:
    report = EvalReport()
    for r in records:
        report.total += 1
        row = {
            "query": r.query, "candidate_id": r.candidate_id, "question": r.question,
            "expected": r.expected, "actual": r.actual, "recovered_via_direct": r.recovered_via_direct,
        }

        if r.actual is None:
            report.not_retrieved += 1
            row["result"] = "not_retrieved"
        elif r.actual == r.expected:
            report.correct += 1
            if r.recovered_via_direct:
                report.recovered_via_direct += 1
            row["result"] = "correct" if not r.recovered_via_direct else "correct (recovered)"
        elif r.expected == "UNKNOWN" and r.actual in ("YES", "NO"):
            report.hallucinations += 1
            if r.recovered_via_direct:
                report.recovered_via_direct += 1
            row["result"] = "hallucination"
        elif r.expected in ("YES", "NO") and r.actual == "UNKNOWN":
            report.over_cautious += 1
            if r.recovered_via_direct:
                report.recovered_via_direct += 1
            row["result"] = "over_cautious"
        else:
            report.wrong_verdict += 1
            if r.recovered_via_direct:
                report.recovered_via_direct += 1
            row["result"] = "wrong_verdict"

        report.by_record.append(row)
    return report