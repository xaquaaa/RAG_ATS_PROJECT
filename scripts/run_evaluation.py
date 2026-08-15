"""
Runs the evaluation harness against a live API instance.

Usage:
    python scripts/run_evaluation.py --ground-truth data/eval/ground_truth.json

Requires the API to be running (locally or deployed) with candidates already
ingested via /ingest, and GROQ_API_KEY set on the server for /screen to
actually reach the generation step (cases gated to UNKNOWN don't need it).

Groups ground-truth entries by query so each distinct query only costs one
/screen call, regardless of how many candidate/question pairs you're
checking against that query's results.
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

from scripts.eval_metrics import EvalRecord, score_records


def load_ground_truth(path: str) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    entries = [e for e in raw if "query" in e and not str(e.get("query", "")).startswith("REPLACE")]
    if not entries:
        raise ValueError(
            f"No real entries found in {path} — it still only contains the "
            "template placeholder. Fill in real ground truth first (see the "
            "_instructions field in that file)."
        )
    return entries


def run(api_base_url: str, ground_truth_path: str, out_dir: str):
    entries = load_ground_truth(ground_truth_path)

    by_query: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_query[e["query"]].append(e)

    records: list[EvalRecord] = []

    for query, group in by_query.items():
        questions = sorted({e["question"] for e in group})
        resp = requests.post(f"{api_base_url}/screen", json={"query": query, "questions": questions})
        resp.raise_for_status()
        data = resp.json()

        results_by_candidate = {r["candidate_id"]: r for r in data["results"]}

        for e in group:
            candidate_result = results_by_candidate.get(e["candidate_id"])
            recovered = False
            if candidate_result is None:
                # /screen's shortlist missed this candidate. Fall back to
                # the direct-candidate path before giving up — this is what
                # actually proves the coverage fix closes the loop, rather
                # than just existing as an untested endpoint.
                try:
                    direct_resp = requests.post(
                        f"{api_base_url}/screen-candidate",
                        json={"candidate_id": e["candidate_id"], "questions": [e["question"]]},
                    )
                    if direct_resp.status_code == 200:
                        direct_data = direct_resp.json()
                        answer = direct_data["answers"].get(e["question"], {})
                        actual = answer.get("raw", answer).get("verdict")
                        recovered = actual is not None
                    else:
                        actual = None
                except requests.RequestException:
                    actual = None
            else:
                answer = candidate_result["answers"].get(e["question"], {})
                actual = answer.get("raw", answer).get("verdict")  # handles both response shapes

            records.append(EvalRecord(
                query=query,
                candidate_id=e["candidate_id"],
                question=e["question"],
                expected=e["expected_verdict"],
                actual=actual,
                recovered_via_direct=recovered,
            ))

    report = score_records(records)

    print(report.summary())
    print("Per-case breakdown:")
    for row in report.by_record:
        flag = "" if row["result"] == "correct" else f"  <-- {row['result'].upper()}"
        print(f"  [{row['candidate_id']}] \"{row['question']}\" expected={row['expected']} actual={row['actual']}{flag}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_file = out_path / f"eval_{timestamp}.json"
    result_file.write_text(json.dumps({
        "summary": {
            "total": report.total, "accuracy": report.accuracy,
            "hallucination_rate": report.hallucination_rate,
            "hallucinations": report.hallucinations, "over_cautious": report.over_cautious,
            "wrong_verdict": report.wrong_verdict, "not_retrieved": report.not_retrieved,
            "recovered_via_direct": report.recovered_via_direct,
        },
        "records": report.by_record,
    }, indent=2))
    print(f"\nSaved full report to {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--ground-truth", default="data/eval/ground_truth.json")
    parser.add_argument("--out-dir", default="eval_results")
    args = parser.parse_args()
    run(args.api_base_url, args.ground_truth, args.out_dir)