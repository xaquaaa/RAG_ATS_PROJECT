"""
Gathers real top_score distributions from your actual data via /score-preview
(zero Groq cost — the evidence gate runs, the LLM never gets called) so you
can set EVIDENCE_THRESHOLD and the confidence-label margins from measured
data instead of guessed constants.

Usage:
    python scripts/analyze_confidence_scores.py --ground-truth data/eval/ground_truth.json

Reuses your ground-truth file's (candidate_id, question, expected_verdict)
triples as the query set — you don't need a separate dataset, and it doubles
as a check that your expected-YES/NO cases actually produce meaningfully
higher scores than your expected-UNKNOWN cases (if they don't, that's a
signal the embedding model itself is struggling on this data, not just a
threshold-tuning problem).
"""
import argparse
import json
import statistics
from pathlib import Path

import requests

from scripts.run_evaluation import load_ground_truth


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * pct), len(sorted_vals) - 1)
    return sorted_vals[idx]


def summarize(label: str, scores: list[float]):
    if not scores:
        print(f"\n{label}: no cases")
        return
    print(f"\n{label}  (n={len(scores)})")
    print(f"  min={min(scores):.3f}  p25={percentile(scores, 0.25):.3f}  "
          f"median={statistics.median(scores):.3f}  p75={percentile(scores, 0.75):.3f}  max={max(scores):.3f}")
    print(f"  all values: {sorted(round(s, 3) for s in scores)}")


def run(api_base_url: str, ground_truth_path: str):
    entries = load_ground_truth(ground_truth_path)

    answerable_scores: list[float] = []   # expected YES or NO — a real answer should be findable
    unknown_scores: list[float] = []      # expected UNKNOWN — no evidence should exist

    for e in entries:
        resp = requests.post(
            f"{api_base_url}/score-preview",
            json={"candidate_id": e["candidate_id"], "questions": [e["question"]]},
        )
        if resp.status_code != 200:
            print(f"  [skip] {e['candidate_id']} / \"{e['question']}\": {resp.text}")
            continue

        data = resp.json()
        score_info = data["scores"][e["question"]]
        top_score = score_info["top_score"]

        bucket = "UNKNOWN" if e["expected_verdict"] == "UNKNOWN" else "answerable"
        (unknown_scores if bucket == "UNKNOWN" else answerable_scores).append(top_score)

        print(f"  [{e['candidate_id']}] \"{e['question']}\" expected={e['expected_verdict']} "
              f"top_score={top_score}  matched_via={score_info['matched_via']}")

    summarize("Expected YES/NO (should score HIGH — these are your 'real matches')", answerable_scores)
    summarize("Expected UNKNOWN (should score LOW — no real evidence exists)", unknown_scores)

    if answerable_scores and unknown_scores:
        gap = min(answerable_scores) - max(unknown_scores)
        print(f"\nSeparation: lowest answerable score ({min(answerable_scores):.3f}) minus "
              f"highest UNKNOWN score ({max(unknown_scores):.3f}) = {gap:.3f}")
        if gap < 0:
            print("  ⚠ These OVERLAP — some UNKNOWN cases scored higher than some real matches. "
                  "No single threshold can perfectly separate them with this data; consider "
                  "collecting more ground truth before trusting a specific cutoff.")

        suggested_threshold = percentile(answerable_scores, 0.10)  # low end of real matches
        print(f"\nSuggested EVIDENCE_THRESHOLD: ~{suggested_threshold:.3f} "
              f"(10th percentile of real-match scores — clears most true matches, "
              f"still above the UNKNOWN cluster if separation above is positive)")

        if len(answerable_scores) >= 4:
            high_cutoff = percentile(answerable_scores, 0.5) - suggested_threshold
            med_cutoff = percentile(answerable_scores, 0.25) - suggested_threshold
            print(f"Suggested confidence margins (replace the hardcoded 0.20 / 0.05 in "
                  f"_confidence_label): High >= {max(high_cutoff, 0.0):.3f}, "
                  f"Medium >= {max(med_cutoff, 0.0):.3f}")
    else:
        print("\nNeed at least one case in each bucket (YES/NO and UNKNOWN) to suggest thresholds. "
              "Add more ground-truth cases covering both.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--ground-truth", default="data/eval/ground_truth.json")
    args = parser.parse_args()
    run(args.api_base_url, args.ground_truth)
