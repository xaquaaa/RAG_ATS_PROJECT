"""
Minimal demo UI. Run with: streamlit run streamlit_app/app.py

Deliberately shows the evidence + scores for every candidate, not just a
verdict — that transparency IS the portfolio differentiator for this project.
Points at the FastAPI backend; set API_BASE_URL to your deployed Render URL.
"""
import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Smarter ATS — RAG Resume Screening", layout="wide")
st.title("Smarter ATS — RAG Resume Screening")
st.caption("Two-tier retrieval · exhaustive coverage + reranking · evidence-gated answers")

with st.sidebar:
    st.header("1. Index resumes")
    resume_dir = st.text_input("Resume directory", "data/resumes")
    if st.button("Ingest"):
        resp = requests.post(f"{API_BASE_URL}/ingest", json={"resume_directory": resume_dir})
        st.write(resp.json())

st.header("2. Screen candidates")
query = st.text_area("Search query (candidate-level fit)", "5+ years backend experience with distributed systems")
questions_raw = st.text_area("Screening questions (one per line)", "Has AWS certification?\nHas led a team?")
shortlist_size = st.slider("Shortlist size (reranker pass)", 5, 50, 25)
show_details = st.checkbox("Show technical details (raw scores)", value=False)

def _render_answers(answers: dict, show_details: bool):
    for question, answer in answers.items():
        verdict = answer.get("verdict", "UNKNOWN")
        color = {"YES": "green", "NO": "red", "UNKNOWN": "orange"}.get(verdict, "gray")
        label = answer.get("confidence_label")
        suffix = f"  ({label} confidence)" if label else ""
        st.markdown(f"**{question}** — :{color}[{verdict}]{suffix}")

        if answer.get("evidence"):
            st.caption(f"Evidence: {answer['evidence']}")
        elif verdict == "UNKNOWN":
            st.caption("No supporting evidence found in this resume.")

        if show_details:
            st.caption(f"Raw retrieval score: {answer['raw'].get('top_score')}")


def _fetch_candidate_ids() -> list[str]:
    try:
        resp = requests.get(f"{API_BASE_URL}/candidates", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("candidate_ids", [])
    except requests.RequestException:
        pass
    return []


if "last_shown_ids" not in st.session_state:
    st.session_state["last_shown_ids"] = set()

if st.button("Run screening"):
    questions = [q.strip() for q in questions_raw.splitlines() if q.strip()]
    resp = requests.post(
        f"{API_BASE_URL}/screen",
        json={"query": query, "questions": questions, "shortlist_size": shortlist_size},
    )
    if resp.status_code != 200:
        st.error(resp.text)
    else:
        data = resp.json()
        st.session_state["last_shown_ids"] = {r["candidate_id"] for r in data["results"]}

        coverage = data.get("coverage", {})
        not_evaluated = coverage.get("not_evaluated_in_detail", 0)
        if not_evaluated > 0:
            st.info(
                f"Showing detailed evaluation for {coverage['shown_in_detail']} of "
                f"{coverage['total_candidates_indexed']} candidates. {not_evaluated} weren't "
                f"in this shortlist for this search — if you're looking for someone specific, "
                f"use **\"Check a specific candidate\"** below to check them directly regardless "
                f"of how they ranked here."
            )

        for result in data["results"]:
            header = f"{result['candidate_id']}  —  Top {result['match_percentile']}% match"
            with st.expander(header):
                if show_details:
                    t = result["technical"]
                    st.caption(f"Tier-1 similarity: {t['tier1_score']}  |  Raw rerank score: {t['rerank_score_raw']}  |  Rank: {result['rank']}")
                _render_answers(result["answers"], show_details)

st.header("3. Check a specific candidate")
st.caption(
    "Bypasses ranking entirely — use this when you already know who you want "
    "to check, instead of searching broadly. A candidate who wouldn't make "
    "the shortlist above (e.g. their resume doesn't match the search query) "
    "can still be checked correctly here."
)

all_candidate_ids = _fetch_candidate_ids()
if all_candidate_ids:
    shown = st.session_state["last_shown_ids"]
    options = [""] + [
        f"{cid}  (not in last search results)" if shown and cid not in shown else cid
        for cid in all_candidate_ids
    ]
    selection = st.selectbox("Candidate", options)
    direct_candidate_id = selection.split("  (")[0] if selection else ""
else:
    st.caption("Candidate list unavailable — enter an ID manually (ingest candidates first).")
    direct_candidate_id = st.text_input("Candidate ID", placeholder="e.g. candidate_46")

direct_questions_raw = st.text_area(
    "Questions (one per line)", "Has AWS certification?\nHas experience using Java?", key="direct_questions"
)

if st.button("Check candidate"):
    direct_questions = [q.strip() for q in direct_questions_raw.splitlines() if q.strip()]
    resp = requests.post(
        f"{API_BASE_URL}/screen-candidate",
        json={"candidate_id": direct_candidate_id, "questions": direct_questions},
    )
    if resp.status_code != 200:
        st.error(resp.text)
    else:
        data = resp.json()
        st.subheader(data["candidate_id"])
        _render_answers(data["answers"], show_details)