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
        for result in data["results"]:
            header = f"{result['candidate_id']}  —  Top {result['match_percentile']}% match"
            with st.expander(header):
                if show_details:
                    t = result["technical"]
                    st.caption(f"Tier-1 similarity: {t['tier1_score']}  |  Raw rerank score: {t['rerank_score_raw']}  |  Rank: {result['rank']}")

                for question, answer in result["answers"].items():
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