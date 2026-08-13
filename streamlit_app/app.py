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
            with st.expander(f"{result['candidate_id']}  —  rerank score: {result['rerank_score']}"):
                st.write(f"Tier-1 score: {result['tier1_score']}")
                for question, answer in result["answers"].items():
                    verdict = answer.get("verdict", "UNKNOWN")
                    color = {"YES": "green", "NO": "red", "UNKNOWN": "orange"}.get(verdict, "gray")
                    st.markdown(f"**{question}** — :{color}[{verdict}]")
                    if answer.get("evidence"):
                        st.caption(f"Evidence: {answer['evidence']}")
                    st.caption(f"Retrieval confidence: {answer.get('top_score')}")
