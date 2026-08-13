"""
Generates synthetic resumes as .txt files in data/resumes/ so you never need
real candidate PII for a public GitHub repo/demo.

Run: python scripts/generate_synthetic_resumes.py --count 100

This is intentionally simple template-based generation. If you want more
realistic variety, swap in an LLM call (Groq free tier) to rewrite each
template — just don't do it per-candidate at query time, only at dataset
build time, or you'll defeat the point of keeping retrieval free.
"""
import argparse
import random
from pathlib import Path

ROLES = ["Backend Engineer", "Data Analyst", "Frontend Developer", "DevOps Engineer", "ML Engineer"]
SKILLS_POOL = ["Python", "SQL", "React", "AWS", "Docker", "Kubernetes", "FastAPI", "PyTorch", "Java", "Go", "Terraform", "PostgreSQL"]
COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella Systems", "Hooli", "Stark Industries"]
DEGREES = ["B.Tech Computer Science", "B.Sc Information Technology", "M.Sc Data Science", "B.Tech Electronics"]

TEMPLATE = """{name}
{role}

SUMMARY
{years} years of experience as a {role} focused on {focus_skill} and scalable systems.

EXPERIENCE
{role} — {company} ({years_start}–2026)
- Built and maintained systems using {skill_a} and {skill_b}.
- Collaborated with cross-functional teams to ship features on schedule.
{leadership_line}

SKILLS
{skills_line}

EDUCATION
{degree}, {grad_year}
"""


def build_resume(idx: int) -> str:
    role = random.choice(ROLES)
    skills = random.sample(SKILLS_POOL, k=random.randint(3, 6))
    years = random.randint(1, 10)
    leadership = random.random() > 0.6
    return TEMPLATE.format(
        name=f"Candidate {idx}",
        role=role,
        years=years,
        focus_skill=skills[0],
        company=random.choice(COMPANIES),
        years_start=2026 - years,
        skill_a=skills[0],
        skill_b=skills[1] if len(skills) > 1 else skills[0],
        leadership_line="- Led a team of engineers on a major platform migration." if leadership else "",
        skills_line=", ".join(skills),
        degree=random.choice(DEGREES),
        grad_year=2026 - years - random.randint(0, 3),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", default="data/resumes")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        (out_dir / f"candidate_{i}.txt").write_text(build_resume(i), encoding="utf-8")

    print(f"Generated {args.count} synthetic resumes in {out_dir}")


if __name__ == "__main__":
    main()
