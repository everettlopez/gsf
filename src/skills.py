"""
skills.py - what the hands know about the team's way of working.

skills/<name>/SKILL.md holds the conventions for one lane (frontend, backend,
deploy) or one kind of session (review); skills/shared/SKILL.md is the
contract every task works within. The executor's system prompt carries the
ones a task needs, picked from the task's area and what it is about.
"""

import re
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

LANES = {"frontend": "frontend", "design": "frontend", "backend": "backend", "devops": "deploy"}
_DEPLOY = re.compile(r"\b(?:deploy\w*|ship\w*|release\w*|publish\w*)\b", re.IGNORECASE)


def for_task(task):
    """The skills a work session on `task` gets: the shared contract, the
    task's lane, and deploy for a task about shipping the site (not one that
    only mentions dist/: that made a build task think it had to ship)."""
    names = ["shared"]
    lane = LANES.get(task.get("area"))
    if lane:
        names.append(lane)
    text = f"{task.get('title', '')} {task.get('description', '')}"
    if "deploy" not in names and _DEPLOY.search(text):
        names.append("deploy")
    return names


def for_review():
    return ["shared", "review"]


def body(name):
    """A skill's text without its front matter, or "" if there is no such skill."""
    path = SKILLS_DIR / name / "SKILL.md"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        text = text.split("---", 2)[2]
    return text.strip()


def load(names):
    return "\n\n".join(b for b in (body(n) for n in names) if b)
