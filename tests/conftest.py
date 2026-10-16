"""
Shared test setup. Nothing here needs Ollama: `fake_llm` replaces llm.generate
and llm.embed with deterministic stand-ins that recognise each prompt type.
"""

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import clock  # noqa: E402
import conversation  # noqa: E402
import llm  # noqa: E402
import memory_stream  # noqa: E402
import project_board  # noqa: E402
import world as world_module  # noqa: E402


def fake_embed(inputs):
    def vec(text):
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in digest[:8]]

    return vec(inputs) if isinstance(inputs, str) else [vec(t) for t in inputs]


def _first_activity(prompt):
    match = re.search(r"Activity:\s*\n\s*(.+)", prompt)
    return match.group(1).strip() if match else ""


def _start_of(activity):
    match = re.search(r"(\d{1,2}:\d{2}\s*[ap]m)", activity, re.IGNORECASE)
    return datetime.strptime(match.group(1).upper(), "%I:%M %p") if match else None


def _lines(start, count, step_minutes, text):
    return "\n".join(
        f"{i + 1}. {(start + timedelta(minutes=i * step_minutes)).strftime('%I:%M %p').lstrip('0').lower()}: {text}"
        for i in range(count)
    )


def fake_generate(prompt, options=None, format=None):
    """Return a plausible reply for whichever prompt the code just built."""
    if format == "json":                                    # world.generate_interactions
        return json.dumps({"interacting_with": None, "changes": []})

    if isinstance(format, dict):                            # structured outputs
        props = format["properties"]
        if "meeting_agreed" in props:                       # plan.extract_outcomes
            return json.dumps({"meeting_agreed": False, "location": "office_1",
                               "activity": "", "minutes": 0, "commitments": []})
        if "actions" in props:                              # project_board.interact
            return json.dumps({"reason": "the board is empty", "actions": [{
                "action": "create_task", "title": "Set up the website API",
                "description": "Create the API the website pages call.",
                "area": "backend", "priority": "high", "depends_on": [],
                "done_when": "The API answers a health check.", "assign_to_self": True,
            }]})
        return json.dumps({"reason": "test", "location": "office_1"})   # choose_location

    if "Respond with only a single integer" in prompt:
        return "5"
    if "Continue the plan as a numbered list" in prompt:
        return ("1. work on development from 7:00 am to 9:00 am\n"
                "2. take a lunch break in the break room from 12:00 pm to 1:00 pm\n"
                "3. review progress from 1:00 pm to 5:00 pm")
    if "Create EXACTLY" in prompt:
        activity = _first_activity(prompt)
        hours = int(re.search(r"EXACTLY (\d+)", prompt).group(1))
        return _lines(_start_of(activity), hours, 60, "Work on the task")
    if "Break it into actions" in prompt:
        return _lines(_start_of(_first_activity(prompt)), 6, 10, "Do a small step")
    if "DECISION" in prompt:
        return "DECISION: 1\nREACTION: none"
    if "most salient high-level questions" in prompt:
        return "1. What is the agent working on?\n2. How does the agent spend the day?\n3. Who does the agent work with?"
    if "high-level insights" in prompt:
        ids = re.findall(r"^(\d+)\.", prompt, re.MULTILINE)
        cite = ", ".join(ids[:2]) or "1"
        return f"1. The agent is steady and focused (because of {cite})\n2. The agent keeps a regular routine (because of {cite})"
    if "core characteristics" in prompt:
        return "A focused engineer."
    return "ok"


@pytest.fixture(autouse=True)
def reset_clock():
    """The sim clock is module state; tests that move it shouldn't leak."""
    clock.set_time("7:00 am")
    yield
    clock.set_time("7:00 am")


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(llm, "generate", fake_generate)
    monkeypatch.setattr(llm, "embed", fake_embed)


@pytest.fixture(scope="session")
def committed_data(tmp_path_factory):
    """The repo's data/ as committed. A sim run rewrites the memory streams and
    world.json in place, and tests built on that leftover state failed until it
    was reverted. Falls back to the working copy outside a git checkout."""
    out = tmp_path_factory.mktemp("committed") / "data"
    shutil.copytree(ROOT / "data", out)
    try:
        listed = subprocess.run(["git", "ls-files", "data"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.split()
        for path in listed:
            blob = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT,
                                  capture_output=True, check=True).stdout
            (out.parent / path).write_bytes(blob)
    except (OSError, subprocess.CalledProcessError):
        pass
    return out


@pytest.fixture
def isolated_data(tmp_path, monkeypatch, committed_data):
    """Point the memory streams and world file at a temp copy of the repo data."""
    data = tmp_path / "data"
    shutil.copytree(committed_data, data)
    monkeypatch.setattr(memory_stream, "DATA_DIR", data)
    monkeypatch.setattr(world_module, "WORLD_PATH", data / "world" / "world.json")
    return data


@pytest.fixture
def world(committed_data):
    return json.loads((committed_data / "world" / "world.json").read_text())


@pytest.fixture
def w(world):
    """A fresh world, with the board's and the conversations' module state
    (pending replans, promises, who talked when) cleared before and after."""
    def reset():
        project_board.clear_workload_changed()
        project_board.clear_commitments()
        project_board.clear_tested()
        conversation.clear_talks()

    reset()
    yield world
    reset()


def seed_module():
    """scripts/seed_memories.py, loaded as a module."""
    spec = importlib.util.spec_from_file_location("seed_memories",
                                                  ROOT / "scripts" / "seed_memories.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- board states several test files start from -------------------------------
def board_api_and_pages(w):
    """Adam's API (T-1) and Eve's three pages (T-2..T-4), each started on one:
    T-1 and T-2 in progress."""
    project_board.get_project(w)["tasks"].clear()
    project_board.create_task(w, "adam", "Design and implement the backend API for the company website",
                              area="backend", priority="high", assign_to_self=True)            # T-1
    for title in ("Design and implement the home page layout",                                # T-2
                  "Design and implement the about page layout",                               # T-3
                  "Design and implement the contact form"):                                   # T-4
        project_board.create_task(w, "eve", title, area="frontend", priority="high", assign_to_self=True)
    project_board.update_status(w, "adam", "T-1", "in_progress")
    project_board.update_status(w, "eve", "T-2", "in_progress")


def board_api_pages_and_db(w):
    """As board_api_and_pages, plus a second backend task for Adam (T-5,
    todo): Eve has three open tasks, Adam two."""
    project_board.get_project(w)["tasks"].clear()
    project_board.create_task(w, "adam", "Design and implement the API for the company website",
                              area="backend", priority="high", assign_to_self=True)            # T-1
    for title in ("Design and implement the homepage layout",                                 # T-2
                  "Design and implement the about page layout",                               # T-3
                  "Design and implement the contact form"):                                   # T-4
        project_board.create_task(w, "eve", title, area="frontend", priority="high", assign_to_self=True)
    project_board.create_task(w, "adam", "Set up database for contact form", area="backend",
                              priority="high", assign_to_self=True)                           # T-5
    project_board.update_status(w, "adam", "T-1", "in_progress")
    project_board.update_status(w, "eve", "T-2", "in_progress")


def board_pages_wait_on_api(w):
    """Adam's T-1 in progress since 7:10 am, and Eve's three pages (T-2..T-4)
    all waiting on it, so she can start nothing."""
    clock.set_time("7:00 am")
    project_board.create_task(w, "adam", "Set up backend API for company website", area="backend",
                              assign_to_self=True)
    for title in ("Design and implement homepage layout", "Design and implement about page layout",
                  "Implement contact form with validation"):
        project_board.create_task(w, "eve", title, area="frontend", depends_on=["T-1"],
                                  assign_to_self=True)
    clock.set_time("7:10 am")
    project_board.update_status(w, "adam", "T-1", "in_progress")
