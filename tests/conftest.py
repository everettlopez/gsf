"""
Shared test setup. Nothing here needs Ollama: `fake_llm` replaces llm.generate
and llm.embed with deterministic stand-ins that recognise each prompt type.
"""

import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import llm  # noqa: E402
import memory_stream  # noqa: E402
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


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(llm, "generate", fake_generate)
    monkeypatch.setattr(llm, "embed", fake_embed)


@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    """Point the memory streams and world file at a temp copy of the repo data."""
    data = tmp_path / "data"
    shutil.copytree(ROOT / "data", data)
    monkeypatch.setattr(memory_stream, "DATA_DIR", data)
    monkeypatch.setattr(world_module, "WORLD_PATH", data / "world" / "world.json")
    return data


@pytest.fixture
def world():
    return json.loads((ROOT / "data" / "world" / "world.json").read_text())
