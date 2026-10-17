"""
chronicle.py - what happened during the run, kept for the team's website.

The website the agents build is a window into their own working days
(snapshot.py turns this into the data it shows). The simulation itself only
keeps the present: where everyone is now, the board as it stands, the memory
streams. The chronicle keeps the story of the run, in plain data:

    steps          every agent's place and action, each step
    events         the run's notable events (log.event: board changes, work and
                   review sessions, merges, questions, promises), with a kind
                   and the agents they are about
    conversations  each conversation's lines and what came of it
    days           each workday's name and the plans the agents started it with

It is saved to data/chronicle.json (not committed: every run writes its own)
and starts afresh with each run.
"""

import json
import re
from pathlib import Path

import clock
import log

FILE = "chronicle.json"              # in the data directory, beside world/ and agents/

MAX_EVENTS = 2000                   # a long run's events; the oldest go first

# An event's kind, from how log.event lines are written (first match wins).
_KINDS = [
    ("board_refused", re.compile(r"^📋✗")),
    ("board", re.compile(r"^📋")),
    ("work", re.compile(r"^🔧")),
    ("review", re.compile(r"^🔍")),
    ("ship", re.compile(r"^🚢")),
    ("cost", re.compile(r"^💲")),
    ("question", re.compile(r"^❓")),
    ("meeting", re.compile(r"agreed to meet|put the meeting in their plans")),
    ("promise", re.compile(r"\bcommitted:")),
    ("board_trip", re.compile(r"heads (?:straight )?to the board")),
    ("reaction", re.compile(r"\breacts:")),
    ("plan", re.compile(r"\breplanned\b")),
    ("hands", re.compile(r"\bhands\b")),
]
_EMOJI = re.compile(r"^[^\w\"'(]+")

_state = {}
_names = {}                         # "ADAM" / "Adam Beck" / "Adam" -> "adam"


def start(world):
    """A new run: forget the last one and start listening to its events."""
    _state.clear()
    _state.update({"version": 1, "start_date": clock.now().date().isoformat(),
                   "days": [], "steps": [], "events": [], "conversations": []})
    _names.clear()
    for agent in world["agents"].values():
        for key in (agent["id"].upper(), agent["name"], agent["name"].split()[0]):
            _names[key] = agent["id"]
    if _listen not in log.listeners:
        log.listeners.append(_listen)


def stop():
    if _listen in log.listeners:
        log.listeners.remove(_listen)


def data():
    """The chronicle so far (the live dict: read, don't change)."""
    return _state


def _now():
    return {"day": clock.day(), "time": clock.now().strftime("%I:%M %p").lstrip("0").lower()}


def agents_in(text):
    """The ids of the agents a line is about, in the order they appear."""
    found = []
    for match in re.finditer(r"[A-Z][A-Za-z]*(?: [A-Z][a-z]+)?", text or ""):
        word = match.group(0)
        agent_id = _names.get(word) or _names.get(word.split()[0])
        if agent_id and agent_id not in found:
            found.append(agent_id)
    return found


def kind_of(text):
    for kind, pattern in _KINDS:
        if pattern.search(text):
            return kind
    return "event"


def _listen(text):
    if not _state:
        return
    events = _state["events"]
    events.append({**_now(), "kind": kind_of(text), "agents": agents_in(text),
                   "text": _EMOJI.sub("", text).strip()})
    del events[:-MAX_EVENTS]


def note_day(plans):
    """A workday begins. `plans`: agent id -> the broad plan it starts with
    (a list of activities)."""
    if _state:
        _state["days"].append({"day": clock.day(), "name": clock.day_name(),
                               "date": clock.now().date().isoformat(), "plans": plans})


def note_step(world):
    """Where every agent is and what they are doing, this step."""
    if not _state:
        return
    when = _now()
    for agent in world["agents"].values():
        _state["steps"].append({**when, "agent": agent["id"],
                                "location": agent["current_location"],
                                "action": agent["current_action"],
                                "object": agent.get("interacting_with")})


def note_conversation(location, topic, lines, outcomes=None):
    """`lines`: [(speaker name, text)]; `outcomes`: plan.extract_outcomes's."""
    if not _state:
        return
    outcomes = outcomes or {}
    speakers = []
    for name, _ in lines:
        agent_id = _names.get(name) or _names.get(name.split()[0])
        if agent_id and agent_id not in speakers:
            speakers.append(agent_id)
    _state["conversations"].append({
        **_now(), "location": location, "topic": (topic or "").strip(), "agents": speakers,
        "lines": [{"agent": _names.get(name) or _names.get(name.split()[0]) or name,
                   "text": text} for name, text in lines],
        "meeting": outcomes.get("meeting"),
        "commitments": {(_names.get(n) or _names.get(n.split()[0]) or n): c
                        for n, c in (outcomes.get("commitments") or {}).items() if c},
    })


def default_path():
    """data/chronicle.json, wherever the memory streams' data directory is."""
    import memory_stream

    return Path(memory_stream.DATA_DIR) / FILE


def save(path=None):
    path = Path(path or default_path())
    if not _state:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_state, indent=1, ensure_ascii=False), encoding="utf-8")


def load(path=None):
    """The chronicle a run saved, or None."""
    path = Path(path or default_path())
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
