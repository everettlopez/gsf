"""
project_board.py - the team's shared project board.

The board is an ordinary object in the world (``project_board`` in the
conference room), so agents only learn about it the way they learn about
anything else: by being in the room and perceiving it. The data behind it lives
in ``world["project"]``, a structured record the LLM never edits directly:

    "project": {
      "name": "Company Website",
      "goal": "Build and deploy the company website",
      "description": "...",
      "status": "not_started" | "in_progress" | "done",
      "tasks": [ <task>, ... ]
    }

Every task has the same shape (see ``new_task``):

    {
      "id": "T-1",
      "title": "Set up the backend API",
      "description": "What the work is.",
      "area": "backend",              # one of AREAS
      "priority": "high",             # one of PRIORITIES
      "status": "todo",               # one of STATUSES
      "assignee": "adam" | null,      # agent id
      "created_by": "adam",
      "depends_on": ["T-2"],          # ids of tasks that must be done first
      "done_when": "How the team knows it is finished.",
      "approved_by": ["eve"],         # teammates who signed off while in review
      "created_at": "7:35 am",
      "updated_at": "7:35 am",
      "history": [{"time": "7:35 am", "agent": "adam", "event": "created"}]
    }

Review rule: the assignee moves their own task todo -> in_progress -> review.
While it is in review, a teammate (anyone but the assignee) approves it at the
board; only then can the assignee move it to done. Moving it back out of review
clears the approvals.

How agents change it (paper: an action changes the state of the object being
used): when an agent is interacting with the board, ``interact`` makes one
schema-constrained LLM call that returns a list of board actions. Those actions
are validated and applied here in plain Python, and each one is written to the
acting agent's memory stream as an observation. Other agents find out by
looking at the board, or by being told in conversation.
"""

import json
import re
from datetime import datetime

import clock
import llm
import log
import persona
from llm import CTX

BOARD_OBJECT = "project_board"

STATUSES = ("todo", "in_progress", "blocked", "review", "done")
PRIORITIES = ("high", "medium", "low")
AREAS = ("backend", "frontend", "design", "devops", "testing", "general")
ACTIONS = ("create_task", "claim_task", "assign_task", "update_status", "edit_task",
           "add_note", "approve_task")

# An activity that is explicitly about the board (routes the agent to it).
_BOARD = re.compile(
    r"\b(?:project board|task board|the board|board)\b"
    r"|\btasks? (?:status(?:es)?|assignments?)\b"            # "update task statuses"
    r"|\bupdat\w* (?:the )?(?:status of )?(?:tasks?\b|T-\d+)"  # "update T-1"
    r"|\b(?:assign|claim|approv)\w*\b.*?\b(?:tasks?\b|T-\d+)"   # "assign T-4 to Adam"
    r"|\b(?:mark|move|set|flag)\w*\s+(?:task\s+)?T-\d+"     # "mark T-1 as completed"
    r"|\b(?:done|review|in[- ]progress|todo|blocked)\s+column\b"  # "move it to the done column"
    r"|\bstart\s+T-\d+\b",                                 # "start T-3" (a status change)
    re.IGNORECASE,
)
# A commitment that can only be kept at the board: it names the board, or a
# board action applied to a task ("move T-1 to review", "approve T-4", "assign
# T-3 to Eve", "update the status of T-2"). Naming a task isn't enough: "I'll
# keep working on T-2" sent both agents back to the board three times in 20 min.
_BOARD_COMMITMENT = re.compile(
    r"\bboard\b"
    r"|\b(?:move|mark|set|update|approve|assign|claim|add|create|start|review|edit|reassign"
    r"|hand)\w*\s+(?:[\w'-]+\s+){0,3}?(?:T-\d+|tasks?)\b"
    r"|\b(?:status|statuses|dependenc\w*|priorit\w*)\s+(?:of|for|on)\s+(?:T-\d+|tasks?)\b"
    r"|\btask\s+(?:status|statuses|assignments?)\b",
    re.IGNORECASE,
)


def mentions_board_explicitly(activity):
    # Reactions name the object as it appears in observations ("the project_board").
    return bool(_BOARD.search((activity or "").replace("_", " ")))


def is_board_commitment(text):
    return bool(_BOARD_COMMITMENT.search(text or ""))

MAX_ACTIONS_PER_VISIT = 4     # keeps one board visit from flooding the board
MAX_TASKS_SHOWN = 12          # observations get long fast; done tasks are summarized


class BoardError(ValueError):
    """An action the board refuses (unknown task, duplicate title, ...)."""


class NoChange(BoardError):
    """The action asks for what is already true (e.g. a status a task already
    has). Skipped quietly: it is not a mistake worth remembering."""


WIP_LIMIT = 1            # tasks one agent can have in progress at once
MAX_BOARD_NOTES = 20     # board-level notes kept on the project

# An agent can't ask for review on work that only just started (the run that
# motivated this moved T-1 to review 50 minutes after starting it, after doing
# nothing but planning). Enforced for agents at the board, not in update_status,
# so scripted moves and tests can still walk a task straight through.
MIN_MINUTES_BEFORE_REVIEW = 120

# Open tasks one agent can take on at the board. Past this, a task they add is
# left unassigned for whoever is free, and they can't claim more. Adam kept
# adding tasks to himself (T-1, T-5, T-6 within an hour) while T-1 had barely
# started. Agent layer only, like MIN_MINUTES_BEFORE_REVIEW.
MAX_OPEN_TASKS_PER_AGENT = 3

# Titles that describe a meeting or a review, not something to build or ship.
_NOT_A_DELIVERABLE = re.compile(
    r"^\s*(?:review|meet|meeting|discuss|sync|align|check[- ]in|stand-?up|coordinate"
    r"|catch up|follow up|talk)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now():
    return clock.now().strftime("%I:%M %p").lstrip("0").lower()


def _normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).strip()


def _clean(text, limit=200):
    return " ".join((text or "").split())[:limit].strip().rstrip(".")


def _parse_clock(text):
    """'7:10 am' -> a datetime on today's sim date, or None."""
    try:
        t = datetime.strptime((text or "").strip().upper(), "%I:%M %p").time()
    except ValueError:
        return None
    return datetime.combine(clock.now().date(), t)


def _last_event(task, suffix):
    """The latest history entry whose event ends with `suffix` (e.g. '-> review')."""
    for entry in reversed(task["history"]):
        if str(entry.get("event", "")).endswith(suffix):
            return entry
    return None


def started_at(task):
    """When the task last went in progress ('7:10 am'), or None."""
    entry = _last_event(task, "-> in_progress")
    return entry["time"] if entry else None


def minutes_in_progress(task):
    """How long the task has been in progress, by the sim clock, or None."""
    start = _parse_clock(started_at(task))
    if task["status"] != "in_progress" or start is None:
        return None
    return int((clock.now() - start).total_seconds() // 60)


def changes_requested(task):
    """Did a teammate leave a note on the task since it went to review?"""
    entered = _last_event(task, "-> review")
    if entered is None:
        return False
    after = task["history"][task["history"].index(entered) + 1:]
    return any(e.get("event") == "note" and e.get("agent") != task["assignee"] for e in after)


def get_project(world):
    """The project record, with defaults filled in for older world files."""
    project = world.setdefault("project", {})
    project.setdefault("name", "Project")
    project.setdefault("goal", "")
    project.setdefault("description", "")
    project.setdefault("status", "not_started")
    project.setdefault("tasks", [])
    project.setdefault("notes", [])
    for task in project["tasks"]:
        task.setdefault("approved_by", [])
    return project


def board_location(world):
    """The room that holds the project board, or None."""
    for room, data in world["locations"].items():
        if BOARD_OBJECT in data.get("objects", {}):
            return room
    return None


def get_task(world, task_id):
    task_id = (task_id or "").strip().upper()
    for task in get_project(world)["tasks"]:
        if task["id"] == task_id:
            return task
    raise BoardError(f"there is no task {task_id or '(none)'} on the board")


def _next_id(project):
    numbers = [int(t["id"].split("-")[1]) for t in project["tasks"]
               if re.fullmatch(r"T-\d+", t.get("id", ""))]
    return f"T-{max(numbers, default=0) + 1}"


def _agent_name(world, agent_id):
    agent = world["agents"].get(agent_id)
    return agent["name"] if agent else agent_id


def _log_event(task, agent_id, event, note=None):
    entry = {"time": _now(), "agent": agent_id, "event": event}
    if note:
        entry["note"] = note
    task["history"].append(entry)
    task["updated_at"] = entry["time"]


def task_waits(world, task):
    """Is this task held up by dependencies that aren't done?"""
    return bool(_unmet_dependencies(world, task))


def _unmet_dependencies(world, task):
    tasks = {t["id"]: t for t in get_project(world)["tasks"]}
    return [d for d in task["depends_on"] if tasks.get(d, {}).get("status") != "done"]


def refresh_project_status(world):
    """Derive the project status from its tasks (no LLM)."""
    project = get_project(world)
    tasks = project["tasks"]
    if not tasks:
        project["status"] = "not_started"
    elif all(t["status"] == "done" for t in tasks):
        project["status"] = "done"
    else:
        project["status"] = "in_progress"
    return project["status"]


def sync_board_object(world):
    """Keep the board object's one-line state in step with the project."""
    room = board_location(world)
    if room is None:
        return
    tasks = get_project(world)["tasks"]
    if not tasks:
        state = "blank, with no tasks on it yet"
    else:
        counts = {s: sum(t["status"] == s for t in tasks) for s in STATUSES}
        parts = [f"{n} {s.replace('_', ' ')}" for s, n in counts.items() if n]
        state = f"listing {len(tasks)} tasks ({', '.join(parts)})"
    world["locations"][room]["objects"][BOARD_OBJECT]["state"] = state


# ---------------------------------------------------------------------------
# Workload changes (read by main.py to replan the rest of the agent's day)
# ---------------------------------------------------------------------------
_workload_changed = {}       # agent id -> notices about what changed for them


def mark_workload_changed(agent_id, notice=None):
    notices = _workload_changed.setdefault(agent_id, [])
    if notice and notice not in notices:
        notices.append(notice)


def workload_changed(agent_id):
    """Did this agent's workload change (took on, finished, or was handed a task)?"""
    return agent_id in _workload_changed


def workload_notices(agent_id):
    """What changed for this agent that they didn't do themselves, e.g. a
    teammate assigning them a task (the board's notification)."""
    return list(_workload_changed.get(agent_id, []))


def clear_workload_changed(agent_id=None):
    if agent_id is None:
        _workload_changed.clear()
    else:
        _workload_changed.pop(agent_id, None)


# ---------------------------------------------------------------------------
# Promises made in conversation that have to be kept at the board
# ---------------------------------------------------------------------------
COMMITMENT_VISITS = 2        # give up on a promise after this many visits without acting
_commitments = {}            # agent id -> [{"text": ..., "visits": n}]


def add_commitment(agent_id, text):
    items = _commitments.setdefault(agent_id, [])
    if all(c["text"] != text for c in items):
        items.append({"text": text, "visits": 0})


def commitments_for(agent_id):
    return [c["text"] for c in _commitments.get(agent_id, [])]


def clear_commitments(agent_id=None):
    if agent_id is None:
        _commitments.clear()
    else:
        _commitments.pop(agent_id, None)


def _after_visit(agent_id, acted):
    """A visit where the agent changed something settles their promises;
    otherwise they get one more visit before the promise is dropped."""
    if acted:
        clear_commitments(agent_id)
        return
    items = _commitments.get(agent_id, [])
    for c in items:
        c["visits"] += 1
    _commitments[agent_id] = [c for c in items if c["visits"] < COMMITMENT_VISITS]


def open_tasks_for(world, agent_id):
    """The agent's own tasks that are not done yet."""
    return [t for t in get_project(world)["tasks"]
            if t["assignee"] == agent_id and t["status"] != "done"]


# ---------------------------------------------------------------------------
# Board operations (deterministic; these are the only ways tasks change)
# ---------------------------------------------------------------------------
def new_task(task_id, agent_id, title, description="", area="general",
             priority="medium", depends_on=(), done_when=""):
    now = _now()
    return {
        "id": task_id,
        "title": title,
        "description": description,
        "area": area,
        "priority": priority,
        "status": "todo",
        "assignee": None,
        "created_by": agent_id,
        "depends_on": list(depends_on),
        "done_when": done_when,
        "approved_by": [],
        "created_at": now,
        "updated_at": now,
        "history": [{"time": now, "agent": agent_id, "event": "created"}],
    }


def create_task(world, agent_id, title, description="", area="general",
                priority="medium", depends_on=(), done_when="", assign_to_self=False):
    project = get_project(world)
    # "T-1: Set up the API" -> "Set up the API": the board numbers tasks itself
    # (a title copied from a promise showed up as "T-1 'T-1: Set up ...'").
    title = _clean(re.sub(r"^\s*T-\d+\s*[:.\-\u2013\u2014]?\s*", "", title or "", flags=re.IGNORECASE), 80)
    if not title:
        raise BoardError("a task needs a title")
    if _NOT_A_DELIVERABLE.match(title):
        raise BoardError("tasks are things to build or ship; reviews happen with "
                         "approve_task and meetings in conversation")

    key = _normalize(title)
    for existing in project["tasks"]:
        if _normalize(existing["title"]) == key:
            raise BoardError(f"{existing['id']} '{existing['title']}' is already on the board")

    area = area if area in AREAS else "general"
    priority = priority if priority in PRIORITIES else "medium"
    known = {t["id"] for t in project["tasks"]}
    deps = list(dict.fromkeys(d.strip().upper() for d in depends_on or ()
                              if d and d.strip().upper() in known))

    task = new_task(_next_id(project), agent_id, title, _clean(description, 300),
                    area, priority, deps, _clean(done_when, 200))
    if assign_to_self:
        task["assignee"] = agent_id
        _log_event(task, agent_id, "claimed")
        mark_workload_changed(agent_id)
    project["tasks"].append(task)
    refresh_project_status(world)
    sync_board_object(world)
    return task


def claim_task(world, agent_id, task_id):
    task = get_task(world, task_id)
    if task["assignee"] == agent_id:
        raise BoardError(f"{task['id']} is already assigned to you")
    if task["assignee"]:
        raise BoardError(f"{task['id']} is already claimed by {_agent_name(world, task['assignee'])}")
    if task["status"] == "done":
        raise BoardError(f"{task['id']} is already done")
    task["assignee"] = agent_id
    _log_event(task, agent_id, "claimed")
    mark_workload_changed(agent_id)
    sync_board_object(world)
    return task


def update_status(world, agent_id, task_id, status, note=""):
    """Only the assignee moves a task (starting an unassigned task claims it).
    Done needs a teammate's approval while the task is in review."""
    task = get_task(world, task_id)
    if status not in STATUSES:
        raise BoardError(f"unknown status {status!r}")
    if task["status"] == status:
        raise NoChange(f"{task['id']} is already {status}")
    if task["assignee"] and task["assignee"] != agent_id:
        raise BoardError(f"{task['id']} belongs to {_agent_name(world, task['assignee'])}")
    if not task["assignee"] and status != "in_progress":
        raise BoardError(f"{task['id']} has no one on it yet; take it on first")
    if status == "in_progress":
        blockers = _unmet_dependencies(world, task)
        if blockers:
            raise BoardError(f"{task['id']} is waiting on {', '.join(blockers)}")
        busy = [t["id"] for t in get_project(world)["tasks"]
                if t["assignee"] == agent_id and t["status"] == "in_progress"
                and t["id"] != task["id"]]
        if len(busy) >= WIP_LIMIT:
            raise BoardError(f"{_agent_name(world, agent_id)} is already working on "
                             f"{', '.join(busy)}; move it to review (or blocked, if it is stuck) first")
        if not task["assignee"]:
            task["assignee"] = agent_id              # starting work claims it
            mark_workload_changed(agent_id)
    if status == "blocked" and task["status"] not in ("in_progress", "review"):
        # Blocked means work that got stuck. Eve marked two untouched tasks
        # "blocked" to mean "later", invented blockers for them, and the next
        # conversation was all about "unblocking" them.
        raise BoardError(f"{task['id']} hasn't been started, so it can't be blocked; if it "
                         f"has to wait for another task, add that task to its depends_on "
                         f"with edit_task")
    if status == "review" and task["status"] not in ("in_progress", "blocked"):
        # Review is for work that was actually done. Without this, a todo task
        # went straight to review while another task sat "in progress" unworked.
        raise BoardError(f"{task['id']} hasn't been started; move it to in progress "
                         f"before asking for review")
    if status == "done":
        if task["status"] != "review":
            raise BoardError(f"{task['id']} has to go to review before it is done")
        if not task["approved_by"]:
            raise BoardError(f"{task['id']} needs a teammate to approve it first")

    old = task["status"]
    if old == "review" and status != "done":
        task["approved_by"] = []                     # reopened: approvals no longer apply
    task["status"] = status
    _log_event(task, agent_id, f"{old} -> {status}", _clean(note) or None)
    if status == "done":
        mark_workload_changed(agent_id)
    refresh_project_status(world)
    sync_board_object(world)
    return task


def assign_task(world, agent_id, task_id, assignee, note=""):
    """Give an unassigned task, or your own task, to a teammate (or yourself).
    A handoff note is stored as the task's latest note and sent with the notice."""
    task = get_task(world, task_id)
    to = (assignee or "").strip().lower()
    if to not in world["agents"]:
        # Accept a name ("Adam Beck", "adam") as well as an id
        to = next((a["id"] for a in world["agents"].values()
                   if a["name"].lower() == to or a["name"].split()[0].lower() == to), to)
    if to not in world["agents"]:
        raise BoardError(f"there is no teammate called {assignee!r}")
    if task["status"] == "done":
        raise BoardError(f"{task['id']} is already done")
    if task["assignee"] == to:
        raise BoardError(f"{task['id']} is already assigned to {_agent_name(world, to)}")
    if task["assignee"] and task["assignee"] != agent_id:
        raise BoardError(f"{task['id']} belongs to {_agent_name(world, task['assignee'])}; "
                         f"only they can hand it over")

    previous = task["assignee"]
    task["assignee"] = to
    task["approved_by"] = []
    _log_event(task, agent_id, f"assigned to {to}")
    note = _clean(note)
    if note:
        _log_event(task, agent_id, "note", note)
    if to != agent_id:
        notice = (f"{_agent_name(world, agent_id)} assigned task {task['id']} "
                  f"'{task['title']}' to {_agent_name(world, to)} on the project board")
        notice += f", noting: \"{note}\"." if note else "."
        mark_workload_changed(to, notice)
    else:
        mark_workload_changed(to)
    if previous == agent_id and to != agent_id:
        mark_workload_changed(agent_id)               # handed off their own work
    sync_board_object(world)
    return task


def edit_task(world, agent_id, task_id, depends_on=None, priority=None,
              description=None, done_when=None):
    """Change a task's details. The assignee can, or anyone while it's unassigned."""
    task = get_task(world, task_id)
    if task["assignee"] and task["assignee"] != agent_id:
        raise BoardError(f"{task['id']} belongs to {_agent_name(world, task['assignee'])}")
    changed = []
    if depends_on is not None:
        known = {t["id"] for t in get_project(world)["tasks"]} - {task["id"]}
        deps = sorted({d.strip().upper() for d in depends_on if d and d.strip().upper() in known})
        if deps != task["depends_on"]:
            task["depends_on"] = deps
            changed.append("dependencies")
    if priority in PRIORITIES and priority != task["priority"]:
        task["priority"] = priority
        changed.append("priority")
    if description and _clean(description, 300) != task["description"]:
        task["description"] = _clean(description, 300)
        changed.append("description")
    if done_when and _clean(done_when, 200) != task["done_when"]:
        task["done_when"] = _clean(done_when, 200)
        changed.append("finish line")
    if not changed:
        raise BoardError(f"nothing to change on {task['id']}")
    _log_event(task, agent_id, "edited " + ", ".join(changed))
    return task, changed


def approve_task(world, agent_id, task_id, note=""):
    """A teammate signs off on a task that is in review."""
    task = get_task(world, task_id)
    if task["status"] != "review":
        raise BoardError(f"{task['id']} is not in review")
    if task["assignee"] == agent_id:
        raise BoardError(f"{task['id']} is your own task; a teammate has to approve it")
    if agent_id in task["approved_by"]:
        raise BoardError(f"you already approved {task['id']}")
    task["approved_by"].append(agent_id)
    _log_event(task, agent_id, "approved", _clean(note) or None)
    return task


def add_board_note(world, agent_id, note):
    """A note on the board itself, not on one task (a blocker, a decision,
    something for whoever comes by next)."""
    note = _clean(note)
    if not note:
        raise BoardError("an empty note")
    notes = get_project(world)["notes"]
    if notes and notes[-1]["agent"] == agent_id and notes[-1]["note"] == note:
        raise NoChange("that note is already on the board")
    notes.append({"time": _now(), "agent": agent_id, "note": note})
    del notes[:-MAX_BOARD_NOTES]
    return notes[-1]


def add_note(world, agent_id, task_id, note):
    task = get_task(world, task_id)
    note = _clean(note)
    if not note:
        raise BoardError("an empty note")
    _log_event(task, agent_id, "note", note)
    return task


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------
def latest_note(task):
    """The most recent note on a task, or None."""
    for entry in reversed(task["history"]):
        if entry.get("event") == "note":
            return entry
    return None


def describe_task(world, task):
    """One stable line per task. Exact wording matters: perceive.py only stores
    lines that changed since the last step."""
    who = (f"assigned to {_agent_name(world, task['assignee'])}"
           if task["assignee"] else "unassigned")
    line = (f"Task {task['id']} on the project board, '{task['title']}' "
            f"({task['area']}, {task['priority']} priority), is "
            f"{task['status'].replace('_', ' ')} and {who}")
    blockers = _unmet_dependencies(world, task) if task["status"] != "done" else []
    if blockers:
        line += f", waiting on {', '.join(blockers)}"
    if task["approved_by"]:
        line += ", approved by " + " and ".join(_agent_name(world, a) for a in task["approved_by"])
    line += "."
    note = latest_note(task)
    if note:
        # Notes are how agents ask each other things on the board, so the
        # latest one is part of what anyone reading the board sees.
        line += (f" The latest note on it, from {_agent_name(world, note['agent'])}, "
                 f"says: \"{note['note']}\"")
    return line


TASK_REF = re.compile(r"\bT-\d+\b", re.IGNORECASE)


def task_ids_in(text):
    """Task ids mentioned in a piece of text, in order, without repeats."""
    return list(dict.fromkeys(m.upper() for m in TASK_REF.findall(text or "")))


def task_fact(world, task):
    """One plain sentence about where a task really stands, spelled out so a
    model can't read "in progress" as "basically done". Used where agents talk
    or write about tasks (conversations, notes, plans)."""
    who = _agent_name(world, task["assignee"]) if task["assignee"] else "nobody"
    status = task["status"]
    if status == "todo":
        state = "todo: not started"
    elif status == "in_progress":
        since = started_at(task)
        state = (f"in progress{' since ' + since if since else ''}: "
                 f"not in review, not approved, not done")
    elif status == "review":
        state = ("in review, approved by " + " and ".join(
            _agent_name(world, a) for a in task["approved_by"])
            + ": the assignee can now move it to done") if task["approved_by"] \
            else "in review, not approved yet: waiting for a teammate to approve it"
    elif status == "blocked":
        state = "blocked"
    else:
        state = "done"
    line = f"{task['id']} '{task['title']}' ({who}'s) is {state}."
    blockers = _unmet_dependencies(world, task) if status != "done" else []
    if blockers:
        line += (f" It can't start until {', '.join(blockers)} "
                 f"{'is' if len(blockers) == 1 else 'are'} done.")
    return line


def task_facts(world, task_ids):
    """task_fact for each existing id, plus the tasks they wait on."""
    tasks = {t["id"]: t for t in get_project(world)["tasks"]}
    ordered = []
    for tid in task_ids:
        task = tasks.get(tid.upper())
        if task is None:
            continue
        ordered.append(task["id"])
        ordered.extend(d for d in task["depends_on"] if d in tasks)
    return [task_fact(world, tasks[t]) for t in dict.fromkeys(ordered)]


def task_id_line(world):
    """The ids that exist, said plainly for prompts. With an empty board the
    model filled in "T-1" and "T-2" on its own (prompts used them as examples)."""
    ids = [t["id"] for t in get_project(world)["tasks"]]
    if not ids:
        return ("The project board has no tasks yet, so no work has a task id or "
                "number yet; talk about the work itself, not task numbers.")
    return (f"The only tasks on the board are {', '.join(ids)}; there is no other "
            f"task id.")


def strip_unknown_task_ids(world, text):
    """Remove task ids that aren't on the board ("work on T-5 'Auth'" ->
    "work on 'Auth'", "review T-1, T-6 and T-7" -> "review T-1")."""
    return strip_task_ids(text, unknown_task_ids(world, text))


def strip_task_ids(text, ids, with_titles=False):
    """Remove these task ids from `text`, keeping the list around them readable.
    with_titles also removes a quoted title right after an id ("T-2 'Home page'")."""
    title = r"(?:\s*'[^']*')?" if with_titles else ""
    for tid in ids:
        t = re.escape(tid)
        for pattern in (rf"(?:\s*,\s*|\s+and\s+){t}\b{title}",     # "T-1, T-6" / "T-1 and T-6"
                        rf"\b{t}{title}(?:\s*,\s*|\s+and\s+)",     # "T-6, T-1" / "T-6 and T-1"
                        rf"\b{t}\b{title}\s*"):                      # "work on T-6 'X'"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    if ids:
        text = re.sub(r"\s{2,}", " ", text).strip()
        # "... and check on T-2" -> "... and check on" -> "..."
        text = re.sub(r"(?:\s+(?:and|or)\s+\w+)?\s+(?:on|with|for|of|to|about|and)\s*$", "", text,
                      flags=re.IGNORECASE)
    return text


def unknown_task_ids(world, text):
    """Ids mentioned in `text` that aren't on the board (e.g. a plan's 'T-5')."""
    known = {t["id"] for t in get_project(world)["tasks"]}
    return [t for t in task_ids_in(text) if t not in known]


def board_observations(world):
    """What anyone standing at the board can read off it."""
    project = get_project(world)
    lines = [f"The project board shows the team goal: {project['goal']}."]
    tasks = project["tasks"]
    if not tasks:
        lines.append("The project board has no tasks on it yet.")
        return lines + _board_note_lines(world)

    open_tasks = [t for t in tasks if t["status"] != "done"]
    done = [t for t in tasks if t["status"] == "done"]
    order = {p: i for i, p in enumerate(PRIORITIES)}
    open_tasks.sort(key=lambda t: (order.get(t["priority"], 9), t["id"]))

    for task in open_tasks[:MAX_TASKS_SHOWN]:
        lines.append(describe_task(world, task))
    if len(open_tasks) > MAX_TASKS_SHOWN:
        lines.append(f"The project board has {len(open_tasks) - MAX_TASKS_SHOWN} more open tasks.")
    if done:
        lines.append(f"The project board shows {len(done)} of {len(tasks)} tasks done.")
    lines.extend(_board_note_lines(world))
    return lines


def _board_note_lines(world):
    notes = get_project(world)["notes"]
    if not notes:
        return []
    n = notes[-1]
    return [f"The latest note on the project board, from {_agent_name(world, n['agent'])}, "
            f"says: \"{n['note']}\""]


def status_for_agent(world, agent_id):
    """What the board means for this agent right now: gaps in their area, work
    waiting on them, and their own tasks. Plain lines, used both at the board
    and when planning the rest of the day after a visit."""
    project = get_project(world)
    tasks = project["tasks"]
    agent = world["agents"][agent_id]
    name, area = agent["name"], agent.get("area")
    lines = []

    if not tasks:
        lines.append("The project board has no tasks yet; the work toward the goal still has to be laid out.")
    else:
        done = sum(t["status"] == "done" for t in tasks)
        lines.append(f"{done} of {len(tasks)} tasks are done.")
        areas = sorted({a.get("area") for a in world["agents"].values() if a.get("area")}
                       | {t["area"] for t in tasks})
        per_area = ", ".join(f"{a} {sum(t['area'] == a and t['status'] != 'done' for t in tasks)}"
                             for a in areas)
        lines.append(f"Open tasks per area: {per_area}.")
        per_person = ", ".join(f"{a['name']} {len(open_tasks_for(world, a['id']))}"
                               for a in world["agents"].values())
        lines.append(f"Open tasks per person: {per_person}.")

    if area:
        mine_area = [t for t in tasks if t["area"] == area and t["status"] != "done"]
        if not mine_area:
            lines.append(f"There are no open {area} tasks on the board, and {area} is "
                         f"{name}'s area: {name} should add the {area} work the goal needs.")
        unassigned = [t for t in mine_area if not t["assignee"]]
        if unassigned:
            lines.append(f"Unassigned {area} tasks: "
                         + ", ".join(f"{t['id']} '{t['title']}'" for t in unassigned) + ".")

    reviews = [t for t in tasks if t["status"] == "review" and t["assignee"]
               and t["assignee"] != agent_id and agent_id not in t["approved_by"]]
    for t in reviews:
        lines.append(f"{_agent_name(world, t['assignee'])}'s {t['id']} '{t['title']}' is "
                     f"waiting for a teammate's review.")

    mine = open_tasks_for(world, agent_id)
    mine_ids = {t["id"] for t in mine}
    for t in tasks:
        waiting = [d for d in t["depends_on"] if d in mine_ids]
        if waiting and t["assignee"] != agent_id and t["status"] != "done":
            who = _agent_name(world, t["assignee"]) if t["assignee"] else "An unassigned task"
            lines.append(f"{who}'s {t['id']} '{t['title']}' is waiting on {name}'s "
                         f"{', '.join(waiting)}.")

    if mine:
        lines.append(f"{name}'s open tasks: " + ", ".join(
            f"{t['id']} '{t['title']}' ({t['status'].replace('_', ' ')})" for t in mine) + ".")
    else:
        lines.append(f"{name} has no open tasks.")
    lines.extend(waiting_lines(world, agent_id))
    return lines


def waiting_lines(world, agent_id):
    """The agent's own todo tasks that can't start because they wait on
    unfinished work. When that is all of them, say so plainly and point at the
    way out: in the run behind this, every frontend task waited on Adam's whole
    backend API, so Eve could not start anything all morning."""
    name = _agent_name(world, agent_id)
    mine = open_tasks_for(world, agent_id)
    todo = [t for t in mine if t["status"] == "todo"]
    waiting = [(t, _unmet_dependencies(world, t)) for t in todo]
    waiting = [(t, deps) for t, deps in waiting if deps]
    if not waiting:
        return []
    lines = [f"{name}'s {t['id']} waits on {', '.join(deps)}, which "
             f"{'is' if len(deps) == 1 else 'are'} not done yet, so it can't start."
             for t, deps in waiting]
    active = [t for t in mine if t["status"] in ("in_progress", "review")]
    if len(waiting) == len(todo) and not active:
        blockers = sorted({d for _, deps in waiting for d in deps})
        lines.append(
            f"None of {name}'s tasks can start until {', '.join(blockers)} "
            f"{'is' if len(blockers) == 1 else 'are'} done. If one of them doesn't really "
            f"need {', '.join(blockers)} (a page can be built against mock data), {name} "
            f"can drop that dependency with edit_task and start it.")
    return lines


def board_text_for_prompt(world):
    """Fuller view for the agent who is actually working at the board."""
    project = get_project(world)
    notes = "".join(f"\n- board note ({_agent_name(world, n['agent'])}, {n['time']}): {n['note']}"
                    for n in project["notes"][-3:])
    if not project["tasks"]:
        return "(no tasks yet)" + notes
    rows = []
    for t in project["tasks"]:
        who = _agent_name(world, t["assignee"]) if t["assignee"] else "unassigned"
        deps = f" depends on {', '.join(t['depends_on'])};" if t["depends_on"] else ""
        approved = (" approved by " + ", ".join(_agent_name(world, a) for a in t["approved_by"])
                    + ";") if t["approved_by"] else ""
        since = (f" in progress since {started_at(t)};"
                 if t["status"] == "in_progress" and started_at(t) else "")
        row = (f"- {t['id']} [{t['status']}] {t['title']} ({t['area']}, "
               f"{t['priority']}, {who});{since}{deps}{approved} done when: {t['done_when'] or 'n/a'}")
        note = latest_note(t)
        if note:
            row += f"\n    latest note ({_agent_name(world, note['agent'])}): {note['note']}"
        rows.append(row)
    return "\n".join(rows) + notes


# ---------------------------------------------------------------------------
# Acting at the board
# ---------------------------------------------------------------------------
def action_schema(world):
    task_ids = [t["id"] for t in get_project(world)["tasks"]]
    task_id = {"type": "string", "enum": task_ids} if task_ids else {"type": "string"}
    # "actions" comes first and "reason" last: the model writes fields in this
    # order, so if it runs out of tokens mid-reason the actions are already
    # complete and can be recovered (see _parse_board_reply).
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": list(ACTIONS)},
                        "task_id": task_id,
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "area": {"type": "string", "enum": list(AREAS)},
                        "priority": {"type": "string", "enum": list(PRIORITIES)},
                        "done_when": {"type": "string"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "assign_to_self": {"type": "boolean"},
                        "assignee": {"type": "string", "enum": list(world["agents"])},
                        "status": {"type": "string", "enum": list(STATUSES)},
                        "note": {"type": "string"},
                    },
                    "required": ["action"],
                },
            },
            "reason": {"type": "string"},
        },
        "required": ["actions", "reason"],
    }


def _parse_board_reply(raw):
    """The board call's JSON, or what can be saved of it. A long "reason" once
    ran past the token limit ("Unterminated string"), losing the whole visit;
    with "actions" first, everything before "reason" is still usable."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cut = raw.rfind(', "reason"')
        if cut == -1:
            cut = raw.rfind(',"reason"')
        if cut != -1:
            try:
                result = json.loads(raw[:cut] + "}")
                log.debug("board reply was cut off in its reason; kept its actions")
                return {**result, "reason": "(cut off)"}
            except json.JSONDecodeError:
                pass
        raise


NOTE_REQUIRED = {"blocked": "say what is blocking it",
                 "review": "say what the reviewer should check"}


_CLAIM_WORDS = r"(complete|completed|done|finished|approved|merged|deployed|in review)"


def check_note_claims(world, note):
    """Refuse a note that says a task is done or approved when the board says
    otherwise. Notes are what teammates read at the board, so a note saying
    "T-2 can proceed since T-1 is complete" (while T-1 was in progress) spread
    a false belief to everyone who walked up. Everything else in a note is
    free text."""
    tasks = {t["id"]: t for t in get_project(world)["tasks"]}
    for tid in task_ids_in(note):
        task = tasks.get(tid)
        if task is None:
            raise BoardError(f"the note mentions {tid}, but there is no {tid} on the board")
        claim = re.search(
            rf"\b{tid}\b(?:\s*'[^']*')?\s*(?:is|has been|was|'s|got|being)?\s*"
            rf"(?:now\s+|already\s+|fully\s+)?{_CLAIM_WORDS}\b", note, re.IGNORECASE)
        if not claim:
            continue
        word = claim.group(1).lower()
        status = task["status"]
        if word == "approved":
            wrong = not task["approved_by"] and status != "done"
        elif word == "in review":
            wrong = status not in ("review", "done")
        else:
            wrong = status != "done"
        if wrong:
            raise BoardError(f"the note says {tid} is {word}, but the board shows {tid} "
                             f"is {status.replace('_', ' ')}; notes have to match the board")


def _has_full_plate(world, agent_id):
    return len(open_tasks_for(world, agent_id)) >= MAX_OPEN_TASKS_PER_AGENT


def in_lane(world, agent_id, area):
    """Is work in `area` this agent's to take on? Each agent has an area
    (backend, frontend); "general" work is anyone's. An agent with no area
    can take anything."""
    own = world["agents"][agent_id].get("area")
    return not own or area in (own, "general")


def _lane_owner(world, area):
    """The teammate whose area this is, or None."""
    return next((a for a in world["agents"].values() if a.get("area") == area), None)


def _check_lane(world, agent_id, area, who="you"):
    if in_lane(world, agent_id, area):
        return
    owner = _lane_owner(world, area)
    raise BoardError(f"that is {area} work" + (f", which is {owner['name']}'s area" if owner else "")
                     + f"; {who} can offer to work on it together with them instead")


# ---------------------------------------------------------------------------
# Ownership (who works on what)
# ---------------------------------------------------------------------------
def owner_name(world, task):
    return _agent_name(world, task["assignee"]) if task["assignee"] else None


def teammate_task_ids(world, agent_id, text):
    """Ids in `text` of tasks that belong to a teammate (not to this agent,
    and not unassigned)."""
    tasks = {t["id"]: t for t in get_project(world)["tasks"]}
    return [tid for tid in task_ids_in(text)
            if tid in tasks and tasks[tid]["assignee"] not in (None, agent_id)]


# "review T-2", "approve Eve's T-2", "review and approve T-1 'X'": looking a
# teammate's work over is everyone's job. ("meets the review criteria so we
# can move T-2 to review" is not: the task id doesn't follow the verb.)
_REVIEW_OF = re.compile(r"\b(?:review|approv)\w*\s+(?:and\s+approv\w*\s+)?(?:[\w']+\s+){0,2}?T-\d+",
                        re.IGNORECASE)


def is_review_of_task(text):
    return bool(_REVIEW_OF.search(text or ""))


def ownership_lines(world, agent_id):
    """Whose tasks are whose, said plainly for prompts. Plans and conversations
    used to treat every task on the board as up for grabs: Adam offered to
    "split up" Eve's frontend tasks, promised to get her T-2 ready for review,
    and spent the morning sketching her home page."""
    tasks = [t for t in get_project(world)["tasks"] if t["status"] != "done"]
    if not tasks:
        return []
    name = _agent_name(world, agent_id)
    lines = []
    for a in world["agents"].values():
        ids = [t["id"] for t in tasks if t["assignee"] == a["id"]]
        if ids:
            who = f"{name}'s own" if a["id"] == agent_id else f"{a['name']}'s"
            lines.append(f"{who} tasks: {', '.join(ids)}.")
    loose = [t["id"] for t in tasks if not t["assignee"]]
    if loose:
        lines.append(f"Unassigned: {', '.join(loose)}.")
    lines.append(f"Each task belongs to its assignee: only they work on it and move it on the "
                 f"board. {name} can review a teammate's task with them, or offer to work on "
                 f"one together with them (the teammate decides), but never does it alone.")
    return lines


def check_note_is_for_teammates(world, agent, note):
    """Refuse a note that only narrates what the writer is doing ("Eve is
    currently addressing high-priority bugs or issues. Please check if this
    affects T-2 progress."). That note described work no task covered, and
    sent Adam off for 15 minutes to "check on T-2" alone. A note is for a
    teammate to act on, so narration passes only when it asks something
    ("?") or names the teammate it is for."""
    first, full = agent["name"].split()[0], agent["name"]
    narrating = re.match(
        rf"\s*(?:{re.escape(full)}|{re.escape(first)}|I)(?:'m|'s|\s+(?:is|am|has been|will be))"
        rf"\s+(?:currently\s+|now\s+|still\s+|busy\s+)?\w+ing\b", note or "", re.IGNORECASE)
    if not narrating or "?" in note:
        return
    for other in world["agents"].values():
        if other["id"] != agent["id"] and re.search(
                rf"\b{re.escape(other['name'].split()[0])}\b", note, re.IGNORECASE):
            return
    raise BoardError("a note is for a teammate to act on (a question, a decision, a "
                     "blocker); it shouldn't only say what you are doing")


def _check_review_habits(world, agent_id, task_id, status):
    """Two habits the core rules don't cover, which the 130-step run broke:
    - review is for finished work, so not right after starting;
    - a task in review waits for the teammate: the assignee only pulls it back
      out when that teammate asked for changes (a note on the task)."""
    try:
        task = get_task(world, task_id)
    except BoardError:
        return                               # update_status reports the bad id
    if task["assignee"] != agent_id:
        return                               # update_status reports who owns it

    if status == "review" and task["status"] == "in_progress":
        minutes = minutes_in_progress(task)
        if minutes is not None and minutes < MIN_MINUTES_BEFORE_REVIEW:
            raise BoardError(
                f"{task['id']} only went in progress at {started_at(task)}; review is "
                f"for finished work, so keep working on it")

    if task["status"] == "review" and status in ("in_progress", "todo") \
            and not changes_requested(task):
        raise BoardError(
            f"{task['id']} is waiting for a teammate's review; leave it in review "
            f"unless a teammate asks for changes")


def _required_note(action, why):
    note = _clean(action.get("note"))
    if not note:
        raise BoardError(f"needs a note to {why}")
    return note


def apply_action(world, agent, action):
    """Apply one action dict. Returns a memory line, or raises BoardError.

    This is the agent-facing layer, so it also enforces the team's habits:
    handing a task to someone, blocking a task, or asking for review always
    comes with a note the teammate can act on."""
    kind = action.get("action")
    aid, name = agent["id"], agent["name"]

    if kind == "create_task":
        to_self = bool(action.get("assign_to_self"))
        area = action.get("area", "general")
        area = area if area in AREAS else "general"
        full = to_self and _has_full_plate(world, aid)
        other_lane = to_self and not in_lane(world, aid, area)
        t = create_task(world, aid, action.get("title"), action.get("description", ""),
                        area, action.get("priority", "medium"),
                        action.get("depends_on") or (), action.get("done_when", ""),
                        to_self and not full and not other_lane)
        if other_lane:
            owner = _lane_owner(world, area)
            mine = (f" and left it unassigned for {owner['name']}, whose area that is" if owner
                    else " and left it unassigned")
        elif full:
            mine = (f" and left it unassigned, since {name} already has "
                    f"{MAX_OPEN_TASKS_PER_AGENT} open tasks")
        else:
            mine = " and took it on" if t["assignee"] == aid else ""
        return (f"{name} added task {t['id']} '{t['title']}' ({t['area']}, "
                f"{t['priority']} priority) to the project board{mine}.")

    if kind == "claim_task":
        if _has_full_plate(world, aid):
            raise BoardError(f"{name} already has {MAX_OPEN_TASKS_PER_AGENT} open tasks; "
                             f"finish one before taking on more")
        _check_lane(world, aid, get_task(world, action.get("task_id"))["area"])
        t = claim_task(world, aid, action.get("task_id"))
        return f"{name} took on task {t['id']} '{t['title']}' from the project board."

    if kind == "update_status":
        status = action.get("status")
        if status in NOTE_REQUIRED:
            _required_note(action, NOTE_REQUIRED[status])
        _check_review_habits(world, aid, action.get("task_id"), status)
        t = update_status(world, aid, action.get("task_id"), status,
                          action.get("note", ""))
        return (f"{name} moved task {t['id']} '{t['title']}' to "
                f"{t['status'].replace('_', ' ')} on the project board.")

    if kind == "assign_task":
        to = (action.get("assignee") or "").strip().lower()
        to_self = to in (aid, name.lower(), name.split()[0].lower())
        note = "" if to_self else _required_note(
            action, "tell the teammate what the task needs and why it is theirs")
        check_note_claims(world, note)
        target = next((a["id"] for a in world["agents"].values()
                       if to in (a["id"], a["name"].lower(), a["name"].split()[0].lower())), None)
        if target is not None:
            task_area = get_task(world, action.get("task_id"))["area"]
            if not in_lane(world, target, task_area):
                who = "you" if target == aid else _agent_name(world, target)
                raise BoardError(f"{task_area} work isn't {who}'s area"
                                 + ("" if target == aid else "; hand it to whoever owns that area"))
        t = assign_task(world, aid, action.get("task_id"), action.get("assignee"), note)
        line = (f"{name} assigned task {t['id']} '{t['title']}' to "
                f"{_agent_name(world, t['assignee'])} on the project board")
        return f"{line}, noting: \"{note}\"." if note else line + "."

    if kind == "edit_task":
        deps = action.get("depends_on")
        if isinstance(deps, str):
            deps = [deps]
        t, changed = edit_task(world, aid, action.get("task_id"), deps,
                               action.get("priority"), action.get("description"),
                               action.get("done_when"))
        return f"{name} updated the {', '.join(changed)} of task {t['id']} '{t['title']}' on the project board."

    if kind == "approve_task":
        t = approve_task(world, aid, action.get("task_id"), action.get("note", ""))
        return (f"{name} approved {_agent_name(world, t['assignee'])}'s task {t['id']} "
                f"'{t['title']}' on the project board.")

    if kind == "add_note":
        check_note_claims(world, action.get("note") or "")
        check_note_is_for_teammates(world, agent, action.get("note") or "")

    if kind == "add_note" and not action.get("task_id"):
        n = add_board_note(world, aid, action.get("note"))
        return f"{name} left a note on the project board: {n['note']}."

    if kind == "add_note":
        t = add_note(world, aid, action.get("task_id"), action.get("note"))
        return f"{name} left a note on task {t['id']} '{t['title']}': {_clean(action.get('note'))}."

    raise BoardError(f"unknown action {kind!r}")


def apply_actions(world, agent, actions, limit=MAX_ACTIONS_PER_VISIT):
    """Apply what the LLM proposed. Bad actions are skipped, not fatal.
    Returns (memory lines for what happened, refusals)."""
    done, refused = [], []
    start_settled = False          # one attempt to start work per visit
    for action in (actions or [])[:limit]:
        if not isinstance(action, dict):
            continue
        starting = (action.get("action") == "update_status"
                    and action.get("status") == "in_progress")
        if starting and start_settled:
            # Already started a task (or hit the WIP limit) this visit: further
            # starts can only be refused the same way, three times over.
            log.debug(f"{agent['name']}: skipped starting {action.get('task_id')} "
                      f"(one start per visit)")
            continue
        try:
            done.append(apply_action(world, agent, action))
            start_settled = start_settled or starting
        except NoChange as e:
            log.debug(f"{agent['name']}: {action.get('action')} skipped ({e})")
        except BoardError as e:
            refused.append(f"{action.get('action')}: {e}")
            start_settled = start_settled or (starting and "already working on" in str(e))
        except (TypeError, AttributeError, ValueError) as e:
            # Malformed fields from the model (a number where a string belongs, ...)
            refused.append(f"{action.get('action')}: malformed action ({e})")
    return done, merge_refusals(refused)


def merge_refusals(refused):
    """One refusal per kind of mistake per visit. Adam tried to approve T-2,
    T-3 and T-4 (none in review) and got three refusals and three memories of
    the same lesson; now it is one: "approve_task: T-2 is not in review (same
    for T-3, T-4)". Refusals that differ in more than the task id stay apart."""
    merged, order = {}, []
    for line in refused:
        key = TASK_REF.sub("T-#", line)
        if key not in merged:
            merged[key] = (line, [])
            order.append(key)
            continue
        first, more = merged[key]
        ids = [i for i in task_ids_in(line) if i not in task_ids_in(first)]
        more.extend(ids)
    out = []
    for key in order:
        first, more = merged[key]
        out.append(f"{first} (same for {', '.join(dict.fromkeys(more))})" if more else first)
    return out


def _role_context(agent, retriever, limit=6):
    """Who this agent is on the team and what they care about, from memory."""
    if retriever is None:
        return ""
    name = agent["name"]
    queries = [f"{name}'s role and responsibilities on the team",
               f"the team's goal for the company website"]
    seen, lines = set(), []
    for q in queries:
        for r in retriever.retrieve(q, n=4, touch=False):
            d = r["memory"].description
            if d not in seen:
                seen.add(d)
                lines.append(f"- {d}")
    return "\n".join(lines[:limit])


def _progress_context(agent, world, retriever, per_task=3, limit=10):
    """What the agent has actually done on each of their open tasks, from memory,
    so status changes at the board reflect real work."""
    tasks = open_tasks_for(world, agent["id"])
    if retriever is None or not tasks:
        return ""
    seen, lines = set(), []
    for t in tasks:
        query = f"{agent['name']}'s progress working on {t['title']}"
        for r in retriever.retrieve(query, n=per_task + 2, touch=False):
            d = r["memory"].description
            # Board lines describe the board, not the work.
            if d in seen or "project board" in d:
                continue
            seen.add(d)
            lines.append(f"- {d}")
            if len(lines) >= limit:
                return "\n".join(lines)
    return "\n".join(lines)


def build_prompt(agent, world, activity, summary, role_context, progress="", commitments=()):
    project = get_project(world)
    others = ", ".join(a["name"] for a in world["agents"].values() if a["id"] != agent["id"])
    return f"""
    {summary}
{persona.style(agent["id"], indent="    ")}
    What {agent['name']} knows about the team and the project:
    {role_context or '- (nothing specific)'}

    {agent['name']} is at the project board, about to: {activity}
    {agent['name']}'s area: {agent.get('area') or 'general'}
    Teammates: {others}

    Project: {project['name']}
    Goal: {project['goal']}
    {project['description']}

    The board right now:
    {board_text_for_prompt(world)}

    What stands out on the board for {agent['name']}:
{chr(10).join("    - " + line for line in status_for_agent(world, agent["id"]))}

    What {agent['name']} has been doing lately:
    {progress or '- (nothing recorded yet)'}
    (If a memory or a conversation says a task is approved, done or unblocked and
    the board above says otherwise, the board is right.)
{_commitments_block(agent, commitments, world)}
    What does {agent['name']} do at the board right now? Choose 0 to {MAX_ACTIONS_PER_VISIT} actions,
    then give the reason in one short sentence:
    - update_status: keep {agent['name']}'s own tasks current, based on what
      {agent['name']} has actually been doing: in_progress once work has started,
      blocked only when work already started got stuck (note: what is blocking it;
      a task that hasn't started and must wait for another gets that task in its
      depends_on through edit_task instead), review
      when the work is finished (note: what the reviewer should check), and done
      only after a teammate has approved it.
    - create_task: add work the goal needs that is not on the board yet. Give a short
      title, a one-sentence description, an area, a priority, depends_on, and
      done_when (how the team knows it is done). depends_on lists only tasks whose
      result this one truly can't start without; leave it empty when the work can
      begin on its own (a page layout can be built against mock data before the
      API exists). Set assign_to_self to true only if it fits {agent['name']}'s role.
    - claim_task: take an unassigned task that fits {agent['name']}'s role.
    - assign_task: give an unassigned task (or one of {agent['name']}'s own) to the
      teammate who should do it. Set task_id, assignee, and a note telling them
      what the task needs and why it is theirs.
    - edit_task: change an unassigned or own task's depends_on, priority,
      description or done_when (e.g. drop a dependency the task doesn't really need).
    - approve_task: sign off on a teammate's task that is in review, once
      {agent['name']} has looked it over or talked it through with them.
    - add_note: leave a note a teammate can act on: a question, a decision, a
      blocker, or what someone needs to know before starting. Set task_id to
      put it on a task, or leave task_id out for a note on the board itself.
      Never a note that only says what {agent['name']} is doing right now.

    Rules:
    - Do what {agent['name']} promised teammates first, if anything.
    - If {agent['name']}'s area has no tasks yet, add the work it needs.
    - Update {agent['name']}'s own tasks if their status is out of date.
    - Only add tasks that move the team toward the goal, and never duplicate a task.
    - A task is something to build or ship (a page, an endpoint, a deployment),
      never a meeting or a review; reviews happen with approve_task.
    - {agent['name']} works on one task at a time: finish the current one (review)
      or, if it is truly stuck, mark it blocked before starting the next. Tasks
      not started yet just stay in todo.
    - Only change a status when it actually changed; leave it alone otherwise.
    - Review is for finished work. A task in review stays there until a teammate
      approves it or asks for changes in a note.
    - {agent['name']} only takes on tasks in their own area ({agent.get('area') or 'any'});
      work in another area is added unassigned, for the teammate whose area it is.
      A teammate's task is theirs: {agent['name']} can review or approve it, or leave a
      note on it (a question, or an offer to work on it together), never move it.
    - Take a task on (assign_to_self, claim_task) only if {agent['name']} will work on
      it soon; leave the rest unassigned so whoever is free can pick it up.
      {agent['name']} can hold at most {MAX_OPEN_TASKS_PER_AGENT} open tasks.
    - Keep each task to a few hours of work: split a big area into separate
      tasks ("auth endpoints", "contact form endpoint", "deploy the API") instead
      of one "build the whole backend" task.
    - Do only what the current activity calls for; doing nothing is fine.
    """


def _refusal_memory(name, refusal):
    """'claim_task: T-1 is already claimed by Adam Beck' ->
    'Adam Beck tried to claim a task on the project board but couldn't: T-1 is ...'"""
    kind, _, reason = refusal.partition(": ")
    what = {"create_task": "add a task", "claim_task": "take on a task",
            "assign_task": "assign a task", "update_status": "move a task",
            "edit_task": "edit a task", "add_note": "leave a note",
            "approve_task": "approve a task"}.get(kind, "change the board")
    return f"{name} tried to {what} on the project board but couldn't: {reason}."


def promised_task_steps(world, agent, commitments):
    """What a promise means for the board, when it names the agent's own tasks.

    "I'll check the accessibility for T-5" never says "start T-5", so the model
    used to start the first task on the list instead. Only one task can be in
    progress, and a task that waits on unfinished work can't start at all, so:
    the first task that can start gets "start it" (setting aside the one in
    progress if needed), tasks that are waiting say so and how to unblock them,
    and any other promised task waits its turn."""
    if world is None:
        return []
    name, aid = agent["name"], agent["id"]
    tasks = {t["id"]: t for t in get_project(world)["tasks"]}
    busy = [t["id"] for t in tasks.values()
            if t["assignee"] == aid and t["status"] == "in_progress"]
    steps, seen, start_given = [], set(), False
    for text in commitments:
        for tid in task_ids_in(text):
            task = tasks.get(tid)
            if task is None or tid in seen or task["assignee"] not in (None, aid):
                continue
            seen.add(tid)
            if task["status"] != "todo":
                continue                      # already started: nothing to switch
            blockers = _unmet_dependencies(world, task)
            if blockers:
                deps = ", ".join(blockers)
                steps.append(
                    f"{tid} waits on {deps}, which {'is' if len(blockers) == 1 else 'are'} "
                    f"not done, so {tid} can't start yet. If {tid} doesn't really need "
                    f"{deps}, drop that dependency with edit_task and then start {tid}; "
                    f"otherwise leave {tid} in todo.")
                continue
            if start_given:
                steps.append(f"{tid} comes after that: one task at a time, so leave it in todo for now.")
                continue
            start_given = True
            others = [b for b in busy if b != tid]
            if others:
                cur = others[0]
                steps.append(
                    f"The promise is about {tid}, so {tid} is the task to work on. "
                    f"{cur} is in progress: move {cur} to review if it is finished, "
                    f"or back to todo if {name} hasn't really worked on it yet, "
                    f"then move {tid} to in progress.")
            else:
                steps.append(f"The promise is about {tid}: move {tid} to in progress "
                             f"(not another task).")
    return steps


def _commitments_block(agent, commitments, world=None):
    if not commitments:
        return ""
    lines = "\n".join(f"    - {c}" for c in commitments)
    steps = "\n".join(f"    - {s}" for s in promised_task_steps(world, agent, commitments))
    steps = f"\n    What that means on the board:\n{steps}\n" if steps else ""
    return f"""
    {agent['name']} promised teammates to do this on the board:
{lines}
{steps}"""


_START_STEP = re.compile(r"\bstart (T-\d+)(?: \(move it to in progress\))?", re.IGNORECASE)
_APPROVE_STEP = re.compile(r"\b(?:review and approve|approve)\s+(T-\d+)(?:\s*'[^']*')?", re.IGNORECASE)


def _reconcile_approval(world, agent_id, activity):
    """A planned "review and approve T-1" when T-1 isn't in review (yet): the
    plan was written hours earlier, on a guess. Say what is actually there to
    do, so the visit doesn't end in a refused approval."""
    match = _APPROVE_STEP.search(activity or "")
    if not match:
        return activity
    try:
        task = get_task(world, match.group(1))
    except BoardError:
        return activity
    if task["status"] == "review" or task["assignee"] in (None, agent_id):
        return activity
    tid, status = task["id"], task["status"].replace("_", " ")
    replacement = (f"check on {tid} (it is {status}, not in review, so there is nothing "
                   f"to approve yet)")
    return activity[:match.start()] + replacement + activity[match.end():]


_QUOTED_TITLE = re.compile(r"\b(T-\d+)(\s*[:(]?\s*)(['\"‘“])([^'\"’”]{2,100})"
                           r"(['\"’”])", re.IGNORECASE)


def fix_task_refs(world, text):
    """A task id next to a title that isn't that task's is a wrong id: the
    model meant other work. A replan wrote "work on T-2 'Implement and secure
    user authentication endpoints'" (T-2 was Eve's home page); putting T-2's
    real title there sent Adam to sketch Eve's home page. So the id goes to the
    task whose title it is, or is dropped and the described work stays. A
    shortened title ("Set up backend API") still counts as the same task."""
    if not world or "project" not in world:
        return text
    tasks = get_project(world)["tasks"]

    def words(t):
        return {w for w in _normalize(t).split() if len(w) > 2}

    def fix(m):
        tid, quoted = m.group(1).upper(), words(m.group(4))
        real = next((t for t in tasks if t["id"] == tid), None)
        if real is None or not quoted or quoted <= words(real["title"]):
            return m.group(0)
        meant = [t for t in tasks if quoted <= words(t["title"])]
        if len(meant) == 1:
            return f"{meant[0]['id']}{m.group(2)}{m.group(3)}{m.group(4)}{m.group(5)}"
        return f"{m.group(3)}{m.group(4)}{m.group(5)}"

    return _QUOTED_TITLE.sub(fix, text or "")


_START_WORK = re.compile(
    r"\b(?:start|begin|switch|move on|pick up|take on|jump|work)\w*\b(?:\s+\w+){0,4}?\s+"
    r"(?:task\s+)?(T-\d+)", re.IGNORECASE)


def start_conflict(world, agent_id, text):
    """Why a reaction to start working on a task can't happen now, or "".

    "Start working on task T-5" while T-1 was in progress replaced Adam's plan
    with T-5 work the board refused to start (one task at a time), and his
    work and the board disagreed for the rest of the morning."""
    if not world or "project" not in world:
        return ""
    match = _START_WORK.search(text or "")
    if not match:
        return ""
    try:
        task = get_task(world, match.group(1))
    except BoardError:
        return ""
    if task["status"] != "todo" or task["assignee"] not in (None, agent_id):
        return ""
    blockers = _unmet_dependencies(world, task)
    if blockers:
        return f"{task['id']} waits on {', '.join(blockers)}"
    busy = [t["id"] for t in get_project(world)["tasks"]
            if t["assignee"] == agent_id and t["status"] == "in_progress" and t["id"] != task["id"]]
    if len(busy) >= WIP_LIMIT:
        return f"{busy[0]} is still in progress (one task at a time)"
    return ""


def teammate_task(world, agent_id, text):
    """The first task mentioned in `text` that belongs to a teammate, or None."""
    if not world or "project" not in world:
        return None
    tasks = {t["id"]: t for t in get_project(world)["tasks"]}
    for tid in task_ids_in(text):
        task = tasks.get(tid)
        if task and task["assignee"] and task["assignee"] != agent_id \
                and task["assignee"] in world["agents"]:
            return task
    return None


def reconcile_activity(world, agent_id, activity):
    """Bring a planned board step up to date with the board as it is now:
    start steps for tasks that already started or can't start yet, and
    approvals of tasks that aren't in review."""
    return _reconcile_start(world, agent_id, _reconcile_approval(world, agent_id, activity))


def _reconcile_start(world, agent_id, activity):
    """Checkpoints like "start T-1 (move it to in progress)" are written into the
    plan when it is made. If the task was started in the meantime, the step
    would ask the model to do something already done, and it picks the next
    status instead (that is how T-1 went to review at 8:00 after 50 minutes).
    If another task is still in progress, the step can't happen yet (WIP limit),
    so it says so instead of collecting refusals."""
    match = _START_STEP.search(activity or "")
    if not match:
        return activity
    try:
        task = get_task(world, match.group(1))
    except BoardError:
        return activity
    tid = task["id"]
    if task["assignee"] not in (None, agent_id):
        # Adam's plan had "start T-2 (move it to in progress)" for Eve's T-2.
        replacement = (f"check on {tid} ({owner_name(world, task)}'s task: only they move it "
                       f"on the board)")
        return activity[:match.start()] + replacement + activity[match.end():]

    blockers = _unmet_dependencies(world, task) if task["status"] == "todo" else []
    if blockers:
        deps = ", ".join(blockers)
        replacement = (f"check on {tid}: it waits on {deps}, which isn't done, so it can't "
                       f"start yet (drop the dependency with edit_task only if {tid} "
                       f"doesn't really need {deps})")
        return activity[:match.start()] + replacement + activity[match.end():]
    if task["status"] == "in_progress":
        replacement = f"report progress on {tid} (already in progress)"
    elif task["status"] in ("review", "done"):
        replacement = f"check on {tid} (it is {task['status']})"
    elif task["status"] == "todo":
        busy = [t for t in get_project(world)["tasks"]
                if t["assignee"] == agent_id and t["status"] == "in_progress" and t["id"] != tid]
        if not busy:
            return activity
        other = busy[0]["id"]
        replacement = (f"check whether {other} is finished: if it is, move {other} to review "
                       f"and start {tid}; if not, keep {other} in progress and leave {tid} for later")
    else:
        return activity
    return activity[:match.start()] + replacement + activity[match.end():]


def interact(agent, world, activity, planner=None, retriever=None, memory_stream=None):
    """
    One board visit: ask the LLM what the agent does, apply it, and remember it.
    Returns the list of memory lines for what changed.
    """
    summary = planner.generate_agent_summary(agent) if planner else agent["name"]
    retriever = retriever or getattr(planner, "memory_retriever", None)
    memory_stream = memory_stream or getattr(planner, "memory_stream", None)

    current = reconcile_activity(world, agent["id"], activity)
    if current != activity:
        log.debug(f"{agent['name']} board step brought up to date: {activity!r} -> {current!r}")
        activity = current

    promised = commitments_for(agent["id"])
    prompt = build_prompt(agent, world, activity, summary, _role_context(agent, retriever),
                          _progress_context(agent, world, retriever), promised)
    try:
        raw = llm.generate(prompt, format=action_schema(world),
                           options={**CTX, "num_predict": 1000, "temperature": 0.3})
        result = _parse_board_reply(raw)
    except Exception as e:
        log.warn(f"project board call failed for {agent['name']} ({e})")
        return []

    done, refused = apply_actions(world, agent, result.get("actions"))
    _after_visit(agent["id"], acted=bool(done))
    log.debug(f"{agent['name']} at board: {result.get('reason', '')}")
    for line in done:
        log.event(f"📋 {line}")
    for line in refused:
        log.event(f"📋✗ {agent['name']} couldn't {line}")

    # A refusal is something the agent learns ("T-1 needs a teammate to approve
    # it first"), which is exactly what should send them to talk to someone.
    remembered = done + [_refusal_memory(agent["name"], line) for line in refused
                         if "malformed action" not in line]     # model noise, not a lesson
    if remembered and memory_stream is not None:
        memory_stream.add_observations(remembered)
    return done
