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
      "created_at": "7:35 am",
      "updated_at": "7:35 am",
      "history": [{"time": "7:35 am", "agent": "adam", "event": "created"}]
    }

How agents change it (paper: an action changes the state of the object being
used): when an agent is interacting with the board, ``interact`` makes one
schema-constrained LLM call that returns a list of board actions. Those actions
are validated and applied here in plain Python, and each one is written to the
acting agent's memory stream as an observation. Other agents find out by
looking at the board, or by being told in conversation.
"""

import json
import re

import clock
import llm
import log
from llm import CTX

BOARD_OBJECT = "project_board"

STATUSES = ("todo", "in_progress", "blocked", "review", "done")
PRIORITIES = ("high", "medium", "low")
AREAS = ("backend", "frontend", "design", "devops", "testing", "general")
ACTIONS = ("create_task", "claim_task", "update_status", "add_note")

MAX_ACTIONS_PER_VISIT = 4     # keeps one board visit from flooding the board
MAX_TASKS_SHOWN = 12          # observations get long fast; done tasks are summarized


class BoardError(ValueError):
    """An action the board refuses (unknown task, duplicate title, ...)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now():
    return clock.now().strftime("%I:%M %p").lstrip("0").lower()


def _normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).strip()


def _clean(text, limit=200):
    return " ".join((text or "").split())[:limit].strip().rstrip(".")


def get_project(world):
    """The project record, with defaults filled in for older world files."""
    project = world.setdefault("project", {})
    project.setdefault("name", "Project")
    project.setdefault("goal", "")
    project.setdefault("description", "")
    project.setdefault("status", "not_started")
    project.setdefault("tasks", [])
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
        "created_at": now,
        "updated_at": now,
        "history": [{"time": now, "agent": agent_id, "event": "created"}],
    }


def create_task(world, agent_id, title, description="", area="general",
                priority="medium", depends_on=(), done_when="", assign_to_self=False):
    project = get_project(world)
    title = _clean(title, 80)
    if not title:
        raise BoardError("a task needs a title")

    key = _normalize(title)
    for existing in project["tasks"]:
        if _normalize(existing["title"]) == key:
            raise BoardError(f"{existing['id']} '{existing['title']}' is already on the board")

    area = area if area in AREAS else "general"
    priority = priority if priority in PRIORITIES else "medium"
    known = {t["id"] for t in project["tasks"]}
    deps = [d.strip().upper() for d in depends_on or () if d and d.strip().upper() in known]

    task = new_task(_next_id(project), agent_id, title, _clean(description, 300),
                    area, priority, deps, _clean(done_when, 200))
    if assign_to_self:
        task["assignee"] = agent_id
        _log_event(task, agent_id, "claimed")
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
    sync_board_object(world)
    return task


def update_status(world, agent_id, task_id, status, note=""):
    task = get_task(world, task_id)
    if status not in STATUSES:
        raise BoardError(f"unknown status {status!r}")
    if task["status"] == status:
        raise BoardError(f"{task['id']} is already {status}")
    if task["assignee"] and task["assignee"] != agent_id and status != "review":
        raise BoardError(f"{task['id']} belongs to {_agent_name(world, task['assignee'])}")
    if status == "in_progress":
        blockers = _unmet_dependencies(world, task)
        if blockers:
            raise BoardError(f"{task['id']} is waiting on {', '.join(blockers)}")
        if not task["assignee"]:
            task["assignee"] = agent_id              # starting work claims it

    old = task["status"]
    task["status"] = status
    _log_event(task, agent_id, f"{old} -> {status}", _clean(note) or None)
    refresh_project_status(world)
    sync_board_object(world)
    return task


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
    return line + "."


def board_observations(world):
    """What anyone standing at the board can read off it."""
    project = get_project(world)
    lines = [f"The project board shows the team goal: {project['goal']}."]
    tasks = project["tasks"]
    if not tasks:
        lines.append("The project board has no tasks on it yet.")
        return lines

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
    return lines


def board_text_for_prompt(world):
    """Fuller view for the agent who is actually working at the board."""
    project = get_project(world)
    if not project["tasks"]:
        return "(the board is empty)"
    rows = []
    for t in project["tasks"]:
        who = _agent_name(world, t["assignee"]) if t["assignee"] else "unassigned"
        deps = f" depends on {', '.join(t['depends_on'])};" if t["depends_on"] else ""
        rows.append(f"- {t['id']} [{t['status']}] {t['title']} ({t['area']}, "
                    f"{t['priority']}, {who});{deps} done when: {t['done_when'] or 'n/a'}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Acting at the board
# ---------------------------------------------------------------------------
def action_schema(world):
    task_ids = [t["id"] for t in get_project(world)["tasks"]]
    task_id = {"type": "string", "enum": task_ids} if task_ids else {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
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
                        "status": {"type": "string", "enum": list(STATUSES)},
                        "note": {"type": "string"},
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["reason", "actions"],
    }


def apply_action(world, agent, action):
    """Apply one action dict. Returns a memory line, or raises BoardError."""
    kind = action.get("action")
    aid, name = agent["id"], agent["name"]

    if kind == "create_task":
        t = create_task(world, aid, action.get("title"), action.get("description", ""),
                        action.get("area", "general"), action.get("priority", "medium"),
                        action.get("depends_on") or (), action.get("done_when", ""),
                        bool(action.get("assign_to_self")))
        mine = " and took it on" if t["assignee"] == aid else ""
        return (f"{name} added task {t['id']} '{t['title']}' ({t['area']}, "
                f"{t['priority']} priority) to the project board{mine}.")

    if kind == "claim_task":
        t = claim_task(world, aid, action.get("task_id"))
        return f"{name} took on task {t['id']} '{t['title']}' from the project board."

    if kind == "update_status":
        t = update_status(world, aid, action.get("task_id"), action.get("status"),
                          action.get("note", ""))
        return (f"{name} moved task {t['id']} '{t['title']}' to "
                f"{t['status'].replace('_', ' ')} on the project board.")

    if kind == "add_note":
        t = add_note(world, aid, action.get("task_id"), action.get("note"))
        return f"{name} left a note on task {t['id']} '{t['title']}': {_clean(action.get('note'))}."

    raise BoardError(f"unknown action {kind!r}")


def apply_actions(world, agent, actions, limit=MAX_ACTIONS_PER_VISIT):
    """Apply what the LLM proposed. Bad actions are skipped, not fatal.
    Returns (memory lines for what happened, refusals)."""
    done, refused = [], []
    for action in (actions or [])[:limit]:
        if not isinstance(action, dict):
            continue
        try:
            done.append(apply_action(world, agent, action))
        except BoardError as e:
            refused.append(f"{action.get('action')}: {e}")
    return done, refused


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


def build_prompt(agent, world, activity, summary, role_context):
    project = get_project(world)
    others = ", ".join(a["name"] for a in world["agents"].values() if a["id"] != agent["id"])
    return f"""
    {summary}

    What {agent['name']} knows about the team and the project:
    {role_context or '- (nothing specific)'}

    {agent['name']} is at the project board, about to: {activity}
    Teammates: {others}

    Project: {project['name']}
    Goal: {project['goal']}
    {project['description']}

    The board right now:
    {board_text_for_prompt(world)}

    What does {agent['name']} do at the board right now? Choose 0 to {MAX_ACTIONS_PER_VISIT} actions:
    - create_task: add work the goal needs that is not on the board yet. Give a short
      title, a one-sentence description, an area, a priority, depends_on (ids of
      tasks that must finish first), and done_when (how the team knows it is done).
      Set assign_to_self to true only if it fits {agent['name']}'s role.
    - claim_task: take an unassigned task that fits {agent['name']}'s role.
    - update_status: move one of {agent['name']}'s own tasks (todo, in_progress,
      blocked, review, done). Add a short note.
    - add_note: leave a short note on a task, e.g. a question for a teammate.

    Rules:
    - Only add tasks that move the team toward the goal, and never duplicate a task.
    - Prefer tasks in {agent['name']}'s own area; leave other areas for teammates.
    - Keep tasks small enough for one person to finish in a day or two.
    - Do only what the current activity calls for; doing nothing is fine.
    """


def interact(agent, world, activity, planner=None, retriever=None, memory_stream=None):
    """
    One board visit: ask the LLM what the agent does, apply it, and remember it.
    Returns the list of memory lines for what changed.
    """
    summary = planner.generate_agent_summary(agent) if planner else agent["name"]
    retriever = retriever or getattr(planner, "memory_retriever", None)
    memory_stream = memory_stream or getattr(planner, "memory_stream", None)

    prompt = build_prompt(agent, world, activity, summary, _role_context(agent, retriever))
    try:
        raw = llm.generate(prompt, format=action_schema(world),
                           options={**CTX, "num_predict": 700, "temperature": 0.3})
        result = json.loads(raw)
    except Exception as e:
        log.warn(f"project board call failed for {agent['name']} ({e})")
        return []

    done, refused = apply_actions(world, agent, result.get("actions"))
    log.debug(f"{agent['name']} at board: {result.get('reason', '')}")
    for line in done:
        log.event(f"📋 {line}")
    for line in refused:
        log.debug(f"board refused {agent['name']}'s {line}")

    if done and memory_stream is not None:
        memory_stream.add_observations(done)
    return done
