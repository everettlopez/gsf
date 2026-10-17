"""
snapshot.py - the simulation, as data for the website the agents build.

The team's website is a window into the team's own working days: who they
are, where they are and what they are doing, the board, their talks, what
they think about it all. The agents build the window; this module supplies
what it shows. build() gathers the world, the personas, the memory streams
and the chronicle (chronicle.py) into one JSON document, and export() commits
it to the sandbox's main as data/snapshot.json, so every task branch started
after that has the latest. main.py exports when the run starts, every hour of
sim time and at the end; scripts/preview_site.py writes a fresh one into the
preview.

The document's shape is the contract the site is built against (SCHEMA, and
skills/shared/SKILL.md, which describes it to the hands). Change both
together, and bump SCHEMA when a change would break a page.
"""

import json
from datetime import datetime
from pathlib import Path

import clock
import persona

SCHEMA = 1
PATH_IN_SANDBOX = "data/snapshot.json"

RECENT_MEMORIES = 30                # per agent
RECENT_REFLECTIONS = 30             # per agent
MAX_TIMELINE = 4000                 # steps x agents; a week of 5-minute steps for two
MAX_EVENTS = 1500


def _clock_text(dt):
    return dt.strftime("%I:%M %p").lstrip("0").lower()


def _day_of(created_at, start_date):
    """A memory's run day (1, 2, ...) and time, from its timestamp."""
    try:
        dt = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return None, None
    day = (dt.date() - start_date).days + 1 if start_date else None
    return (day if day and day >= 1 else None), _clock_text(dt)


def _plan(chunks):
    return [{"start": c.get("start_time"), "minutes": c.get("duration"),
             "activity": c.get("activity"), "location": c.get("location")}
            for c in chunks or []]


def _memories(stream, start_date, kind, limit):
    if stream is None:
        return []
    picked = [m for m in stream.memories if m.memory_type == kind][-limit:]
    out = []
    for m in picked:
        day, time = _day_of(m.created_at, start_date)
        out.append({"day": day, "time": time, "text": m.description,
                    "importance": m.importance})
    return out


def _agent(agent, stream, plan_chunks, start_date):
    try:
        p = persona.load(agent["id"])
    except FileNotFoundError:
        p = {}
    observations = _memories(stream, start_date, "observation", RECENT_MEMORIES)
    return {
        "id": agent["id"],
        "name": agent["name"],
        "age": p.get("age"),
        "area": agent.get("area"),
        "office": agent.get("office"),
        "traits": p.get("innate_traits", []),
        "about": p.get("occupation", []),
        "work_style": p.get("work_style", []),
        "communication_style": p.get("communication_style", []),
        "aspirations": p.get("aspirations", []),
        "now": {"location": agent["current_location"], "action": agent["current_action"],
                "object": agent.get("interacting_with")},
        "plan": _plan(plan_chunks),
        "memories": observations,
        "reflections": _memories(stream, start_date, "reflection", RECENT_REFLECTIONS),
        "stats": {"memories": len(stream.memories) if stream else 0,
                  "reflections": sum(1 for m in stream.memories if m.memory_type == "reflection")
                  if stream else 0},
    }


def _rooms(world):
    rooms = []
    for room_id, room in world["locations"].items():
        rooms.append({
            "id": room_id,
            "name": room_id.replace("_", " ").title(),
            "state": room.get("state"),
            "objects": [{"name": name, "state": data.get("state")}
                        for name, data in room.get("objects", {}).items()],
            "occupants": [a["id"] for a in world["agents"].values()
                          if a["current_location"] == room_id],
        })
    return rooms


def build(world, chronicle=None, streams=None, plans=None, generated=None):
    """The snapshot document.

    world      the live world (agents, rooms, the board)
    chronicle  chronicle.data() or a saved one (chronicle.load()), or None
    streams    agent id -> MemoryStream, for memories and reflections
    plans      agent id -> the agent's current broad plan chunks; without
               them, the plan each agent started the latest day with
    generated  {"day", "day_name", "time"} the snapshot is as of; default
               the sim clock (outside a run: as_of(chronicle))"""
    chronicle = chronicle or {}
    streams = streams or {}
    try:
        start_date = datetime.fromisoformat(chronicle["start_date"]).date()
    except (KeyError, TypeError, ValueError):
        start_date = clock.now().date()
    days = chronicle.get("days") or []
    if plans is None:
        latest = days[-1]["plans"] if days else {}
        plans = {aid: [{"activity": a} for a in acts] for aid, acts in latest.items()}

    project = world.get("project", {})
    return {
        "schema": SCHEMA,
        "generated": generated or {"day": clock.day(), "day_name": clock.day_name(),
                                   "time": _clock_text(clock.now())},
        "company": {
            "name": project.get("company", {}).get("name", ""),
            "email": project.get("company", {}).get("email", ""),
            "project": project.get("name", ""),
            "goal": project.get("goal", ""),
            "status": project.get("status", ""),
        },
        "days": [{"day": d["day"], "name": d["name"]} for d in days],
        "agents": [_agent(a, streams.get(a["id"]), plans.get(a["id"]), start_date)
                   for a in world["agents"].values()],
        "rooms": _rooms(world),
        "board": {"tasks": project.get("tasks", []), "notes": project.get("notes", [])},
        "timeline": (chronicle.get("steps") or [])[-MAX_TIMELINE:],
        "events": (chronicle.get("events") or [])[-MAX_EVENTS:],
        "conversations": chronicle.get("conversations") or [],
    }


def as_of(chronicle):
    """When a saved run's chronicle ends: its last step's day and time."""
    steps = (chronicle or {}).get("steps") or []
    if not steps:
        return None
    last = steps[-1]
    return {"day": last["day"], "day_name": clock.day_name(last["day"]), "time": last["time"]}


def from_disk(world_path=None):
    """A snapshot of the office as the last run left it (world.json, the
    memory streams, data/chronicle.json), for use outside a run."""
    import chronicle as chronicle_module
    import world as world_module
    from memory_stream import MemoryStream

    world = json.loads(Path(world_path or world_module.WORLD_PATH).read_text(encoding="utf-8"))
    saved = chronicle_module.load()
    streams = {aid: MemoryStream(aid) for aid in world["agents"]}
    return build(world, saved, streams=streams, generated=as_of(saved))


def dumps(snapshot):
    return json.dumps(snapshot, indent=1, ensure_ascii=False) + "\n"


def write(snapshot, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(snapshot), encoding="utf-8", newline="\n")


def export(snapshot, root=None, message=None):
    """Commit the snapshot (and the other host-owned files) to the sandbox's
    main. Returns True if main changed. Raises workspace.WorkspaceError if git
    fails."""
    import workspace

    when = snapshot["generated"]
    return workspace.update_host_files(
        {PATH_IN_SANDBOX: dumps(snapshot)}, root=root,
        message=message or f"Snapshot of the office: {when['day_name']} {when['time']}")
