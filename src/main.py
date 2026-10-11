"""
main.py - Generative Agents simulation driver (Adam + Eve)

Per step, the loop runs three phases for BOTH agents (so they act
"simultaneously" and don't see each other's mid-step changes):

    1. PERCEIVE  - build observations, store only the NEW ones in memory
    2. DECIDE    - continue the plan (1) or react (2); the LLM is only asked
                   when there is something new to react to, and it returns a
                   short reaction that drives the replan (paper, section 4.3.1)
    3. ACT       - update agent/world state (1) or regenerate the plan (2)

Planning follows the paper's top-down scheme:

    broad plan  ->  hour chunks  ->  5-15 minute chunks

By default each level is expanded lazily (and cached) when the agent reaches
it. Set EAGER_DECOMPOSITION (or PAPER_STRICT) to decompose the whole plan up
front, exactly as the paper describes. Every level is saved to the memory
stream as a plan either way.

On a reaction, only the plan from the current time onward is regenerated;
what already happened today is kept (the chunk in progress is cut at "now").

Reflection (paper section 4.2) runs twice per simulated day, at the times in
REFLECTION_TIMES. Each agent reflects on what it has experienced since its last
reflection, and the resulting insights go back into its memory stream.

Project board (see project_board.py): when an agent takes on or finishes a
task at the board, the rest of their day is replanned around their open tasks
once they step away from it, so later work blocks name the task and include
return trips to the board to keep it up to date.

Verbosity: set GA_LOG=quiet|info|debug (default info). See log.py.
"""

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import clock
import log
import project_board
from conversation import run_conversation
from memory_retriever import MemoryRetriever
from memory_stream import MemoryStream
from perceive import perceive_environment
from plan import DAY_END, Planner
from reflect import Reflector
from world import generate_interactions, load_world, save_world, update_agent_state


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
START_TIME = "7:00 am"
NUM_STEPS = 30
STEP_MINUTES = 5            # simulated minutes per step
STEP_DELAY_SECONDS = 0      # wall-clock padding per step (0 = run flat out)

# Reflection happens twice a day (paper 4.2: "two or three times a day").
# Each time is reached once per simulated day; times before START_TIME are skipped.
REFLECTION_TIMES = ["12:00 pm", "5:00 pm"]

# --- Efficiency shortcuts (NOT in the paper) --------------------------------

# Decompose the whole plan up front (paper) instead of when each block is
# reached. Costs roughly 100s per agent at startup on a local model.
EAGER_DECOMPOSITION = False

# Only ask the LLM "continue or react?" when something NEW was observed.
# With this True, an agent's own state lines (e.g. "Adam is currently at...")
# don't count, so agents don't "react" to the changes they just made themselves.
REACT_ONLY_TO_EXTERNAL = True

# If the agent is still on the same plan chunk and keeps going (decision 1),
# skip update_agent_state + generate_interactions (2+ LLM calls).
SKIP_UNCHANGED_CHUNKS = True

# After a react (replan), don't ask the LLM about reacting again for this many
# steps. Stops agents from replanning on nearly every step. A teammate walking
# into the room still gets asked about (see decide_phase), and a reaction that
# only moves the agent to another room doesn't start the cooldown.
REACT_COOLDOWN_STEPS = 6

# The same two agents don't start another conversation within this many
# minutes, unless the reaction names a specific task (T-3). Stops the loop of
# meeting -> commitment "review with Eve" -> meeting again.
CONVERSATION_COOLDOWN_MINUTES = 60

# Plan chunks this short are already at the paper's 5-15 minute grain, so they
# aren't sent to the LLM for minute-level decomposition (e.g. board steps).
SHORT_CHUNK_MINUTES = 15

# Step 1 sees the whole room as "new". Treat it as a baseline and don't react.
SKIP_REACT_ON_FIRST_STEP = True

# --- One switch to turn all the shortcuts off -------------------------------
# True = behave as the paper describes: decompose everything up front, and ask
# the react/continue question every step about everything currently perceived.
# Slow on a local model, but it's the baseline to compare the shortcuts against.
PAPER_STRICT = False

if PAPER_STRICT:
    EAGER_DECOMPOSITION = True
    REACT_ONLY_TO_EXTERNAL = False
    SKIP_UNCHANGED_CHUNKS = False
    REACT_COOLDOWN_STEPS = 0
    SKIP_REACT_ON_FIRST_STEP = False


# ---------------------------------------------------------------------------
# TIMING (where is the time going?)
# ---------------------------------------------------------------------------
timings = {}  # label -> (total_seconds, call_count)


@contextmanager
def timed(label):
    """Time a block, print it, and accumulate it for the end-of-run summary."""
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        total, count = timings.get(label, (0.0, 0))
        timings[label] = (total + dt, count + 1)
        log.debug(f"timing {label}: {dt:.1f}s")


def print_timing_summary():
    if not timings:
        return
    log.banner("TIMING SUMMARY (slowest first)")
    for label, (total, count) in sorted(timings.items(), key=lambda kv: -kv[1][0]):
        log.info(f"  {total:7.1f}s total | {count:3d} calls | {total / count:6.1f}s avg | {label}")


# ---------------------------------------------------------------------------
# PER-AGENT RUNTIME
# ---------------------------------------------------------------------------
@dataclass
class AgentRuntime:
    """Everything one agent needs, built around ONE shared MemoryStream."""
    agent: dict                      # same dict object as world["agents"][id]
    stream: MemoryStream
    retriever: MemoryRetriever
    planner: Planner
    reflector: Reflector
    reflected_at: set = field(default_factory=set)   # REFLECTION_TIMES already handled

    # The plan, at three levels of detail. The two caches are keyed by
    # (start_time, activity), so a replan can never hit a stale entry.
    broad_chunks: list = field(default_factory=list)
    hour_cache: dict = field(default_factory=dict)
    minute_cache: dict = field(default_factory=dict)

    prev_obs: list = field(default_factory=list)
    prev_company: set = field(default_factory=set)    # teammates in the room last step
    last_chunk: Optional[dict] = None
    last_react_step: int = -10_000
    override: Optional[dict] = None
    queued: Optional[dict] = None
    commitments: list = field(default_factory=list)   # promises already planned around

    # Per-step scratch space (filled in by the phases below)
    observations: list = field(default_factory=list)
    new_obs: list = field(default_factory=list)
    chunk: Optional[dict] = None
    status: str = ""
    decision: str = "1"
    reaction: str = ""

    @property
    def tag(self):
        return self.agent["id"].upper()


def make_runtime(agent_id, world):
    # Use the agent dict that lives inside world so there is a single source of
    # truth (previously load_agent() and load_world() gave two diverging copies).
    agent = world["agents"][agent_id]

    stream = MemoryStream(agent_id)
    retriever = MemoryRetriever(agent_id, stream)
    planner = Planner(agent, stream, retriever)

    return AgentRuntime(
        agent=agent,
        stream=stream,
        retriever=retriever,
        planner=planner,
        reflector=Reflector(agent, stream, retriever),
    )


# ---------------------------------------------------------------------------
# PLANNING
# ---------------------------------------------------------------------------
def expand_hours(rt, broad):
    """broad block -> hour chunks (cached, saved to the memory stream)."""
    p = rt.planner
    bkey = (broad["start_time"], broad["activity"])

    if bkey not in rt.hour_cache:
        with timed(f"{rt.tag} expand broad {bkey[0]} -> hours (LLM)"):
            hours = p.decompose_chunks([broad], rt.agent)
        if hours:
            p.create_plan(hours)
        # If the output didn't parse, fall back to the coarser chunk.
        rt.hour_cache[bkey] = hours or [broad]

    return rt.hour_cache[bkey]


def expand_minutes(rt, hour):
    """hour chunk -> 5-15 minute chunks (cached, saved to the memory stream)."""
    p = rt.planner
    hkey = (hour["start_time"], hour["activity"])

    if hkey not in rt.minute_cache and (hour.get("duration") or 60) <= SHORT_CHUNK_MINUTES:
        rt.minute_cache[hkey] = [hour]          # already 5-15 minutes: nothing to break down

    if hkey not in rt.minute_cache:
        with timed(f"{rt.tag} expand hour {hkey[0]} -> minutes (LLM)"):
            minutes = p.decompose_hour_chunks([hour], rt.agent)
        if minutes:
            p.create_plan(minutes)
        rt.minute_cache[hkey] = minutes or [hour]

    return rt.minute_cache[hkey]


def expand_plan(rt, chunks):
    """Eagerly decompose the given broad chunks all the way down (the paper's
    up-front version). Fills the same caches the lazy path uses."""
    for broad in chunks:
        for hour in expand_hours(rt, broad):
            expand_minutes(rt, hour)


def initial_planning(rt, world):
    """Day plan in broad strokes, then (optionally) the full decomposition."""
    with timed(f"{rt.tag} initial broad plan (LLM)"):
        daily_plan = rt.planner.generate_initial_plan(rt.agent, world)

    rt.broad_chunks = rt.planner.apply_schedule(
        rt.planner.create_chunks(daily_plan, rt.agent, world), world, START_TIME)
    rt.hour_cache, rt.minute_cache = {}, {}

    if not rt.broad_chunks:
        # No chunks means get_chunk_lazy always returns None.
        log.warn(f"{rt.tag}: initial plan produced no chunks. Daily plan was: {daily_plan}")
        return

    log.info(f"  {rt.tag} day plan:")
    for chunk in rt.broad_chunks:
        log.info(f"      {chunk['activity']}")

    rt.planner.create_plan(rt.broad_chunks)  # plans live in the memory stream

    if EAGER_DECOMPOSITION:
        expand_plan(rt, rt.broad_chunks)


def get_chunk_lazy(rt, current_time, world=None):
    """
    Return the minute-level chunk covering current_time.

    Anything not decomposed yet is expanded now (1 LLM call per level, then
    cached); with EAGER_DECOMPOSITION everything is already cached.
    """
    p = rt.planner

    while rt.override:
        now = p._parse_time(current_time)
        start = p._parse_time(rt.override["start_time"])
        end = start + timedelta(minutes=rt.override["duration"])
        if start <= now < end:
            return rt.override
        # Expired: promote the queued chunk (or clear). Loop re-checks it.
        rt.override, rt.queued = rt.queued, None


    broad = p.get_current_minute_chunk(current_time, rt.broad_chunks)
    if broad is None:
        return None
    if not p.covers(broad, current_time):
        # The broad plan itself has a gap (a replan ended a block at 11:20 and
        # the next starts at 12:30). Don't replay the last block that started.
        broad = gap_block(rt, current_time, world)
        if broad is None:
            return None

    hour = p.get_current_minute_chunk(current_time, expand_hours(rt, broad))
    if hour is None or not p.covers(hour, current_time):
        # The hour decomposition left a gap (the model returned too few hours).
        # Fill it from the broad block instead of replaying the last minute action
        # of an hour that already ended.
        hour = gap_hour(rt, broad, current_time)
        if hour is None:
            return None

    return p.get_current_minute_chunk(current_time, expand_minutes(rt, hour))


def gap_block(rt, current_time, world=None):
    """A broad block filling a gap in the plan: from where the last block ended
    to where the next one starts (or the end of the day). The agent carries on
    with the task they have in progress, or their own work if they have none."""
    p = rt.planner
    now = p._parse_time(current_time)
    ends, starts = [], []
    for c in rt.broad_chunks:
        start = p._parse_time(c.get("start_time"))
        if start is None:
            continue
        end = start + timedelta(minutes=c.get("duration") or 0)
        if end <= now:
            ends.append(end)
        elif start > now:
            starts.append(start)
    day_end = p._parse_time(DAY_END)
    gap_start = max(ends) if ends else now
    gap_end = min(starts + [day_end])
    if gap_end <= now:
        return None                                    # the workday is over

    working = [t for t in project_board.open_tasks_for(world, rt.agent["id"])
               if t["status"] == "in_progress"] if world else []
    what = (f"keep working on {working[0]['id']} '{working[0]['title']}'"
            if working else f"focus on {rt.agent['name']}'s own work")
    start_text, end_text = p._fmt(gap_start), p._fmt(gap_end)
    log.debug(f"{rt.tag} broad plan gap {start_text}-{end_text}: {what}")
    return {
        "activity": f"{what} in the office from {start_text} to {end_text}",
        "location": rt.agent.get("office"),
        "start_time": start_text,
        "duration": int((gap_end - gap_start).total_seconds() // 60),
        "target_agent": None,
        "board_work": False,
    }


def gap_hour(rt, broad, current_time):
    """An hour chunk covering `current_time`, cut from the broad block.

    It is anchored where the gap starts (the end of the last hour chunk), in
    whole hours, so every step inside the same gap gets the same chunk and it
    is decomposed only once (cached by start time + activity)."""
    p = rt.planner
    now = p._parse_time(current_time)
    block_start = p._parse_time(broad.get("start_time"))
    if now is None or block_start is None:
        return None
    block_end = block_start + timedelta(minutes=broad.get("duration") or 0)
    if now >= block_end:
        return None

    anchor = block_start
    for hour in expand_hours(rt, broad):
        hs = p._parse_time(hour.get("start_time"))
        if hs is not None and hs <= now:
            anchor = max(anchor, hs + timedelta(minutes=hour.get("duration") or 0))
    if anchor > now:
        anchor = block_start
    whole_hours = int((now - anchor).total_seconds() // 3600)
    start = anchor + timedelta(hours=whole_hours)
    minutes = int(min(timedelta(hours=1), block_end - start).total_seconds() // 60)

    start_text = start.strftime("%I:%M %p").lstrip("0").lower()
    activity = re.sub(r"^\d{1,2}:\d{2}\s*[ap]m:\s*", "", broad["activity"], flags=re.IGNORECASE)
    log.debug(f"{rt.tag} plan gap at {current_time}, filling from {start_text}: {activity}")
    return {
        "activity": f"{start_text}: {activity}",
        "location": broad.get("location"),
        "start_time": start_text,
        "duration": minutes,
        "target_agent": broad.get("target_agent"),
    }


# ---------------------------------------------------------------------------
# TIME / PARSING HELPERS
# ---------------------------------------------------------------------------
def step_time(current_time, minutes=STEP_MINUTES):
    current_dt = datetime.strptime(current_time.upper(), "%I:%M %p")
    new_dt = current_dt + timedelta(minutes=minutes)
    # Drop the zero-pad so times read "7:05 am", matching the plan's format.
    return new_dt.strftime("%I:%M %p").lstrip("0").lower()


# ---------------------------------------------------------------------------
# PHASE 1: PERCEIVE
# ---------------------------------------------------------------------------
def perceive_phase(rt, world):
    with timed(f"{rt.tag} perceive (stores new obs only)"):
        rt.observations, rt.new_obs = perceive_environment(
            rt.agent, world, rt.stream, rt.prev_obs
        )
    rt.prev_obs = rt.observations

    log.debug(f"{rt.tag} {len(rt.observations)} observations, {len(rt.new_obs)} new")
    for obs in rt.observations:
        marker = "*" if obs in rt.new_obs else " "
        log.debug(f"  {marker} {obs}")


# ---------------------------------------------------------------------------
# PHASE 2: DECIDE
# ---------------------------------------------------------------------------
def reaction_candidates(rt):
    """Observations worth asking the LLM about."""
    if PAPER_STRICT:
        return rt.observations  # paper: react to everything perceived this step
    if not REACT_ONLY_TO_EXTERNAL:
        return rt.new_obs
    # Self-observations all start with the agent's own name. This also drops
    # "Adam is interacting with the computer"-style lines about themselves.
    name = rt.agent["name"]
    return [o for o in rt.new_obs if not o.startswith(name)]


def decide_phase(rt, step, current_time, world):
    planner = rt.planner
    rt.reaction = ""

    rt.status = planner.get_current_status(rt.agent, current_time)
    rt.chunk = get_chunk_lazy(rt, current_time, world)

    log.debug(f"{rt.tag} status: {rt.status}")
    if rt.chunk:
        log.debug(f"{rt.tag} planned: {rt.chunk['activity']} (default loc: {rt.chunk['location']})")
    else:
        log.debug(f"{rt.tag} planned: (no chunk covers {current_time})")

    steps_since_react = step - rt.last_react_step

    # Who just showed up in the room (or whose room this agent just walked into).
    company = teammates_here(rt, world)
    arrivals = company - rt.prev_company
    rt.prev_company = company
    partner = planned_partner(rt, world, current_time)

    # Gate 1: nothing to follow -> nothing to decide.
    if rt.chunk is None:
        rt.decision, reason = "1", "no current chunk"

    # Gate 1b: step 1 is just the baseline snapshot of the room.
    elif SKIP_REACT_ON_FIRST_STEP and step == 1:
        rt.decision, reason = "1", "first step baseline (LLM skipped)"

    # Gate 1c: the plan says to work with a teammate who is right here: talk.
    # (Planned collaboration used to happen only if the react LLM happened to
    # choose "talk", so "collaborate with Eve 11-12" passed in silence.)
    elif partner is not None:
        activity = TIME_PREFIX.sub("", rt.chunk["activity"]).strip()
        activity = rt.planner._SPAN.sub("", activity).strip()
        rt.decision = "2"
        if rt.chunk.get("is_reaction"):                              # already "greet Eve ..."
            rt.reaction = activity
        elif rt.chunk.get("meeting"):
            activity = re.sub(r"(?i)^(?:attend|join)\s+(?:the\s+)?", "", activity)
            rt.reaction = (f"talk with {partner['name']} at the {activity}: what each is "
                           f"working on, what is next, and anything blocking them")
        else:
            rt.reaction = f"talk with {partner['name']} about: {activity}"
        reason = f"planned to work with {partner['name']}, who is here (LLM skipped)"

    # Gate 1d: recently replanned, give the new plan time to play out, unless
    # a teammate just walked in (that is worth a decision even mid-cooldown).
    elif steps_since_react < REACT_COOLDOWN_STEPS and not arrivals:
        rt.decision, reason = "1", (
            f"react cooldown, {steps_since_react} steps since replan (LLM skipped)"
        )

    else:
        candidates = reaction_candidates(rt)

        # Gate 2: nothing to react to -> skip the LLM entirely.
        if not candidates:
            rt.decision, reason = "1", "nothing new observed (LLM skipped)"
        else:
            with timed(f"{rt.tag} continue_plan_or_react (LLM)"):
                raw = planner.continue_plan_or_react(
                    rt.agent, rt.status, rt.chunk, candidates, world
                )
            rt.decision, rt.reaction = parse_decision(raw)
            reason = f"LLM said {raw.strip()!r} after {len(candidates)} obs"

            room = relocation_target(rt.reaction, rt.agent, world) if rt.decision == "2" else None
            if room is not None:
                # "Move to office_1 to keep debugging" isn't a reaction to
                # anything: the plan is fine, only the room was wrong. Move
                # without replacing the plan or starting the react cooldown.
                relocate(rt, room, current_time)
                reason += f" -> just a room change to {room} (no react, no cooldown)"
                rt.decision, rt.reaction = "1", ""

    log.debug(f"{rt.tag} decision {rt.decision}: {reason}")


TIME_PREFIX = re.compile(r"^\d{1,2}:\d{2}\s*[ap]m:\s*", re.IGNORECASE)
_MOVE_VERB = re.compile(
    r"^\s*(?:let'?s\s+)?(?:move|go|head|walk|return|relocate|switch|get)\w*\b(?:\s+back)?\s+to\b",
    re.IGNORECASE)


def teammates_here(rt, world):
    room = rt.agent["current_location"]
    return {a["id"] for a in world["agents"].values()
            if a["id"] != rt.agent["id"] and a["current_location"] == room}


def planned_partner(rt, world, current_time):
    """The teammate the current chunk is about, if they're in the same room and
    the two haven't just talked. Covers planned collaboration ("collaborate with
    Eve 11-12") and a reaction that walked over to someone ("greet Eve"), which
    used to arrive and then say nothing. The conversation cooldown keeps it from
    repeating, including right after a talk (the settle chunk targets the partner)."""
    chunk = rt.chunk
    if chunk and chunk.get("meeting"):
        return meeting_partner(rt, world, current_time)
    if not chunk or not chunk.get("target_agent"):
        return None
    other = world["agents"].get(chunk["target_agent"])
    if (other is None or other["id"] == rt.agent["id"]
            or other["current_location"] != rt.agent["current_location"]):
        return None
    if _talked_recently_ids(rt.agent["id"], other["id"], rt.planner._parse_time(current_time)):
        return None
    return other


def meeting_partner(rt, world, current_time):
    """At a team meeting, the teammate here to talk with: anyone in the room who
    hasn't already talked with this agent since the meeting started. The usual
    conversation cooldown doesn't apply (a chat at 7:05 shouldn't cancel the
    8:00 stand-up)."""
    p = rt.planner
    started = p._parse_time(rt.chunk.get("start_time"))
    for other in world["agents"].values():
        if other["id"] == rt.agent["id"] or other["current_location"] != rt.agent["current_location"]:
            continue
        last = _last_talk.get(frozenset((rt.agent["id"], other["id"])))
        if last is None or started is None or last < started:
            return other
    return None


def relocation_target(reaction, agent, world):
    """The room a reaction only moves the agent to ("Move to office_1 to keep
    debugging"), or None if it is a real reaction (names a teammate, names no
    room, or names a room the agent is already in)."""
    if not reaction or not _MOVE_VERB.search(reaction):
        return None
    text = reaction.lower()
    for other in world["agents"].values():
        if other["id"] != agent["id"] and re.search(
                rf"\b{other['name'].split()[0].lower()}\b", text):
            return None
    rooms = [r for r in world["locations"]
             if r in text or r.replace("_", " ") in text]
    if "my office" in text or "own office" in text:
        rooms.append(agent.get("office"))
    rooms = [r for r in dict.fromkeys(rooms) if r]
    if len(rooms) != 1 or rooms[0] == agent["current_location"]:
        return None
    return rooms[0]


def relocate(rt, room, current_time):
    """Keep doing the planned activity, but in `room`, until the chunk ends.
    Pinned, so choose_location can't send the agent back."""
    chunk = rt.chunk
    rt.override = {**chunk, "location": room, "pinned": True, "target_agent": None,
                   "start_time": chunk.get("start_time") or current_time,
                   "duration": chunk.get("duration") or STEP_MINUTES}
    rt.chunk = rt.override
    rt.last_chunk = None                    # different room: apply it this step


def parse_decision(raw):
    m = re.search(r"DECISION:\s*([12])", raw)
    decision = m.group(1) if m else "1"
    m = re.search(r"REACTION:\s*(.+)", raw)
    reaction = m.group(1).strip().strip('."') if m else ""
    if reaction.lower() in ("", "none"):
        decision = "1"          # "react" with nothing to do is just continue
    return decision, reaction

def resolve_conversations(runtimes, step, current_time, world):
    """If a reaction targets an agent in the same room, talk instead of
    just relabelling the action. Agents are handled one at a time, so the
    second agent responds to the first instead of making their own greeting."""
    by_id = {rt.agent["id"]: rt for rt in runtimes}
    busy = set()

    for rt in runtimes:
        if rt.decision != "2" or rt.agent["id"] in busy:
            continue

        target_id = rt.planner.extract_target_agent(rt.reaction, rt.agent, world)
        other = by_id.get(target_id)
        if (other is None
                or other.agent["id"] in busy
                or other.agent["current_location"] != rt.agent["current_location"]):
            continue                      # not a conversation; normal react() runs

        at_meeting = bool((rt.chunk or {}).get("meeting"))
        if (talked_recently(rt, other, current_time) and not at_meeting
                and not TASK_ID.search(rt.reaction)):
            # They just talked; nothing specific is new. Keep working instead.
            log.debug(f"{rt.tag} skips talking to {other.tag} again so soon: {rt.reaction}")
            rt.decision, rt.reaction = "1", ""
            continue

        log.conversation_start(rt.tag, other.tag, rt.agent["current_location"], rt.reaction)
        with timed(f"{rt.tag}<->{other.tag} conversation (LLM per turn)"):
            history = run_conversation(rt, other, current_time, rt.reaction, world)
        remember_conversation(rt, other, current_time, rt.reaction)

        # One call: shared meeting (if any) + each agent's own commitment
        outcomes = {"meeting": None, "commitments": {}}
        if history:
            transcript = "\n".join(f"{who}: {text}" for who, text in history)
            with timed(f"{rt.tag}<->{other.tag} extract outcomes (LLM)"):
                outcomes = rt.planner.extract_outcomes(
                    transcript, rt.agent, other.agent, world
                )
            log_outcomes(rt, other, outcomes)

        busy.update({rt.agent["id"], other.agent["id"]})

        for a, b in ((rt, other), (other, rt)):
            commitment = outcomes["commitments"].get(a.agent["name"])
            replanned = bool(commitment) and replan_after_conversation(
                a, commitment, current_time, world)
            settle_after_conversation(a, b, step, current_time, world, outcomes["meeting"])
            if replanned and not outcomes["meeting"]:
                queue_board_visit(a, commitment, current_time, world)


TASK_ID = re.compile(r"\bT-\d+\b", re.IGNORECASE)
_last_talk = {}          # frozenset({id_a, id_b}) -> datetime of their last conversation


def _talked_recently_ids(a_id, b_id, now):
    last = _last_talk.get(frozenset((a_id, b_id)))
    return (last is not None and now is not None
            and now - last < timedelta(minutes=CONVERSATION_COOLDOWN_MINUTES))


def talked_recently(rt, other, current_time):
    return _talked_recently_ids(rt.agent["id"], other.agent["id"],
                                rt.planner._parse_time(current_time))


def remember_conversation(rt, other, current_time, topic):
    """Both agents remember that this talk happened and what it was about, so
    later reactions (and reflection) can see they already covered it."""
    _last_talk[frozenset((rt.agent["id"], other.agent["id"]))] = rt.planner._parse_time(current_time)
    a, b = rt.agent["name"], other.agent["name"]
    line = f"{a} and {b} talked at {current_time} about: {topic.strip().rstrip('.')}."
    rt.stream.add_observations([line])
    other.stream.add_observations([line])


def _words(text):
    return {w for w in re.findall(r"[a-z0-9-]+", text.lower()) if len(w) > 3}


def is_repeat_commitment(rt, commitment, threshold=0.5):
    """Is this (nearly) a promise the agent already planned around today?"""
    new = _words(commitment)
    for old in rt.commitments:
        before = _words(old)
        if new and before and len(new & before) / len(new | before) >= threshold:
            return True
    return False


def log_outcomes(rt, other, outcomes):
    """Readable summary of what a conversation led to."""
    meeting = outcomes["meeting"]
    if meeting:
        log.event(f"agreed to meet in {meeting['location']}: "
                  f"{meeting['activity']} ({meeting['minutes']} min)")
    for who in (rt, other):
        commitment = outcomes["commitments"].get(who.agent["name"])
        if commitment:
            log.event(f"{who.tag} committed: {commitment}")
    if not meeting and not outcomes["commitments"]:
        log.event("no commitments made")


def replan_rest_of_day(rt, start, reason, world, label, extra_guidance=""):
    """Regenerate the plan from `start` on, keeping what already happened.
    If the agent has tasks on the board, the new plan is built around them.
    Returns True if the new plan was used."""
    done, remaining = rt.planner.split_plan(rt.broad_chunks, start)
    guidance = rt.planner.task_guidance(rt.agent, world) + extra_guidance

    with timed(f"{rt.tag} {label} (LLM)"):
        items = rt.planner.regenerate_plan(
            rt.agent, start, reason, done, remaining, guidance=guidance, world=world
        )
    new_chunks = rt.planner.create_chunks(items, rt.agent, world)

    if not new_chunks or any(c["start_time"] is None for c in new_chunks):
        log.warn(f"{rt.tag}: replan unusable, keeping old plan")
        return False

    new_chunks = rt.planner.apply_schedule(new_chunks, world, start)
    new_chunks = rt.planner.add_task_checkpoints(
        new_chunks, rt.agent, project_board.open_tasks_for(world, rt.agent["id"])
    )
    new_chunks = rt.planner.thin_board_visits(new_chunks)

    rt.broad_chunks = done + new_chunks
    rt.hour_cache, rt.minute_cache = {}, {}     # old decompositions are stale
    rt.planner.create_plan(new_chunks)
    log.debug(f"{rt.tag} new plan from {start}:")
    for chunk in new_chunks:
        log.debug(f"    {chunk['activity']}")
    return True


def replan_after_conversation(rt, commitment, current_time, world):
    """Rewrite the rest of the day around a commitment (paper: replan after dialogue).
    A promise that can only be kept at the board is also remembered for the next
    board visit, and the new plan has to include that visit. A promise the agent
    already planned around is not planned again. Returns True if it replanned."""
    if is_repeat_commitment(rt, commitment):
        log.event(f"{rt.tag} already planned around this: {commitment}")
        return False
    rt.commitments.append(commitment)
    start = step_time(current_time)        # the talk override covers the current step
    extra = ""
    if project_board.is_board_commitment(commitment):
        project_board.add_commitment(rt.agent["id"], commitment)
        extra = f"""
        {rt.agent['name']} can only keep this promise at the project board, so the
        plan must include a short visit to update the project board in the
        conference room soon after {start}.
        """
    if not replan_rest_of_day(rt, start,
                              f"{rt.agent['name']} agreed in a conversation to: {commitment}",
                              world, "replan after conversation", extra):
        return False
    rt.stream.add_observations([f"{rt.agent['name']} committed to: {commitment}"])
    log.event(f"{rt.tag} replanned the day around: {commitment}")
    absorb_board_changes(rt)
    return True


def absorb_board_changes(rt):
    """The plan was just rebuilt from the board as it is now (task_guidance), so
    a pending 'workload changed' flag would only trigger a second, identical
    replan. Keep any handoff notices as memories, then clear the flag."""
    agent_id = rt.agent["id"]
    if not project_board.workload_changed(agent_id):
        return
    notices = project_board.workload_notices(agent_id)
    if notices:
        rt.stream.add_observations(notices)
        for notice in notices:
            log.event(f"{rt.tag} notified: {notice}")
    project_board.clear_workload_changed(agent_id)


def replan_after_board(rt, current_time, world):
    """After a board visit where the agent took on or finished work, plan the
    rest of the day around their open tasks. Waits until they've stepped away
    from the board, so one visit leads to one replan."""
    if not project_board.workload_changed(rt.agent["id"]):
        return
    if rt.agent["interacting_with"] == project_board.BOARD_OBJECT:
        return

    notices = project_board.workload_notices(rt.agent["id"])
    project_board.clear_workload_changed(rt.agent["id"])
    if notices:
        # The board's notification: a teammate handed them work.
        rt.stream.add_observations(notices)
        for notice in notices:
            log.event(f"{rt.tag} notified: {notice}")

    tasks = project_board.open_tasks_for(world, rt.agent["id"])
    summary = ", ".join(f"{t['id']} ({t['status'].replace('_', ' ')})" for t in tasks)
    reason = " ".join(notices) or "updated the project board;"
    reason += f" {rt.agent['name']}'s open tasks are now {summary or 'none'}"

    if replan_rest_of_day(rt, step_time(current_time), reason, world,
                          "replan after project board"):
        log.event(f"{rt.tag} replanned the day around their tasks: {summary or 'none open'}")


def settle_after_conversation(rt, partner, step, current_time, world, agreement=None):
    """Hold both agents in place for one step, then send them to any agreed meeting."""
    rt.last_react_step = step
    rt.override = {
        "activity": f"{current_time}: Talk with {partner.agent['name']}",
        "location": rt.agent["current_location"],
        "start_time": current_time,
        "duration": STEP_MINUTES,
        "target_agent": partner.agent["id"],
        "is_reaction": True,
    }
    rt.chunk = rt.last_chunk = rt.override
    update_agent_state(rt.agent, rt.chunk, world)
    rt.agent["interacting_with"] = None
    rt.decision = "3"

    if agreement:
        start = step_time(current_time)
        rt.queued = {
            "activity": f"{start}: {agreement['activity']}",
            "location": agreement["location"],
            "start_time": start,
            "duration": agreement["minutes"],
            "target_agent": None,        # location is pinned, don't chase the partner
            "is_reaction": True,
            "pinned": True,
        }

def queue_board_visit(rt, commitment, current_time, world, minutes=10):
    """Promised something about the board while standing at it: do it right after
    the conversation instead of hoping the replan comes back to it."""
    room = project_board.board_location(world)
    if (room is None or rt.agent["current_location"] != room
            or not project_board.is_board_commitment(commitment)):
        return
    start = step_time(current_time)
    rt.queued = {
        "activity": f"{start}: update the project board: {commitment}",
        "location": room,
        "start_time": start,
        "duration": minutes,
        "target_agent": None,
        "is_reaction": True,
        "pinned": True,
        "board_work": True,
    }
    log.event(f"{rt.tag} heads straight to the board to: {commitment}")


# ---------------------------------------------------------------------------
# PHASE 3: ACT
# ---------------------------------------------------------------------------
def apply_chunk(rt, world):
    """Move the agent to its chunk's action/location and update the world."""
    chunk = rt.chunk

    board_room = project_board.board_location(world)
    if chunk.get("pinned"):
        # Agreed meeting place: no LLM, no second-guessing
        loc = chunk["location"]
    elif board_room and (chunk.get("board_work")
                         or project_board.mentions_board_explicitly(chunk["activity"])):
        # Board work happens at the board (also "review the board with Eve"):
        # no LLM, so it can't send them to their office instead.
        loc = board_room
    elif chunk.get("target_agent"):
        # update_agent_state follows the target agent's current room
        loc = None
    elif (obvious := rt.planner.obvious_location(chunk["activity"], rt.agent, world)):
        loc = obvious
        log.debug(f"{rt.tag} location {loc} (obvious, LLM skipped)")
    else:
        with timed(f"{rt.tag} choose_location (LLM)"):
            loc = rt.planner.choose_location(
                chunk["activity"], rt.agent, world,
                observations=rt.observations,
            )
        log.debug(f"{rt.tag} chose location: {loc}")

    update_agent_state(rt.agent, chunk, world, location=loc)

    with timed(f"{rt.tag} generate_interactions (LLM x2 + summary)"):
        rt.agent, _ = generate_interactions(
            rt.agent, world, rt.observations, rt.planner, chunk=chunk
        )

    world["agents"][rt.agent["id"]] = rt.agent  # keep world in sync


def react(rt, step, current_time, world):
    """React with a short interruption instead of regenerating the day."""
    log.event(f"{rt.tag} reacts: {rt.reaction}")
    rt.last_react_step = step
    rt.override = rt.planner.make_reaction_chunk(
        rt.agent, current_time, rt.reaction, world
    )
    rt.chunk = rt.override
    log.debug(f"{rt.tag} reaction chunk: {rt.chunk}")
    apply_chunk(rt, world)
    return True


def act_phase(rt, step, current_time, world):

    if rt.decision == "3":
        return

    if rt.decision == "1":
        if rt.chunk is None:
            log.debug(f"{rt.tag} act: no chunk, state unchanged")
            return

        if SKIP_UNCHANGED_CHUNKS and rt.chunk == rt.last_chunk:
            if rt.chunk.get("target_agent"):
                # cheap check: did the target move?
                old = rt.agent["current_location"]
                update_agent_state(rt.agent, rt.chunk, world)
                if rt.agent["current_location"] != old:
                    apply_chunk(rt, world)
                else:
                    log.debug(f"{rt.tag} act: target still here, skipping update")
            else:
                log.debug(f"{rt.tag} act: same chunk as last step, skipping update")
            return

        apply_chunk(rt, world)

    elif rt.decision == "2":
        if not react(rt, step, current_time, world):
            return

    rt.last_chunk = rt.chunk


# ---------------------------------------------------------------------------
# PHASE 4: REFLECT (twice a day) and per-step summary
# ---------------------------------------------------------------------------
def due_reflection_times(rt, current_time):
    """Scheduled reflection times that have arrived and haven't run yet."""
    now = rt.planner._parse_time(current_time)
    return [
        t for t in REFLECTION_TIMES
        if t not in rt.reflected_at
        and (scheduled := rt.planner._parse_time(t)) is not None
        and now >= scheduled
    ]


def reflect_phase(rt, current_time):
    due = due_reflection_times(rt, current_time)
    if not due:
        return

    rt.reflected_at.update(due)   # a late start still reflects once, not once per time
    log.reflection_header(rt.tag, current_time)

    with timed(f"{rt.tag} reflect (LLM x3+)"):
        reflections = rt.reflector.reflect()

    if not reflections:
        log.event(f"{rt.tag} had nothing new to reflect on")
        return

    for reflection in reflections:
        log.reflection_item(reflection.description)

    # New insights change who the agent is: rebuild the summary next time it's needed.
    rt.planner.invalidate_summary()


def log_step_summary(rt):
    """One compact line per agent: where they are, what they're doing, what they're using."""
    a = rt.agent
    log.agent_line(rt.tag, a["current_location"], a["current_action"], a["interacting_with"])


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    world = load_world()
    runtimes = [make_runtime("adam", world), make_runtime("eve", world)]

    log.banner(f"Generative Agents · {START_TIME} · {NUM_STEPS} steps x {STEP_MINUTES} min")
    log.info(f"  PAPER_STRICT={PAPER_STRICT} EAGER_DECOMPOSITION={EAGER_DECOMPOSITION} "
             f"REACT_COOLDOWN_STEPS={REACT_COOLDOWN_STEPS} REFLECTION_TIMES={REFLECTION_TIMES}")

    current_time = START_TIME
    clock.set_time(current_time)
    project_board.clear_workload_changed()      # nothing carried over from a previous run
    project_board.clear_commitments()
    _last_talk.clear()

    for rt in runtimes:
        # A reflection time that is already behind us at start-up never fires.
        start = rt.planner._parse_time(START_TIME)
        rt.reflected_at = {t for t in REFLECTION_TIMES if rt.planner._parse_time(t) < start}
        initial_planning(rt, world)

    try:
        for step in range(1, NUM_STEPS + 1):
            step_start = time.time()
            clock.set_time(current_time)
            log.step_header(step, NUM_STEPS, current_time)

            # Run each phase for every agent before moving on, so one agent's
            # actions don't leak into the other's perception mid-step.
            for rt in runtimes:
                perceive_phase(rt, world)

            for rt in runtimes:
                decide_phase(rt, step, current_time, world)

            resolve_conversations(runtimes, step, current_time, world)

            for rt in runtimes:
                act_phase(rt, step, current_time, world)

            for rt in runtimes:
                replan_after_board(rt, current_time, world)

            for rt in runtimes:
                log_step_summary(rt)

            for rt in runtimes:
                reflect_phase(rt, current_time)

            # One disk write per step instead of one per action / retrieval.
            save_world(world)
            for rt in runtimes:
                rt.stream.flush()

            elapsed = time.time() - step_start
            log.debug(f"step {step} took {elapsed:.1f}s")

            current_time = step_time(current_time)

            sleep_time = max(0, STEP_DELAY_SECONDS - elapsed)
            if sleep_time > 0:
                log.debug(f"sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

    finally:
        # Runs on Ctrl-C or a crash, so access timestamps and observations
        # from the interrupted step still get written.
        for rt in runtimes:
            rt.stream.flush()
        print_timing_summary()


if __name__ == "__main__":
    main()