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

Assumes these are in place:
    - perceive_environment(agent, world, stream, prev_obs) -> (all_obs, new_obs)
    - Planner(agent, memory_stream, memory_retriever)
    - Planner.split_plan(chunks, current_time) -> (done, remaining)
    - Planner.regenerate_plan(agent, current_time, reaction, done, remaining)
    - Planner.continue_plan_or_react(agent, status, chunk, observations, world) ->
      text containing "DECISION: 1|2" and "REACTION: ..."
    - MemoryRetriever(agent_id, memory_stream)
    - MemoryStream.flush() / mark_dirty() / add_observations() / add_plan()
    - generate_interactions(agent, world, observations, planner)
"""

import re
import time
import clock
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from memory_retriever import MemoryRetriever
from memory_stream import MemoryStream
from perceive import perceive_environment
from plan import Planner
from world import generate_interactions, load_world, save_world, update_agent_state

from conversation import run_conversation


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
START_TIME = "7:00 am"
NUM_STEPS = 30
STEP_MINUTES = 5            # simulated minutes per step
STEP_DELAY_SECONDS = 0      # wall-clock padding per step (0 = run flat out)

DEBUG_OBSERVATIONS = False  # True = print every observation each step

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
# steps. Stops agents from replanning on nearly every step.
REACT_COOLDOWN_STEPS = 6

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
        print(f"    [timing] {label}: {dt:.1f}s")


def print_timing_summary():
    if not timings:
        return
    print("\n" + "=" * 50)
    print("TIMING SUMMARY (slowest first)")
    for label, (total, count) in sorted(timings.items(), key=lambda kv: -kv[1][0]):
        print(f"  {total:7.1f}s total | {count:3d} calls | {total / count:6.1f}s avg | {label}")


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
    initial_plan_fn: Callable        # planner.generate_initial_plan_<name>

    # The plan, at three levels of detail. The two caches are keyed by
    # (start_time, activity), so a replan can never hit a stale entry.
    broad_chunks: list = field(default_factory=list)
    hour_cache: dict = field(default_factory=dict)
    minute_cache: dict = field(default_factory=dict)

    prev_obs: list = field(default_factory=list)
    last_chunk: Optional[dict] = None
    last_react_step: int = -10_000
    override: Optional[dict] = None
    queued: Optional[dict] = None

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


# Adam and Eve still have separate initial-plan prompts.
INITIAL_PLAN_METHODS = {
    "adam": "generate_initial_plan_adam",
    "eve": "generate_initial_plan_eve",
}


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
        initial_plan_fn=getattr(planner, INITIAL_PLAN_METHODS[agent_id]),
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
        daily_plan = rt.initial_plan_fn()

    rt.broad_chunks = rt.planner.create_chunks(daily_plan, rt.agent, world)
    rt.hour_cache, rt.minute_cache = {}, {}

    print(f"{rt.tag} BROAD PLAN")
    if not rt.broad_chunks:
        # No chunks means get_chunk_lazy always returns None.
        print(f"WARNING [{rt.tag}]: initial plan produced no chunks. "
              f"Daily plan was: {daily_plan}")
        return

    rt.planner.create_plan(rt.broad_chunks)  # plans live in the memory stream

    if EAGER_DECOMPOSITION:
        expand_plan(rt, rt.broad_chunks)


def get_chunk_lazy(rt, current_time):
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

    hour = p.get_current_minute_chunk(current_time, expand_hours(rt, broad))
    if hour is None:
        return None

    return p.get_current_minute_chunk(current_time, expand_minutes(rt, hour))


# ---------------------------------------------------------------------------
# TIME / PARSING HELPERS
# ---------------------------------------------------------------------------
def step_time(current_time, minutes=STEP_MINUTES):
    current_dt = datetime.strptime(current_time.upper(), "%I:%M %p")
    new_dt = current_dt + timedelta(minutes=minutes)
    # %I zero-pads ("07:05 am"), which the planner's strptime handles fine.
    return new_dt.strftime("%I:%M %p").lower()


# ---------------------------------------------------------------------------
# PHASE 1: PERCEIVE
# ---------------------------------------------------------------------------
def perceive_phase(rt, world):
    with timed(f"{rt.tag} perceive (stores new obs only)"):
        rt.observations, rt.new_obs = perceive_environment(
            rt.agent, world, rt.stream, rt.prev_obs
        )
    rt.prev_obs = rt.observations

    print(f"[{rt.tag}] {len(rt.observations)} observations, {len(rt.new_obs)} new")
    if DEBUG_OBSERVATIONS:
        for obs in rt.observations:
            marker = "*" if obs in rt.new_obs else " "
            print(f"    {marker} {obs}")


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
    rt.chunk = get_chunk_lazy(rt, current_time)

    print(f"[{rt.tag}] STATUS: {rt.status}")
    if rt.chunk:
        print(f"[{rt.tag}] PLANNED: {rt.chunk['activity']} (default loc: {rt.chunk['location']})")
    else:
        print(f"[{rt.tag}] PLANNED: (no chunk covers {current_time})")

    steps_since_react = step - rt.last_react_step

    # Gate 1: nothing to follow -> nothing to decide.
    if rt.chunk is None:
        rt.decision, reason = "1", "no current chunk"

    # Gate 1b: step 1 is just the baseline snapshot of the room.
    elif SKIP_REACT_ON_FIRST_STEP and step == 1:
        rt.decision, reason = "1", "first step baseline (LLM skipped)"

    # Gate 1c: recently replanned, give the new plan time to play out.
    elif steps_since_react < REACT_COOLDOWN_STEPS:
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

    print(f"[{rt.tag}] DECISION: {rt.decision} ({reason})")

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

        print(f"[{rt.tag}] CONVERSATION with {other.tag}: {rt.reaction}")
        with timed(f"{rt.tag}<->{other.tag} conversation (LLM per turn)"):
            history = run_conversation(rt, other, current_time, rt.reaction)

        # One call: shared meeting (if any) + each agent's own commitment
        outcomes = {"meeting": None, "commitments": {}}
        if history:
            transcript = "\n".join(f"{who}: {text}" for who, text in history)
            with timed(f"{rt.tag}<->{other.tag} extract outcomes (LLM)"):
                outcomes = rt.planner.extract_outcomes(
                    transcript, rt.agent, other.agent, world
                )
            print(f"[{rt.tag}] OUTCOMES: {outcomes}")

        busy.update({rt.agent["id"], other.agent["id"]})

        for a, b in ((rt, other), (other, rt)):
            commitment = outcomes["commitments"].get(a.agent["name"])
            if commitment:
                replan_after_conversation(a, commitment, current_time, world)
            settle_after_conversation(a, b, step, current_time, world, outcomes["meeting"])


def replan_after_conversation(rt, commitment, current_time, world):
    """Rewrite the rest of the day around a commitment (paper: replan after dialogue)."""
    start = step_time(current_time)        # the talk override covers the current step
    done, remaining = rt.planner.split_plan(rt.broad_chunks, start)

    with timed(f"{rt.tag} replan after conversation (LLM)"):
        items = rt.planner.regenerate_plan(
            rt.agent, start, f"Committed in conversation: {commitment}", done, remaining
        )
    new_chunks = rt.planner.create_chunks(items, rt.agent, world)

    if not new_chunks or any(c["start_time"] is None for c in new_chunks):
        print(f"[{rt.tag}] replan unusable, keeping old plan")
        return

    rt.broad_chunks = done + new_chunks
    rt.hour_cache, rt.minute_cache = {}, {}     # old decompositions are stale
    rt.planner.create_plan(new_chunks)
    rt.stream.add_observations([f"{rt.agent['name']} committed to: {commitment}"])
    print(f"[{rt.tag}] REPLANNED around: {commitment}")

    
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

# ---------------------------------------------------------------------------
# PHASE 3: ACT
# ---------------------------------------------------------------------------
def apply_chunk(rt, world):
    """Move the agent to its chunk's action/location and update the world."""
    chunk = rt.chunk

    if chunk.get("pinned"):
        # Agreed meeting place: no LLM, no second-guessing
        loc = chunk["location"]
    elif chunk.get("target_agent"):
        # update_agent_state follows the target agent's current room
        loc = None
    else:
        with timed(f"{rt.tag} choose_location (LLM)"):
            loc = rt.planner.choose_location(
                chunk["activity"], rt.agent, world,
                observations=rt.observations,
            )
        print(f"[{rt.tag}] CHOSE LOCATION: {loc}")

    update_agent_state(rt.agent, chunk, world, location=loc)

    with timed(f"{rt.tag} generate_interactions (LLM x2 + summary)"):
        rt.agent, _ = generate_interactions(
            rt.agent, world, rt.observations, rt.planner, chunk=chunk
        )

    world["agents"][rt.agent["id"]] = rt.agent  # keep world in sync

    print(f"[{rt.tag}] NEW ACTION:   {rt.agent['current_action']}")
    print(f"[{rt.tag}] NEW ROOM:     {rt.agent['current_location']}")
    print(f"[{rt.tag}] INTERACTING:  {rt.agent['interacting_with']}")


def react(rt, step, current_time, world):
    """React with a short interruption instead of regenerating the day."""
    print(f"[{rt.tag}] ACT: reacting ({rt.reaction})")
    rt.last_react_step = step
    rt.override = rt.planner.make_reaction_chunk(
        rt.agent, current_time, rt.reaction, world
    )
    rt.chunk = rt.override
    print(f"[{rt.tag}] REACTION CHUNK: {rt.chunk}")
    apply_chunk(rt, world)
    return True


def act_phase(rt, step, current_time, world):

    if rt.decision == "3":
        return

    if rt.decision == "1":
        if rt.chunk is None:
            print(f"[{rt.tag}] ACT: no chunk, state unchanged")
            return

        if SKIP_UNCHANGED_CHUNKS and rt.chunk == rt.last_chunk:
            if rt.chunk.get("target_agent"):
                # cheap check: did the target move?
                old = rt.agent["current_location"]
                update_agent_state(rt.agent, rt.chunk, world)
                if rt.agent["current_location"] != old:
                    apply_chunk(rt, world)
                else:
                    print(f"[{rt.tag}] ACT: target still here, skipping update")
            else:
                print(f"[{rt.tag}] ACT: same chunk as last step, skipping update")
            return

        apply_chunk(rt, world)

    elif rt.decision == "2":
        if not react(rt, step, current_time, world):
            return

    rt.last_chunk = rt.chunk


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    world = load_world()
    runtimes = [make_runtime("adam", world), make_runtime("eve", world)]

    print(f"PAPER_STRICT={PAPER_STRICT} | EAGER_DECOMPOSITION={EAGER_DECOMPOSITION} | "
          f"REACT_COOLDOWN_STEPS={REACT_COOLDOWN_STEPS}")

    current_time = START_TIME
    clock.set_time(current_time)

    for rt in runtimes:
        initial_planning(rt, world)

    try:
        for step in range(1, NUM_STEPS + 1):
            step_start = time.time()
            clock.set_time(current_time)
            print("\n" + "=" * 50)
            print(f"STEP {step}/{NUM_STEPS} | TIME: {current_time}")

            # Run each phase for every agent before moving on, so one agent's
            # actions don't leak into the other's perception mid-step.
            for rt in runtimes:
                perceive_phase(rt, world)

            for rt in runtimes:
                decide_phase(rt, step, current_time, world)

            resolve_conversations(runtimes, step, current_time, world)

            for rt in runtimes:
                act_phase(rt, step, current_time, world)

            # One disk write per step instead of one per action / retrieval.
            save_world(world)
            for rt in runtimes:
                rt.stream.flush()

            elapsed = time.time() - step_start
            print(f"\nStep {step} took {elapsed:.1f}s")

            current_time = step_time(current_time)

            sleep_time = max(0, STEP_DELAY_SECONDS - elapsed)
            if sleep_time > 0:
                print(f"Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

    finally:
        # Runs on Ctrl-C or a crash, so access timestamps and observations
        # from the interrupted step still get written.
        for rt in runtimes:
            rt.stream.flush()
        print_timing_summary()


if __name__ == "__main__":
    main()