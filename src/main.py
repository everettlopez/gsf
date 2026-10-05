"""
main.py - Generative Agents simulation driver (Adam + Eve)

Per step, the loop runs three phases for BOTH agents (so they act
"simultaneously" and don't see each other's mid-step changes):

    1. PERCEIVE  - build observations, store only the NEW ones in memory
    2. DECIDE    - continue the plan (1) or react (2); the LLM is only asked
                   when there is something new to react to
    3. ACT       - update agent/world state (1) or regenerate the plan (2)

Assumes the earlier refactors are in place:
    - perceive_environment(agent, world, stream, prev_obs) -> (all_obs, new_obs)
    - Planner(agent, memory_stream, memory_retriever)
    - MemoryRetriever(agent_id, memory_stream)
    - MemoryStream.flush() / mark_dirty() / add_observations()
    - generate_interactions(agent, world, observations, planner)
"""

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from memory_retriever import MemoryRetriever
from memory_stream import MemoryStream
from perceive import perceive_environment
from plan import Planner
from world import generate_interactions, load_world, save_world, update_agent_state


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
START_TIME = "7:00 am"
NUM_STEPS = 30
STEP_MINUTES = 5            # simulated minutes per step
STEP_DELAY_SECONDS = 0      # wall-clock padding per step (was 120; 0 = run flat out)

DEBUG_OBSERVATIONS = False  # True = print every observation each step

# Only ask the LLM "continue or react?" when something NEW was observed.
# With this True, an agent's own state lines (e.g. "Adam is currently at...")
# don't count, so agents don't "react" to the changes they just made themselves.
REACT_ONLY_TO_EXTERNAL = True

# If the agent is still on the same plan chunk and keeps going (decision 1),
# skip update_agent_state + generate_interactions (2+ LLM calls). Set False to
# restore the old behaviour of re-running them every step.
SKIP_UNCHANGED_CHUNKS = True


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
    regenerate_plan_fn: Callable     # planner.regenerate_plan_<name>

    minute_chunks: list = field(default_factory=list)
    prev_obs: list = field(default_factory=list)
    last_chunk: Optional[dict] = None

    # Per-step scratch space (filled in by the phases below)
    observations: list = field(default_factory=list)
    new_obs: list = field(default_factory=list)
    chunk: Optional[dict] = None
    status: str = ""
    decision: str = "1"

    @property
    def tag(self):
        return self.agent["id"].upper()


# Adam and Eve have separate plan prompts, so map each id to its method names.
PLAN_METHODS = {
    "adam": ("generate_initial_plan_adam", "regenerate_plan_adam"),
    "eve": ("generate_initial_plan_eve", "regenerate_plan_eve"),
}


def make_runtime(agent_id, world):
    # Use the agent dict that lives inside world so there is a single source of
    # truth (previously load_agent() and load_world() gave two diverging copies).
    agent = world["agents"][agent_id]

    stream = MemoryStream(agent_id)
    retriever = MemoryRetriever(agent_id, stream)
    planner = Planner(agent, stream, retriever)

    initial_name, regen_name = PLAN_METHODS[agent_id]
    return AgentRuntime(
        agent=agent,
        stream=stream,
        retriever=retriever,
        planner=planner,
        initial_plan_fn=getattr(planner, initial_name),
        regenerate_plan_fn=getattr(planner, regen_name),
    )


# ---------------------------------------------------------------------------
# PLANNING
# ---------------------------------------------------------------------------
def build_plan_hierarchy(rt, daily_plan):
    """
    broad plan -> hourly chunks -> 5-15 minute chunks.
    Each level is also saved to memory as a "plan" memory (as before).
    Returns the minute chunks (empty list if the LLM output couldn't be parsed).
    """
    planner, agent, tag = rt.planner, rt.agent, rt.tag

    chunks = planner.create_chunks(daily_plan, agent)
    print(f"{tag} BROAD PLAN")
    planner.create_plan(chunks)

    with timed(f"{tag} decompose_chunks (hour, LLM)"):
        hour_chunks = planner.decompose_chunks(chunks, agent)
    print(f"{tag} HOUR CHUNKS")
    planner.create_plan(hour_chunks)

    with timed(f"{tag} decompose_hour_chunks (minute, LLM)"):
        minute_chunks = planner.decompose_hour_chunks(hour_chunks, agent)
    print(f"{tag} MINUTE CHUNKS")
    planner.create_plan(minute_chunks)

    return minute_chunks


def initial_planning(rt):
    with timed(f"{rt.tag} initial broad plan (LLM)"):
        daily_plan = rt.initial_plan_fn()

    rt.minute_chunks = build_plan_hierarchy(rt, daily_plan)

    if not rt.minute_chunks:
        # No chunks means get_current_minute_chunk always returns None.
        print(f"WARNING [{rt.tag}]: initial plan produced no chunks. "
              f"Daily plan was: {daily_plan}")


# ---------------------------------------------------------------------------
# TIME HELPERS
# ---------------------------------------------------------------------------
def step_time(current_time, minutes=STEP_MINUTES):
    current_dt = datetime.strptime(current_time.upper(), "%I:%M %p")
    new_dt = current_dt + timedelta(minutes=minutes)
    # %I zero-pads ("07:05 am"), which the planner's strptime handles fine.
    return new_dt.strftime("%I:%M %p").lower()


def parse_decision(raw):
    """Pull the first 1 or 2 out of the model's reply; default to continue."""
    match = re.search(r"[12]", raw or "")
    if not match:
        print(f"WARNING: could not parse decision from {raw!r}, defaulting to 1")
        return "1"
    return match.group(0)


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
    """New observations that are worth asking the LLM about."""
    if not REACT_ONLY_TO_EXTERNAL:
        return rt.new_obs
    # Self-observations all start with the agent's own name. This also drops
    # "Adam is interacting with the computer"-style lines about themselves.
    name = rt.agent["name"]
    return [o for o in rt.new_obs if not o.startswith(name)]


def decide_phase(rt, current_time, world):
    planner = rt.planner

    rt.status = planner.get_current_status(rt.agent, current_time)
    rt.chunk = planner.get_current_minute_chunk(current_time, rt.minute_chunks)

    print(f"[{rt.tag}] STATUS: {rt.status}")
    if rt.chunk:
        print(f"[{rt.tag}] PLANNED: {rt.chunk['activity']} @ {rt.chunk['location']}")
    else:
        print(f"[{rt.tag}] PLANNED: (no chunk covers {current_time})")

    # Gate 1: nothing to follow -> nothing to decide.
    if rt.chunk is None:
        rt.decision, reason = "1", "no current chunk"

    else:
        candidates = reaction_candidates(rt)

        # Gate 2: nothing new to react to -> skip the LLM entirely.
        if not candidates:
            rt.decision, reason = "1", "nothing new observed (LLM skipped)"
        else:
            with timed(f"{rt.tag} continue_plan_or_react (LLM)"):
                raw = planner.continue_plan_or_react(
                    rt.agent, rt.status, rt.chunk, rt.observations, world
                )
            rt.decision = parse_decision(raw)
            reason = f"LLM said {raw.strip()!r} after {len(candidates)} new obs"

    print(f"[{rt.tag}] DECISION: {rt.decision} ({reason})")


# ---------------------------------------------------------------------------
# PHASE 3: ACT
# ---------------------------------------------------------------------------
def apply_chunk(rt, world):
    """Move the agent to its chunk's action/location and update the world."""
    update_agent_state(rt.agent, rt.chunk)

    with timed(f"{rt.tag} generate_interactions (LLM x2 + summary)"):
        rt.agent, _ = generate_interactions(
            rt.agent, world, rt.observations, rt.planner
        )

    world["agents"][rt.agent["id"]] = rt.agent  # keep world in sync

    print(f"[{rt.tag}] NEW ACTION:   {rt.agent['current_action']}")
    print(f"[{rt.tag}] NEW ROOM:     {rt.agent['current_location']}")
    print(f"[{rt.tag}] INTERACTING:  {rt.agent['interacting_with']}")


def act_phase(rt, current_time, world):
    if rt.decision == "1":
        if rt.chunk is None:
            print(f"[{rt.tag}] ACT: no chunk, state unchanged")
            return

        if SKIP_UNCHANGED_CHUNKS and rt.chunk == rt.last_chunk:
            print(f"[{rt.tag}] ACT: same chunk as last step, skipping update")
            return

        apply_chunk(rt, world)

    elif rt.decision == "2":
        print(f"[{rt.tag}] ACT: reacting, regenerating plan from {current_time}")

        with timed(f"{rt.tag} regenerate plan (LLM)"):
            daily_plan = rt.regenerate_plan_fn(
                rt.agent, current_time, rt.chunk, rt.observations, rt.status, world
            )

        new_minute_chunks = build_plan_hierarchy(rt, daily_plan) if daily_plan else []

        if not new_minute_chunks:
            # Don't wipe the working plan if the model's output didn't parse.
            print(f"WARNING [{rt.tag}]: regenerated plan was empty/unparseable, "
                  f"keeping the old plan. Raw plan items: {daily_plan}")
            return

        rt.minute_chunks = new_minute_chunks
        rt.chunk = rt.planner.get_current_minute_chunk(current_time, rt.minute_chunks)
        print(f"[{rt.tag}] NEW CURRENT CHUNK: {rt.chunk}")

        if rt.chunk is not None:
            apply_chunk(rt, world)

    rt.last_chunk = rt.chunk


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    world = load_world()
    runtimes = [make_runtime("adam", world), make_runtime("eve", world)]

    # Initial plans (the old time.sleep(45) pauses are gone; add them back
    # inside build_plan_hierarchy if you need to throttle the GPU).
    for rt in runtimes:
        initial_planning(rt)

    current_time = START_TIME

    try:
        for step in range(1, NUM_STEPS + 1):
            step_start = time.time()
            print("\n" + "=" * 50)
            print(f"STEP {step}/{NUM_STEPS} | TIME: {current_time}")

            # Run each phase for every agent before moving on, so one agent's
            # actions don't leak into the other's perception mid-step.
            for rt in runtimes:
                perceive_phase(rt, world)

            for rt in runtimes:
                decide_phase(rt, current_time, world)

            for rt in runtimes:
                act_phase(rt, current_time, world)

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