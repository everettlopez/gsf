"""
main.py - Generative Agents simulation driver (Adam + Eve)

The world clock and the step loop. Each agent (agent.py) perceives, decides,
acts and reflects; per step, each phase runs for BOTH agents before the next
one starts, so they act "simultaneously" and don't see each other's mid-step
changes:

    1. PERCEIVE  - every agent
    2. DECIDE    - every agent
    3. CONVERSE  - a reaction aimed at a teammate in the same room becomes a
                   conversation (paper 4.3.2), resolved here because it takes two
    4. ACT       - every agent, then replans after board visits
    5. REFLECT   - every agent whose perceived importance reached the threshold

The per-agent switches (PAPER_STRICT, EAGER_DECOMPOSITION, REFLECTION_THRESHOLD,
...) are at the top of agent.py.

A run is one workday by default (GA_STEPS steps from START_TIME), or
GA_DAYS whole workdays (DAY_STEPS steps each, 7:00 am to 5:00 pm): each new
day the agents plan afresh from the day before (agent.start_day), while their
memories, the board and the hands' work carry over.

Along the way the chronicle (chronicle.py) keeps the story of the run, and
with real hands (GA_EXECUTOR=local) a snapshot of the office (snapshot.py)
goes to the sandbox every hour of sim time, for the website the agents are
building about their own working days.

Verbosity: set GA_LOG=quiet|info|debug (default info). See log.py.
"""

import os
import time
from datetime import timedelta

import agent
import chronicle
import clock
import log
import persona
import project_board
import snapshot
from agent import (TASK_ID, Agent, end_day, initial_planning, meeting_pending, note_board_use,
                   queue_board_visit, replan_after_board, replan_after_conversation,
                   settle_after_conversation, start_day)
from clock import STEP_MINUTES, step_time
from executor import hands_for, run_budget
from conversation import clear_talks, record_talk, run_conversation, talked_recently_ids
from log import print_timing_summary, timed
from world import load_world, save_world


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
START_TIME = "7:00 am"
NUM_STEPS = 30               # GA_STEPS overrides it for one run
DAY_STEPS = 120             # a whole workday, 7:00 am to 5:00 pm (GA_DAYS)
SNAPSHOT_EVERY = 12         # steps between snapshots for the website: an hour
STEP_DELAY_SECONDS = 0      # wall-clock padding per step (0 = run flat out)


# ---------------------------------------------------------------------------
# CONVERSATIONS
# ---------------------------------------------------------------------------
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

        if (talked_recently(rt, other, current_time) and not TASK_ID.search(rt.reaction)
                and not meeting_pending(rt, rt.chunk)):
            # They just talked; nothing specific is new. Keep working instead.
            log.debug(f"{rt.tag} skips talking to {other.tag} again so soon: {rt.reaction}")
            rt.decision, rt.reaction = "1", ""
            continue

        # Before the talk is recorded: is this the agreed meeting being held?
        at_meeting = meeting_pending(rt, rt.chunk) or meeting_pending(other, other.chunk)
        log.conversation_start(rt.tag, other.tag, rt.agent["current_location"], rt.reaction)
        with timed(f"{rt.tag}<->{other.tag} conversation (LLM per turn)"):
            history = run_conversation(rt, other, current_time, rt.reaction, world)
        remember_conversation(rt, other, current_time, rt.reaction)
        where = rt.agent["current_location"]

        # One call: shared meeting (if any) + each agent's own commitment
        outcomes = {"meeting": None, "commitments": {}}
        if history:
            transcript = "\n".join(f"{who}: {text}" for who, text in history)
            with timed(f"{rt.tag}<->{other.tag} extract outcomes (LLM)"):
                outcomes = rt.planner.extract_outcomes(
                    transcript, rt.agent, other.agent, world, current_time
                )
            log_outcomes(rt, other, outcomes)
        if history:
            chronicle.note_conversation(where, rt.reaction, history, outcomes)

        busy.update({rt.agent["id"], other.agent["id"]})

        meeting = outcomes["meeting"]
        # At an agreed meeting, "meet now" is the meeting they're in.
        meet_now = bool(meeting) and not meeting.get("start") and not at_meeting
        if meet_now:
            # Right after this step. It used to be only a one-off queued chunk
            # outside the plan, and the replan (told to drop blocks with the
            # partner for the next hour) sent Eve back to her office instead.
            meeting = {**meeting, "start": step_time(current_time)}
        if meeting and meeting.get("start"):
            # Into both days before replanning, so the replans plan around it.
            schedule_meeting(rt, other, meeting, current_time)

        for a, b in ((rt, other), (other, rt)):
            commitment = outcomes["commitments"].get(a.agent["name"])
            replanned = bool(commitment) and replan_after_conversation(
                a, commitment, current_time, world, partner=b.agent["name"])
            settle_after_conversation(a, b, step, current_time, world)
            if replanned and not meet_now:
                queue_board_visit(a, commitment, current_time, world)



def talked_recently(rt, other, current_time):
    return talked_recently_ids(rt.agent["id"], other.agent["id"],
                               rt.planner._parse_time(current_time))


def remember_conversation(rt, other, current_time, topic):
    """Both agents remember that this talk happened and what it was about, so
    later reactions (and reflection) can see they already covered it."""
    record_talk(rt.agent["id"], other.agent["id"], rt.planner._parse_time(current_time))
    a, b = rt.agent["name"], other.agent["name"]
    line = f"{a} and {b} talked at {current_time} about: {topic.strip().rstrip('.')}."
    rt.stream.add_observations([line])
    other.stream.add_observations([line])

def log_outcomes(rt, other, outcomes):
    """Readable summary of what a conversation led to."""
    meeting = outcomes["meeting"]
    if meeting:
        when = f" at {meeting['start']}" if meeting.get("start") else ""
        log.event(f"agreed to meet in {meeting['location']}{when}: "
                  f"{meeting['activity']} ({meeting['minutes']} min)")
    for who in (rt, other):
        commitment = outcomes["commitments"].get(who.agent["name"])
        if commitment:
            log.event(f"{who.tag} committed: {commitment}")
    if not meeting and not outcomes["commitments"]:
        log.event("no commitments made")


def schedule_meeting(rt, other, meeting, current_time):
    """A meeting the two agreed on for later today goes into both their days
    (paper: Isabella's invitations become plans for the people she told).
    Before this, each agent wrote its own "collaborate with ..." blocks: Adam
    planned 11:15 with Eve, who had lunch 11:00-12:00, and Eve planned 3:30 with
    Adam, which wasn't in his plan. The block is pinned to the agreed place and
    kept by every replan; both agents remember the agreement."""
    start = rt.planner._parse_time(meeting["start"])
    end = rt.planner._fmt(start + timedelta(minutes=meeting["minutes"]))
    topic = (meeting.get("activity") or "").strip().rstrip(".") or "the project"
    line = (f"{rt.agent['name']} and {other.agent['name']} agreed to meet in "
            f"{meeting['location']} at {meeting['start']}: {topic}.")
    for a, b in ((rt, other), (other, rt)):
        block = {
            "activity": (f"meet {b.agent['name']} in {meeting['location']} about: {topic} "
                         f"from {meeting['start']} to {end}"),
            "location": meeting["location"],
            "start_time": meeting["start"],
            "duration": meeting["minutes"],
            "target_agent": b.agent["id"],
            "pinned": True,
            "agreed": True,
            "board_work": False,
        }
        a.broad_chunks = a.planner.insert_block(a.broad_chunks, block)
        a.planner.create_plan([block])
        a.stream.add_observations([line])
    log.event(f"{rt.tag} and {other.tag} put the meeting in their plans: "
              f"{meeting['location']} {meeting['start']}-{end}")


# ---------------------------------------------------------------------------
# THE WEBSITE'S DATA
# ---------------------------------------------------------------------------
def sandbox_root(runtimes):
    """Where the sandbox repository is, if the agents have real hands."""
    for rt in runtimes:
        root = getattr(rt.workspace, "root", None)
        if root is not None:
            return root
    return None


def publish_snapshot(runtimes, world):
    """The office as it is now, committed to the sandbox's main for the
    website (snapshot.py). Nothing without real hands; a failure is a warning,
    never the end of the run."""
    from workspace import WorkspaceError

    root = sandbox_root(runtimes)
    if root is None:
        return
    shot = snapshot.build(world, chronicle.data(),
                          streams={rt.agent["id"]: rt.stream for rt in runtimes},
                          plans={rt.agent["id"]: rt.broad_chunks for rt in runtimes})
    try:
        if snapshot.export(shot, root):
            log.debug(f"snapshot of the office published ({shot['generated']['day_name']} "
                      f"{shot['generated']['time']})")
    except WorkspaceError as e:
        log.warn(f"couldn't publish the snapshot of the office: {e}")


def _day_plans(runtimes):
    return {rt.agent["id"]: [c.get("activity") for c in rt.broad_chunks] for rt in runtimes}


def run_length():
    """(days, steps per day): GA_DAYS whole workdays (each GA_STEPS steps if
    that is set too), else one day of GA_STEPS (or NUM_STEPS) steps."""
    days = int(os.environ.get("GA_DAYS") or 0)
    steps = int(os.environ.get("GA_STEPS") or 0)
    if days > 0:
        return days, steps or DAY_STEPS
    return 1, steps or NUM_STEPS


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    days, day_steps = run_length()
    steps = days * day_steps
    world = load_world()
    budget = run_budget()                       # one cap on executor sessions for the run
    runtimes = [Agent.load(agent_id, world, *hands_for(world["agents"][agent_id], budget))
                for agent_id in ("adam", "eve")]

    length = (f"{days} days x {day_steps} steps" if days > 1 else f"{steps} steps")
    log.banner(f"Generative Agents · {START_TIME} · {length} x {STEP_MINUTES} min")
    log.info(f"  PAPER_STRICT={agent.PAPER_STRICT} EAGER_DECOMPOSITION={agent.EAGER_DECOMPOSITION} "
             f"REACT_COOLDOWN_STEPS={agent.REACT_COOLDOWN_STEPS} REFLECTION_THRESHOLD={agent.REFLECTION_THRESHOLD}")

    current_time = START_TIME
    clock.reset(current_time)
    project_board.clear_workload_changed()      # nothing carried over from a previous run
    project_board.clear_commitments()
    project_board.clear_tested()
    project_board.clear_reviews()
    persona.clear_previous_days()
    clear_talks()
    chronicle.start(world)

    for rt in runtimes:
        # What is already in the stream (the seeds, earlier days) was not
        # perceived today and doesn't count toward the first reflection.
        rt.reflected_through = max((m.id for m in rt.stream.memories), default=0)
        initial_planning(rt, world)
    chronicle.note_day(_day_plans(runtimes))
    publish_snapshot(runtimes, world)

    try:
        for step in range(1, steps + 1):
            if step > 1 and (step - 1) % day_steps == 0:
                current_time = next_workday(runtimes, world)
            step_start = time.time()
            clock.set_time(current_time)
            log.step_header(step, steps,
                            f"{clock.day_name()} {current_time}" if days > 1 else current_time)

            # Run each phase for every agent before moving on, so one agent's
            # actions don't leak into the other's perception mid-step.
            for rt in runtimes:
                rt.perceive(world)

            for rt in runtimes:
                rt.decide(step, current_time, world)

            resolve_conversations(runtimes, step, current_time, world)

            for rt in runtimes:
                rt.act(step, current_time, world)
                note_board_use(rt, current_time)

            for rt in runtimes:
                replan_after_board(rt, current_time, world)

            for rt in runtimes:
                rt.log_step()
            chronicle.note_step(world)

            for rt in runtimes:
                rt.reflect(current_time)

            # One disk write per step instead of one per action / retrieval.
            save_world(world)
            for rt in runtimes:
                rt.stream.flush()
            if step % SNAPSHOT_EVERY == 0:
                chronicle.save()
                publish_snapshot(runtimes, world)

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
        chronicle.save()
        publish_snapshot(runtimes, world)
        chronicle.stop()
        print_timing_summary()
        if budget.spent:
            log.info(f"  executor sessions cost about ${budget.spent:.2f} this run "
                     f"(estimate; the Anthropic console has the real bill)")


def next_workday(runtimes, world):
    """The workday is over: the agents go home, and the next one starts at
    START_TIME with fresh plans that continue from today. Returns its time."""
    log.event(f"{clock.day_name()} is over: everyone goes home")
    for rt in runtimes:
        end_day(rt)
    save_world(world)
    publish_snapshot(runtimes, world)
    clock.next_day(START_TIME)
    # Yesterday's promises, talks and board pings were about yesterday's plans.
    project_board.clear_workload_changed()
    project_board.clear_commitments()
    clear_talks()
    log.banner(f"{clock.day_name()} · day {clock.day()}")
    for rt in runtimes:
        start_day(rt, world)
    chronicle.note_day(_day_plans(runtimes))
    return START_TIME


if __name__ == "__main__":
    main()
