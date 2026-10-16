"""
agent.py - one generative agent: its state and its per-step phases.

The Agent (paper section 4) is built around ONE memory stream. Each step the
simulation loop (main.py) runs a phase for every agent before moving on to the
next phase, so the two act "simultaneously":

    1. perceive  - build observations, store only the NEW ones in memory
    2. decide    - continue the plan or react; the LLM is only asked when there
                   is something new to react to, and it returns a short
                   reaction that drives the replan (paper, section 4.3.1)
    3. act       - update agent/world state, or react: do the reaction now and
                   regenerate the rest of the day
    4. reflect   - when the importance of what the agent perceived since its
                   last reflection reaches REFLECTION_THRESHOLD (paper 4.2)

Conversations between two agents are resolved by main.py between decide and
act; the per-agent parts of them (settling, replanning around a promise) are
here.

Planning follows the paper's top-down scheme:

    broad plan  ->  hour chunks  ->  5-15 minute chunks

By default each level is expanded lazily (and cached) when the agent reaches
it. Set EAGER_DECOMPOSITION (or PAPER_STRICT) to decompose the whole plan up
front, exactly as the paper describes. Every level is saved to the memory
stream as a plan either way.

On a reaction, only the plan from the current time onward is regenerated;
what already happened today is kept (the chunk in progress is cut at "now").

Project board (see project_board.py): when an agent takes on or finishes a
task at the board, the rest of their day is replanned around their open tasks
once they step away from it, so later work blocks name the task and include
return trips to the board to keep it up to date.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import log
import project_board
from clock import STEP_MINUTES, step_time
from conversation import talked_recently_ids, last_talk
from log import timed
from memory_retriever import MemoryRetriever
from memory_stream import MemoryStream
from perceive import perceive_environment
from plan import DAY_END, Planner
from reflect import Reflector
from world import generate_interactions, update_agent_state


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Paper 4.2: an agent reflects when the summed importance of the events it
# has perceived since its last reflection exceeds a threshold (150 in the
# paper). Seeded memories don't count: they aren't events the agent perceived.
# None turns reflection off.
REFLECTION_THRESHOLD = 150

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

# After a react, don't ask the LLM about reacting again for this many steps.
# 2 steps = the reaction's own 10 minutes: an agent doesn't reconsider while
# still carrying out what it just decided. (It was 6, which mostly hid junk
# reactions; those are now filtered out instead, see pointless_reaction.) A
# teammate walking into the room still gets asked about, and a reaction that
# only moves the agent to another room doesn't start the cooldown.
REACT_COOLDOWN_STEPS = 2

# A reaction to go update the board is ignored while the agent is at the board,
# or within this many minutes of using it ("Move to the project_board to update
# tasks" was 7 of the 9 reactions in a run, mostly made at the board itself).
BOARD_REACTION_GAP_MINUTES = 30

# Plan chunks this short are already at the paper's 5-15 minute grain, so they
# aren't sent to the LLM for minute-level decomposition (e.g. board steps).
SHORT_CHUNK_MINUTES = 15

# Step 1 sees the whole room as "new". Treat it as a baseline and don't react.
SKIP_REACT_ON_FIRST_STEP = True

# --- Paper behaviour --------------------------------------------------------

# Paper 4.3.1: when an agent reacts, its plan is regenerated from that moment.
# True: carry out the reaction (a short block), then replan the rest of the day
# around it. False: the old shortcut, a 10-minute detour and back to the old plan.
REACT_REPLANS = True

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
    REACT_REPLANS = True


# ---------------------------------------------------------------------------
# THE AGENT
# ---------------------------------------------------------------------------
@dataclass
class Agent:
    """Everything one agent needs, built around ONE shared MemoryStream."""
    agent: dict                      # same dict object as world["agents"][id]
    stream: MemoryStream
    retriever: MemoryRetriever
    planner: Planner
    reflector: Reflector
    reflected_through: int = 0      # memory id: everything up to here counted for reflection

    # The plan, at three levels of detail. The two caches are keyed by
    # (start_time, activity), so a replan can never hit a stale entry.
    broad_chunks: list = field(default_factory=list)
    hour_cache: dict = field(default_factory=dict)
    minute_cache: dict = field(default_factory=dict)

    prev_obs: list = field(default_factory=list)
    prev_company: set = field(default_factory=set)    # teammates in the room last step
    last_board_at: Optional[datetime] = None          # last step spent using the board
    last_chunk: Optional[dict] = None
    last_react_step: int = -10_000
    override: Optional[dict] = None
    queued: Optional[dict] = None
    commitments: list = field(default_factory=list)   # promises already planned around

    # The hands (executor.py): who does the work on a task, and where. None
    # means the agent only plans and talks, as in the paper.
    executor: Optional[object] = None
    workspace: Optional[object] = None
    work_log: dict = field(default_factory=dict)       # task id -> work sessions on it
    shipped: set = field(default_factory=set)          # task ids merged into main
    reviewed: dict = field(default_factory=dict)       # teammate task id -> review round done

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

    @classmethod
    def load(cls, agent_id, world, executor=None, workspace=None):
        # Use the agent dict that lives inside world so there is a single source of
        # truth (previously load_agent() and load_world() gave two diverging copies).
        agent = world["agents"][agent_id]
        stream = MemoryStream(agent_id)
        retriever = MemoryRetriever(agent_id, stream)
        return cls(
            agent=agent,
            stream=stream,
            retriever=retriever,
            planner=Planner(agent, stream, retriever),
            reflector=Reflector(agent, stream, retriever),
            executor=executor,
            workspace=workspace,
        )

    # -- PHASE 1: PERCEIVE ---------------------------------------------------
    def perceive(self, world):
        with timed(f"{self.tag} perceive (stores new obs only)"):
            self.observations, self.new_obs = perceive_environment(
                self.agent, world, self.stream, self.prev_obs
            )
        self.prev_obs = self.observations

        log.debug(f"{self.tag} {len(self.observations)} observations, {len(self.new_obs)} new")
        for obs in self.observations:
            marker = "*" if obs in self.new_obs else " "
            log.debug(f"  {marker} {obs}")

    # -- PHASE 2: DECIDE -----------------------------------------------------
    def decide(self, step, current_time, world):
        planner = self.planner
        self.reaction = ""

        self.status = planner.get_current_status(self.agent, current_time)
        self.chunk = get_chunk_lazy(self, current_time, world)
        drop_spent_target(self, current_time, world)

        log.debug(f"{self.tag} status: {self.status}")
        if self.chunk:
            log.debug(f"{self.tag} planned: {self.chunk['activity']} (default loc: {self.chunk['location']})")
        else:
            log.debug(f"{self.tag} planned: (no chunk covers {current_time})")

        steps_since_react = step - self.last_react_step

        # Who just showed up in the room (or whose room this agent just walked into).
        company = teammates_here(self, world)
        arrivals = company - self.prev_company
        self.prev_company = company
        partner = planned_partner(self, world, current_time)

        # Gate 1: nothing to follow -> nothing to decide.
        if self.chunk is None:
            self.decision, reason = "1", "no current chunk"

        # Gate 1b: step 1 is just the baseline snapshot of the room.
        elif SKIP_REACT_ON_FIRST_STEP and step == 1:
            self.decision, reason = "1", "first step baseline (LLM skipped)"

        # Gate 1c: the plan says to work with a teammate who is right here: talk.
        # (Planned collaboration used to happen only if the react LLM happened to
        # choose "talk", so "collaborate with Eve 11-12" passed in silence.)
        elif partner is not None:
            activity = TIME_PREFIX.sub("", self.chunk["activity"]).strip()
            activity = self.planner._SPAN.sub("", activity).strip()
            self.decision = "2"
            if self.chunk.get("is_reaction") or self.chunk.get("agreed"):    # "greet Eve ...", "meet Eve ..."
                self.reaction = activity
            else:
                self.reaction = f"talk with {partner['name']} about: {activity}"
            reason = f"planned to work with {partner['name']}, who is here (LLM skipped)"

        # Gate 1d: recently replanned, give the new plan time to play out, unless
        # a teammate just walked in (that is worth a decision even mid-cooldown).
        elif steps_since_react < REACT_COOLDOWN_STEPS and not arrivals:
            self.decision, reason = "1", (
                f"react cooldown, {steps_since_react} steps since replan (LLM skipped)"
            )

        else:
            candidates = reaction_candidates(self)

            # Gate 2: nothing to react to -> skip the LLM entirely.
            if not candidates:
                self.decision, reason = "1", "nothing new observed (LLM skipped)"
            else:
                with timed(f"{self.tag} continue_plan_or_react (LLM)"):
                    raw = planner.continue_plan_or_react(
                        self.agent, self.status, self.chunk, candidates, world
                    )
                self.decision, self.reaction = parse_decision(raw)
                reason = f"LLM said {raw.strip()!r} after {len(candidates)} obs"

                skip = pointless_reaction(self, self.reaction, current_time, world) if self.decision == "2" else ""
                if skip:
                    reason += f" -> ignored ({skip})"
                    self.decision, self.reaction = "1", ""

                aimed = aim_at_owner(self, self.reaction, world) if self.decision == "2" else self.reaction
                if aimed != self.reaction:
                    reason += f" -> asks the task's owner: {aimed}"
                    self.reaction = aimed

                room = (relocation_target(self.reaction, self.agent, world, self.chunk["activity"])
                        if self.decision == "2" else None)
                if room is not None:
                    # "Move to office_1 to keep debugging" isn't a reaction to
                    # anything: the plan is fine, only the room was wrong. Move
                    # without replacing the plan or starting the react cooldown.
                    relocate(self, room, current_time)
                    reason += f" -> just a room change to {room} (no react, no cooldown)"
                    self.decision, self.reaction = "1", ""

        log.debug(f"{self.tag} decision {self.decision}: {reason}")

    # -- PHASE 3: ACT (and REACT) --------------------------------------------
    def act(self, step, current_time, world):
        """Carry out the step (or the reaction), then, if the step is work on a
        task of the agent's own, let the hands do that work (see work_on_task),
        and merge any of their tasks the board now shows as done."""
        if self.decision == "3":
            return
        self._carry_out(step, current_time, world)
        work_on_task(self, current_time, world)
        ship_approved_work(self, world)

    def _carry_out(self, step, current_time, world):

        if self.decision == "1":
            if self.chunk is None:
                log.debug(f"{self.tag} act: no chunk, state unchanged")
                return

            if SKIP_UNCHANGED_CHUNKS and self.chunk == self.last_chunk:
                if self.chunk.get("target_agent"):
                    # cheap check: did the target move?
                    old = self.agent["current_location"]
                    update_agent_state(self.agent, self.chunk, world)
                    if self.agent["current_location"] != old:
                        apply_chunk(self, world)
                    else:
                        log.debug(f"{self.tag} act: target still here, skipping update")
                else:
                    log.debug(f"{self.tag} act: same chunk as last step, skipping update")
                    refresh_board_step_text(self, world)
                return

            apply_chunk(self, world)

        elif self.decision == "2":
            if not self.react(step, current_time, world):
                return

        self.last_chunk = self.chunk

    def react(self, step, current_time, world):
        """React (paper 4.3.1): do the reaction now, as a short block, and, with
        REACT_REPLANS, regenerate the rest of the day from when it ends, so the plan
        takes what just happened into account instead of resuming as if nothing had."""
        log.event(f"{self.tag} reacts: {self.reaction}")
        self.last_react_step = step
        self.override = self.planner.make_reaction_chunk(
            self.agent, current_time, self.reaction, world
        )
        self.chunk = self.override
        log.debug(f"{self.tag} reaction chunk: {self.chunk}")
        apply_chunk(self, world)

        if REACT_REPLANS:
            reaction = self.reaction
            after = step_time(current_time, self.override.get("duration") or STEP_MINUTES)
            if replan_rest_of_day(self, after, f"{self.agent['name']} reacted to what was "
                                  f"happening and decided to: {reaction}", world,
                                  "replan after reaction"):
                log.event(f"{self.tag} replanned the day after reacting: {reaction}")
                absorb_board_changes(self)          # the new plan already reflects the board
        return True

    # -- PHASE 4: REFLECT ----------------------------------------------------
    def importance_since_reflection(self):
        """Summed importance of the events perceived since the last reflection
        (paper 4.2). Plans and reflections aren't perceived events."""
        return sum(m.importance for m in self.stream.memories
                   if m.id > self.reflected_through and m.memory_type == "observation")

    def reflect(self, current_time):
        if REFLECTION_THRESHOLD is None:
            return
        pending = self.importance_since_reflection()
        if pending < REFLECTION_THRESHOLD:
            return

        log.reflection_header(self.tag, current_time)
        log.debug(f"{self.tag} importance since last reflection: {pending} >= {REFLECTION_THRESHOLD}")
        with timed(f"{self.tag} reflect (LLM x3+)"):
            reflections = self.reflector.reflect()
        # Counted either way, so a pass that finds nothing isn't retried every step.
        self.reflected_through = max((m.id for m in self.stream.memories), default=0)

        if not reflections:
            log.event(f"{self.tag} had nothing new to reflect on")
            return

        for reflection in reflections:
            log.reflection_item(reflection.description)

        # New insights change who the agent is: rebuild the summary next time it's needed.
        self.planner.invalidate_summary()

    def log_step(self):
        """One compact line per agent: where they are, what they're doing, what they're using."""
        a = self.agent
        log.agent_line(self.tag, a["current_location"], a["current_action"], a["interacting_with"])


# ---------------------------------------------------------------------------
# PLANNING
# ---------------------------------------------------------------------------
def expand_hours(rt, broad, world=None):
    """broad block -> hour chunks (cached, saved to the memory stream)."""
    p = rt.planner
    bkey = (broad["start_time"], broad["activity"])

    if bkey not in rt.hour_cache and (broad.get("board_work") or broad.get("pinned")):
        # One board visit is one step: broken into 5-minute pieces,
        # every piece was its own board LLM call (20 calls, 8 changes, in the run).
        rt.hour_cache[bkey] = [broad]

    if bkey not in rt.hour_cache:
        with timed(f"{rt.tag} expand broad {bkey[0]} -> hours (LLM)"):
            hours = p.decompose_chunks([broad], rt.agent, focus=p.work_focus(rt.agent, world), world=world)
        if hours:
            p.create_plan(hours)
        # If the output didn't parse, fall back to the coarser chunk.
        rt.hour_cache[bkey] = hours or [broad]

    return rt.hour_cache[bkey]


def expand_minutes(rt, hour, world=None):
    """hour chunk -> 5-15 minute chunks (cached, saved to the memory stream)."""
    p = rt.planner
    hkey = (hour["start_time"], hour["activity"])

    if hkey not in rt.minute_cache and ((hour.get("duration") or 60) <= SHORT_CHUNK_MINUTES
                                        or hour.get("board_work") or hour.get("pinned")):
        # Already short, one board visit, or an agreed meeting: keep whole.
        rt.minute_cache[hkey] = [hour]

    if hkey not in rt.minute_cache:
        with timed(f"{rt.tag} expand hour {hkey[0]} -> minutes (LLM)"):
            minutes = p.decompose_hour_chunks([hour], rt.agent, focus=p.work_focus(rt.agent, world), world=world)
        if minutes:
            p.create_plan(minutes)
        rt.minute_cache[hkey] = minutes or [hour]

    return rt.minute_cache[hkey]


def expand_plan(rt, chunks, world=None):
    """Eagerly decompose the given broad chunks all the way down (the paper's
    up-front version). Fills the same caches the lazy path uses."""
    for broad in chunks:
        for hour in expand_hours(rt, broad, world):
            expand_minutes(rt, hour, world)


def initial_planning(rt, world):
    """Day plan in broad strokes, then (optionally) the full decomposition."""
    with timed(f"{rt.tag} initial broad plan (LLM)"):
        daily_plan = rt.planner.generate_initial_plan(rt.agent, world)

    p = rt.planner
    rt.broad_chunks = p.thin_board_visits(p.cap_board_blocks(
        p.create_chunks(daily_plan, rt.agent, world)))
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
        expand_plan(rt, rt.broad_chunks, world)


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

    hour = p.get_current_minute_chunk(current_time, expand_hours(rt, broad, world))
    if hour is None or not p.covers(hour, current_time):
        # The hour decomposition left a gap (the model returned too few hours).
        # Fill it from the broad block instead of replaying the last minute action
        # of an hour that already ended.
        hour = gap_hour(rt, broad, current_time, world)
        if hour is None:
            return None

    return p.get_current_minute_chunk(current_time, expand_minutes(rt, hour, world))


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

    what = own_work(rt, world)
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


def own_work(rt, world):
    """See Planner.own_work."""
    return rt.planner.own_work(rt.agent, world)


def gap_hour(rt, broad, current_time, world=None):
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
    for hour in expand_hours(rt, broad, world):
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
# HELPERS FOR DECIDE
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



TIME_PREFIX = re.compile(r"^\d{1,2}:\d{2}\s*[ap]m:\s*", re.IGNORECASE)
_MOVE_VERB = re.compile(
    r"^\s*(?:let'?s\s+)?(?:move|go|head|walk|return|relocate|switch|get)\w*\b(?:\s+back)?\s+to\b",
    re.IGNORECASE)


def _gist(text):
    text = TIME_PREFIX.sub("", text or "").lower()
    return " ".join(re.findall(r"[a-z0-9-]+", text))


_FILLER = {"from", "with", "into", "onto", "then", "that", "this", "their", "there"}


def _content_words(text):
    words = set(re.findall(r"[a-z]{4,}", _gist(text)))
    return words - _FILLER


def _is_the_plan(reaction, planned, share=0.75):
    """Does the reaction just restate the planned step? At 7:05 both agents
    "reacted" with "go to the conference room to check the project board" while
    their plan said "check the project board in the conference room", and each
    reaction cost a full replan."""
    wanted = _content_words(reaction)
    return bool(wanted) and len(wanted & _content_words(planned)) / len(wanted) >= share


def _named_teammate(text, agent, world):
    return any(re.search(rf"\b{o['name'].split()[0]}\b", text or "", re.IGNORECASE)
               for o in world["agents"].values() if o["id"] != agent["id"])


def pointless_reaction(rt, reaction, current_time, world):
    """Why a reaction would only repeat what is already happening, or "".

    - the reaction the agent is already carrying out (Eve reacted "Move to the
      project_board to update task T-2 ..." at 8:30 and again at 8:35), or what
      the plan already says to do right now;
    - going to update the board while at the board, or right after using it."""
    chunk = rt.chunk or {}
    if chunk.get("is_reaction") and _gist(reaction) == _gist(chunk.get("activity")):
        return "already doing this"
    conflict = project_board.start_conflict(world, rt.agent["id"], reaction)
    if conflict:
        return conflict
    routine = object_only(reaction, rt.agent, world)
    if routine:
        return routine
    if chunk.get("activity") and _is_the_plan(reaction, chunk["activity"]):
        return "that is the plan already"
    if re.match(r"\s*(?:continue|keep|resume|carry on|stay)\b", reaction or "", re.IGNORECASE) \
            and not _named_teammate(reaction, rt.agent, world):
        return "that is the plan already"
    if not project_board.mentions_board_explicitly(reaction):
        return ""
    if chunk.get("board_work") and rt.agent["interacting_with"] == project_board.BOARD_OBJECT:
        return "already at the board"
    if chunk.get("board_work") and not chunk.get("is_reaction"):
        # "go to the conference room to review the project board" while the
        # planned step is a board visit: the plan already takes them there.
        return "that is the plan already"
    now = rt.planner._parse_time(current_time)
    if (rt.last_board_at is not None and now is not None
            and now - rt.last_board_at < timedelta(minutes=BOARD_REACTION_GAP_MINUTES)):
        return f"used the board at {rt.planner._fmt(rt.last_board_at)}"
    return ""


# What a reaction says besides naming an object, a room or a person, when it
# is only about that object ("check the time_clock", "go to the reception to
# check the time_clock with Adam Beck").
_ROUTINE_WORDS = {
    "a", "an", "the", "to", "in", "at", "on", "of", "and", "with", "for", "if", "it", "is",
    "whether", "that", "this", "my", "his", "her", "their", "again", "quickly", "briefly",
    "check", "checks", "checking", "look", "looks", "looking", "see", "use", "using", "inspect",
    "verify", "confirm", "make", "sure", "go", "goes", "head", "walk", "move", "return", "back",
    "working", "works", "properly", "correctly", "status", "state", "room", "office", "over",
}


def object_only(reaction, agent, world):
    """Why a reaction is only about one of the room's ordinary objects, or "".

    At 7:05 both agents reacted to the time clock (Adam: "check the time_clock
    in the reception"; Eve: "go to the reception to check the time_clock with
    Adam Beck"), and Eve's started a 70-second conversation in which Adam made
    up work he had done. The board is excluded (it has its own rules), and so is
    anything that also does something ("fix the broken coffee machine")."""
    text = (reaction or "").lower()
    if TASK_ID.search(text) or project_board.mentions_board_explicitly(text):
        return ""
    objects = {name for loc in world["locations"].values() for name in loc["objects"]
               if name != project_board.BOARD_OBJECT}
    named = [o for o in sorted(objects) if o in text or o.replace("_", " ") in text]
    if not named:
        return ""
    names = named + list(world["locations"]) + [
        part.lower() for a in world["agents"].values() for part in a["name"].split()]
    for name in sorted(names, key=len, reverse=True):
        for form in {name.lower(), name.lower().replace("_", " ")}:
            text = re.sub(rf"\b{re.escape(form)}\b", " ", text)
    rest = set(re.findall(r"[a-z]+", text)) - _ROUTINE_WORDS
    return "" if rest else f"only about the {named[0].replace('_', ' ')}"


def aim_at_owner(rt, reaction, world):
    """A reaction about a teammate's task goes to that teammate.

    "Check on T-2 progress and confirm if it affects T-1 or T-5" (T-2 is Eve's)
    named nobody, so Adam spent 15 minutes on it alone in his office. Asking
    Eve is how he'd find out; the conversation cooldown is skipped for a task id."""
    if not reaction or _named_teammate(reaction, rt.agent, world) \
            or project_board.mentions_board_explicitly(reaction):
        return reaction
    task = project_board.teammate_task(world, rt.agent["id"], reaction)
    if task is None:
        return reaction
    owner = world["agents"][task["assignee"]]["name"]
    return f"talk with {owner} about {task['id']}: {reaction}"


def note_board_use(rt, current_time):
    """Remember when the agent last used the board (see pointless_reaction)."""
    if rt.agent["interacting_with"] == project_board.BOARD_OBJECT:
        rt.last_board_at = rt.planner._parse_time(current_time)


def drop_spent_target(rt, current_time, world=None):
    """A planned "collaborate with Eve" step right after the two of them talked
    can't become a conversation (the conversation cooldown), so following Eve
    only means standing in her office in silence. The talk was the meeting:
    the agent goes back to their own work in their own office instead, until
    the cooldown is over. (Eve spent 8:10 "attending a stand-up meeting with
    Adam Beck" alone in her office, right after the stand-up talk.)"""
    chunk = rt.chunk
    if (not chunk or not chunk.get("target_agent") or chunk.get("is_reaction")
            or chunk.get("board_work")):
        return
    target = chunk["target_agent"]
    if chunk.get("agreed"):
        if meeting_pending(rt, chunk):
            return                         # not held yet: go and meet
    elif chunk.get("pinned") or not talked_recently_ids(
            rt.agent["id"], target, rt.planner._parse_time(current_time)):
        return
    relocate(rt, rt.agent.get("office") or rt.agent["current_location"], current_time)
    start = rt.override["start_time"]
    rt.override["activity"] = f"{start}: {own_work(rt, world)}"
    log.debug(f"{rt.tag} just talked with {target}; instead of '{chunk['activity']}': "
              f"{rt.override['activity']}")


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
    if not chunk or not chunk.get("target_agent"):
        return None
    other = world["agents"].get(chunk["target_agent"])
    if (other is None or other["id"] == rt.agent["id"]
            or other["current_location"] != rt.agent["current_location"]):
        return None
    if chunk.get("agreed") and rt.agent["current_location"] != chunk["location"]:
        return None                        # the meeting is held where they agreed
    if talked_recently_ids(rt.agent["id"], other["id"], rt.planner._parse_time(current_time)) \
            and not meeting_pending(rt, chunk):
        return None
    return other


def meeting_pending(rt, chunk):
    """An agreed meeting the two haven't held yet: no talk since it started.
    It starts a conversation even inside the conversation cooldown (meeting is
    the point of it), and only one."""
    if not chunk or not chunk.get("agreed"):
        return False
    last = last_talk(rt.agent["id"], chunk["target_agent"])
    start = rt.planner._parse_time(chunk.get("start_time"))
    return last is None or start is None or last < start


_SAME_WORK = re.compile(r"\b(?:continu\w*|keep\w*|resum\w*|carry on|as planned|get back to|finish\w*)\b",
                        re.IGNORECASE)


def relocation_target(reaction, agent, world, current_activity=""):
    """The room a reaction only moves the agent to ("Move to office_1 to keep
    debugging"), or None if it is a real reaction: it names a teammate, names
    no room or the room the agent is already in, is about the board, or moves
    the agent somewhere to do something else. ("Move to the conference room
    to update the project board" was taken for a room change twice, and Eve
    went on designing the homepage in the conference room.)"""
    if not reaction or not _MOVE_VERB.search(reaction):
        return None
    if project_board.mentions_board_explicitly(reaction):
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

    # What the move is for: the words after the room ("... office_1 to set up the env").
    room = rooms[0]
    for name in (room, room.replace("_", " "), "my office", "own office"):
        if name in text:
            purpose = text.split(name, 1)[1]
            break
    else:
        purpose = ""
    purpose = re.sub(r"^\W*(?:to|and|for)\b", "", purpose).strip()
    if not purpose or _SAME_WORK.search(purpose):
        return room
    current = {w for w in re.findall(r"[a-z]{4,}", current_activity.lower())}
    wanted = {w for w in re.findall(r"[a-z]{4,}", purpose)}
    return room if current & wanted else None


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


TASK_ID = re.compile(r"\bT-\d+\b", re.IGNORECASE)


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

def replan_rest_of_day(rt, start, reason, world, label, extra_guidance=""):
    """Regenerate the plan from `start` on, keeping what already happened.
    If the agent has tasks on the board, the new plan is built around them.
    Returns True if the new plan was used."""
    done, remaining = rt.planner.split_plan(rt.broad_chunks, start)
    kept = teammate_blocks(rt, remaining)
    guidance = rt.planner.task_guidance(rt.agent, world) + extra_guidance
    if kept:
        lines = "\n".join(f"        - {c['activity']}" for c in kept)
        guidance += f"""
        Keep these plans with teammates at the same time and place; the teammate
        is counting on them:
{lines}
"""

    with timed(f"{rt.tag} {label} (LLM)"):
        items = rt.planner.regenerate_plan(
            rt.agent, start, reason, done, remaining, guidance=guidance, world=world
        )
    new_chunks = rt.planner.create_chunks(items, rt.agent, world)

    if not new_chunks or any(c["start_time"] is None for c in new_chunks):
        log.warn(f"{rt.tag}: replan unusable, keeping old plan")
        return False

    new_chunks = rt.planner.keep_blocks(new_chunks, kept)
    new_chunks = rt.planner.cap_board_blocks(new_chunks)
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


def teammate_blocks(rt, remaining):
    """Remaining plan blocks with a teammate that a replan should keep: planned
    work or meetings with them (not reactions or board visits), except ones
    that start within the conversation cooldown of their last talk (they just
    covered it; see replan_after_conversation)."""
    out = []
    for c in remaining:
        target = c.get("target_agent")
        if c.get("agreed") and target:
            out.append(c)                  # both agreed to it: always kept
            continue
        if not target or c.get("is_reaction") or c.get("board_work") or c.get("pinned"):
            continue
        start = rt.planner._parse_time(c.get("start_time"))
        if start is None or talked_recently_ids(rt.agent["id"], target, start):
            continue
        out.append(c)
    return out


def replan_after_conversation(rt, commitment, current_time, world, partner=None):
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
    if partner:
        # Eve's plan kept "collaborate with Adam 7:30-8:00" right after they had
        # talked, and she spent it standing silently in his office.
        extra += f"""
        {rt.agent['name']} and {partner} just talked, so leave out any block in the
        next hour that is only about talking or collaborating with {partner}, except
        a meeting the two of them agreed on (keep that one as it is).
        """
    if project_board.is_board_commitment(commitment):
        project_board.add_commitment(rt.agent["id"], commitment)
        extra += f"""
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


def settle_after_conversation(rt, partner, step, current_time, world):
    """Hold both agents in place for one step; their plans take over after that
    (including a meeting they just agreed to hold right away)."""
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
# HELPERS FOR ACT
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


def refresh_board_step_text(rt, world):
    """Second step of a board visit: the visit isn't redone (no LLM), but the
    shown action follows the board, so "starting T-1 (move it to in progress)"
    becomes "reporting progress on T-1 (already in progress)" once it has."""
    if rt.agent["interacting_with"] != project_board.BOARD_OBJECT or not rt.chunk.get("board_work"):
        return
    current = project_board.reconcile_activity(world, rt.agent["id"], rt.chunk["activity"])
    if current != rt.chunk["activity"]:
        update_agent_state(rt.agent, {**rt.chunk, "activity": current}, world,
                           location=rt.agent["current_location"])


# ---------------------------------------------------------------------------
# THE HANDS: work on a task (executor.py)
# ---------------------------------------------------------------------------
def at_desk_work(rt, world):
    """Is the current step work at the agent's own desk: not a break, a board
    visit, a talk, or time with a teammate?"""
    chunk = rt.chunk
    activity = (chunk or {}).get("activity") or ""
    return not (not chunk or "project" not in world or chunk.get("board_work")
                or chunk.get("target_agent")
                or rt.agent["current_location"] != rt.agent.get("office")
                or rt.planner._BREAK.search(activity)
                or project_board.mentions_board_explicitly(activity))


def task_at_hand(rt, world):
    """The task the agent's current step is work on, or None.

    Only a task the agent owns and has in progress, worked on at their own
    desk: not a break, a board visit, a talk, or time with a teammate. Helping
    a teammate means working on their task together with them, which is a
    conversation, never the hands taking it over (lanes)."""
    if not at_desk_work(rt, world):
        return None
    activity = rt.chunk.get("activity") or ""
    tasks = {t["id"]: t for t in project_board.get_project(world)["tasks"]}
    named = [tasks[i] for i in project_board.task_ids_in(activity) if i in tasks]
    if any(t["assignee"] != rt.agent["id"] for t in named):
        return None                                     # a teammate's task
    candidates = named or project_board.open_tasks_for(world, rt.agent["id"])
    working = [t for t in candidates if t["status"] == "in_progress"]
    return working[0] if working else None


def _work_block(rt, current_time):
    """The plan block the step belongs to: one work session per block."""
    if rt.override:
        return rt.override.get("start_time")
    broad = rt.planner.get_current_minute_chunk(current_time, rt.broad_chunks)
    return broad.get("start_time") if broad else current_time


def work_on_task(rt, current_time, world):
    """Hand the task at hand to the executor, once per plan block, and remember
    what came of it. The clock waits while the hands work.

    A finished session (tests pass) becomes a promise to move the task to
    review at the next board visit; a blocked one sends the agent to the board
    to mark it blocked, with the reason. A task already finished isn't worked
    on again until something on the board changes it (a teammate's note in
    review, say). Returns the ExecResult, or None if there was no work to do."""
    if rt.executor is None:
        return None
    if not rt.executor.available():
        if not getattr(rt.executor, "_said_spent", False):
            rt.executor._said_spent = True
            log.event(f"{rt.tag}'s hands are done for this run: the executor session budget is spent")
        return None
    block = _work_block(rt, current_time)
    if any(s["block"] == block for log_ in rt.work_log.values() for s in log_):
        return None                                     # this block's session already ran
    review = review_at_hand(rt, world)
    if review is not None:
        return review_teammate_work(rt, review, block, current_time, world)
    task = task_at_hand(rt, world)
    if task is None:
        return None
    sessions = rt.work_log.setdefault(task["id"], [])
    if sessions and sessions[-1]["block"] == block:
        return None                                     # this block's session already ran
    if sessions and sessions[-1]["result"].finished and sessions[-1]["history"] == len(task["history"]):
        return None                                     # finished, and nothing new since

    log.event(f"🔧 {rt.tag} works on {task['id']} '{task['title']}'")
    with timed(f"{rt.tag} executor on {task['id']}"):
        result = _use_hands(rt, lambda: rt.executor.run(rt, task, rt.workspace))
    if result is None:
        return None
    sessions.append({"block": block, "time": current_time, "result": result,
                     "history": len(task["history"])})
    remember_work(rt, task, result, current_time)

    if result.blocked:
        head_to_board(rt, f"mark {task['id']} blocked on the project board: {result.blocked_reason}",
                      current_time, world)
    elif result.finished:
        # What the reviewer should check goes on the card with the move.
        files = ", ".join(result.files_changed[:6])
        project_board.add_commitment(
            rt.agent["id"], f"move {task['id']} to review on the project board, noting for the "
                            f"reviewer: {result.summary.strip().rstrip('.')}. Changed: {files}. "
                            f"Tests pass on branch task/{task['id']}.")
    return result


def _use_hands(rt, session):
    """Run one executor session. If the executor itself fails (the model API
    is unreachable, out of credit, misconfigured; the workspace's git breaks),
    the hands stop for the rest of the run and the agent carries on as in the
    paper. It isn't something the agent experienced, so nothing is remembered."""
    from llm import LLMError
    from workspace import WorkspaceError

    try:
        return session()
    except (LLMError, WorkspaceError) as e:
        log.warn(f"{rt.tag}'s hands stop for this run: {e}")
        rt.executor = None
        return None


def remember_work(rt, task, result, current_time):
    """The session's outcome, as observations in the agent's memory stream."""
    name, ref = rt.agent["name"], f"{task['id']} '{task['title']}'"
    lines = [f"{name} worked on {ref} at {current_time}: {result.summary.strip().rstrip('.')}."]
    if result.files_changed:
        shown = ", ".join(result.files_changed[:6])
        more = f" and {len(result.files_changed) - 6} more" if len(result.files_changed) > 6 else ""
        lines.append(f"{name} changed {shown}{more} for {task['id']}.")
    if result.tests_passed is not None:
        lines.append(f"The tests for {task['id']} {'pass' if result.tests_passed else 'fail'}.")
    if result.blocked:
        lines.append(f"{name} is stuck on {task['id']}: {result.blocked_reason.strip().rstrip('.')}.")
    rt.stream.add_observations(lines)
    for line in lines:
        log.event(f"🔧 {line}")


def head_to_board(rt, commitment, current_time, world, minutes=10):
    """Go to the board right after this step to do `commitment` there (the
    board visit is where the board changes; the promise tells it what to do)."""
    room = project_board.board_location(world)
    if room is None:
        return
    project_board.add_commitment(rt.agent["id"], commitment)
    start = step_time(current_time)
    visit = {
        "activity": f"{start}: {commitment}",
        "location": room,
        "start_time": start,
        "duration": minutes,
        "target_agent": None,
        "is_reaction": True,
        "pinned": True,
        "board_work": True,
    }
    if rt.override and rt.planner._parse_time(rt.override["start_time"]) <= rt.planner._parse_time(start)             < rt.planner._parse_time(rt.override["start_time"]) + timedelta(minutes=rt.override["duration"]):
        rt.queued = visit                       # after the step already under way
    else:
        rt.override = visit
    log.event(f"{rt.tag} heads to the board to: {commitment}")


def ship_approved_work(rt, world):
    """A task of the agent's own that is done on the board (a teammate
    approved it in review, then its owner moved it to done) has its branch
    merged into the sandbox's main, so every task started after it builds on
    it. A merge that fails is remembered, not retried."""
    if rt.workspace is None or not hasattr(rt.workspace, "integrate") or "project" not in world:
        return
    from workspace import WorkspaceError

    name = rt.agent["name"]
    for task in project_board.get_project(world)["tasks"]:
        if (task["assignee"] != rt.agent["id"] or task["status"] != "done"
                or task["id"] not in rt.work_log or task["id"] in rt.shipped):
            continue
        rt.shipped.add(task["id"])
        try:
            rt.workspace.integrate(task["id"])
            line = f"{name} merged {task['id']} '{task['title']}' into the website's main branch."
        except WorkspaceError as e:
            line = f"{name} couldn't merge {task['id']} into the website's main branch: {e}"
        rt.stream.add_observations([line])
        log.event(f"🚢 {line}")


# ---------------------------------------------------------------------------
# THE HANDS: review a teammate's task
# ---------------------------------------------------------------------------
def _review_round(task):
    """How many times the task has gone to review: a new round needs a new review."""
    return sum(1 for e in task["history"] if e.get("event", "").endswith("-> review"))


def review_at_hand(rt, world):
    """A teammate's task waiting in review that the agent hasn't reviewed this
    round, if the step is desk work. One named in the step comes first."""
    if not at_desk_work(rt, world):
        return None
    waiting = [t for t in project_board.get_project(world)["tasks"]
               if t["status"] == "review" and t["assignee"] and t["assignee"] != rt.agent["id"]
               and rt.agent["id"] not in t["approved_by"]
               and rt.reviewed.get(t["id"]) != _review_round(t)]
    named = set(project_board.task_ids_in(rt.chunk.get("activity") or ""))
    waiting.sort(key=lambda t: t["id"] not in named)
    return waiting[0] if waiting else None


def review_teammate_work(rt, task, block, current_time, world):
    """The hands read and run the teammate's published branch. The verdict is
    remembered and becomes a promise for the board: approve the task there, or
    leave a note asking for the changes. Approving stays a board action."""
    owner = world["agents"][task["assignee"]]["name"]
    log.event(f"🔍 {rt.tag} reviews {owner}'s {task['id']} '{task['title']}'")
    with timed(f"{rt.tag} executor review of {task['id']}"):
        result = _use_hands(rt, lambda: rt.executor.review(rt, task, rt.workspace, owner_name=owner))
    if result is None:
        return None
    rt.work_log.setdefault(f"review {task['id']}", []).append(
        {"block": block, "time": current_time, "result": result, "history": len(task["history"])})

    name, ref = rt.agent["name"], f"{owner}'s {task['id']} '{task['title']}'"
    if result.verdict == "none":
        line = f"{name} couldn't review {ref} yet: {result.summary.strip().rstrip('.')}."
        rt.stream.add_observations([line])
        log.event(f"🔍 {line}")
        return result
    rt.reviewed[task["id"]] = _review_round(task)
    lines = [f"{name} reviewed {ref} at {current_time}: {result.summary.strip().rstrip('.')}."]
    if result.tests_passed is not None:
        lines.append(f"The tests on {task['id']}'s branch {'pass' if result.tests_passed else 'fail'}.")
    if result.verdict == "approve":
        lines.append(f"{name} thinks {task['id']} is ready to approve.")
        promise = f"approve {task['id']} on the project board: {result.summary.strip().rstrip('.')}"
    else:
        asks = "; ".join(result.changes) or result.summary.strip().rstrip(".")
        lines.append(f"{name} wants changes to {task['id']}: {asks}.")
        promise = f"leave a note on {task['id']} on the project board asking {owner} for changes: {asks}"
    rt.stream.add_observations(lines)
    project_board.add_commitment(rt.agent["id"], promise)
    for line in lines:
        log.event(f"🔍 {line}")
    return result
