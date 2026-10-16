from datetime import datetime, timedelta
import json
import math
import re

import llm
import log
import project_board
from llm import CTX
from memory_retriever import MemoryRetriever
from memory_stream import MemoryStream

TIME_FORMAT = "%I:%M %p"

AGENT_PROFILES = {
    "adam": """Name: Adam Beck (age: 24)

        Innate traits: collaborative, technical, growth-oriented

        Adam Beck is a backend software engineer who primarily develops software using Python and TypeScript.

        He enjoys building reliable and scalable software systems and values clean code.

        Adam Beck's team has been asked to build and deploy the company website. Adam owns the backend side of it: the API, the data it needs, and getting the site deployed. Eve Smith owns the frontend.

        The team keeps track of its work on the project board in the conference room. Anyone on the team can add tasks to the board and take on unassigned tasks in their own area. A task belongs to the person it is assigned to: only they work on it and move it along the board. When Adam wants to help with one of Eve's tasks, he offers to work on it together with her, and it is her call. Adam checks the project board at the start of each workday and adds any backend work the website still needs. When Adam starts, finishes, or gets stuck on a task, he walks to the project board to update it, and he updates his tasks there before lunch and before clocking out. A task is only done once a teammate has reviewed and approved it, so when a teammate's task is in review, Adam looks it over with them and approves it at the board if it is ready. When Adam hands a task to a teammate, he leaves a note on it saying what it needs.

        Adam Beck works closely with Eve Smith and enjoys collaborating on technical challenges.

        Adam Beck hopes to become a software architect. He wants to build software that positively impacts many people.

        On Wednesday, Adam:
        1) clocked in and started work at 7:00 am,
        2) attended the kickoff meeting for the company website project from 8:00 am to 9:00 am,
        3) researched hosting and API options for the website from 9:00 am to 12:00 pm,
        4) took a lunch break in the break room from 12:00 pm to 1:00 pm,
        5) collaborated with coworkers on technical challenges from 1:00 pm to 2:00 pm,
        6) reviewed documentation and set up his development environment from 2:00 pm to 5:00 pm,
        7) clocked out and finished work at 5:00 pm.""",
    "eve": """Name: Eve Smith (age: 21)

        Innate traits: shy, detail-oriented, thoughtful

        Eve Smith is a frontend software engineer who primarily develops software using TypeScript and React.

        She enjoys building polished and accessible user interfaces and values clean code.

        Eve Smith's team has been asked to build and deploy the company website. Eve owns the frontend side of it: the pages, the layout, and how the site looks and feels. Adam Beck owns the backend.

        The team keeps track of its work on the project board in the conference room. Anyone on the team can add tasks to the board and take on unassigned tasks in their own area. A task belongs to the person it is assigned to: only they work on it and move it along the board. When Eve wants to help with one of Adam's tasks, she offers to work on it together with him, and it is his call. Eve checks the project board at the start of each workday and adds any frontend work the website still needs. When Eve starts, finishes, or gets stuck on a task, she walks to the project board to update it, and she updates her tasks there before lunch and before clocking out. A task is only done once a teammate has reviewed and approved it, so when a teammate's task is in review, Eve looks it over with them and approves it at the board if it is ready. When Eve hands a task to a teammate, she leaves a note on it saying what it needs.

        Eve Smith works closely with Adam Beck, and prefers collaborating one-on-one or in writing on technical challenges.

        Eve Smith hopes to become a lead frontend engineer. She wants to build software that feels simple and welcoming for many people to use.

        On Wednesday, Eve:
        1) clocked in and started work at 7:00 am,
        2) attended the kickoff meeting for the company website project from 8:00 am to 9:00 am,
        3) sketched page layouts for the website from 9:00 am to 12:00 pm,
        4) took a lunch break in the break room from 12:00 pm to 1:00 pm,
        5) collaborated with coworkers on technical challenges from 1:00 pm to 2:00 pm,
        6) reviewed documentation and set up her development environment from 2:00 pm to 5:00 pm,
        7) clocked out and finished work at 5:00 pm.""",
}

ROOM_PURPOSES = {
    "reception": "arriving, clocking in/out, visitors",
    "break_room": "meals, coffee, breaks, casual chat",
    "conference_room": "meetings, group discussion, collaborating, presentations, "
                       "and the team's project board (checking, adding and updating tasks)",
}

# The workday. The first plan item is given to the model, as in the paper.
DAY_START = "7:00 am"
DAY_END = "5:00 pm"
FIRST_PLAN_ITEM = f"clock in at reception from {DAY_START} to 7:05 am"
CLOCK_IN_MINUTES = 5            # "clock in at 7:00 am" with no end time used to fill the whole first hour

# The example line shown to the model in the initial-plan prompt.
PLAN_EXAMPLES = {
    "adam": "2. check the project board in the conference room from 7:05 am to 7:20 am",
    "eve": "2. check the project board in the conference room from 7:05 am to 7:20 am",
}


_PAST_CLAIM = re.compile(
    r"^\s*(?:i'?ve|i have|i had|i already|already|i\s+(?:approved|marked|moved|finished|"
    r"completed|reviewed|did|made|merged|deployed|started))\b", re.IGNORECASE)
_THEN_WILL = re.compile(r"\b(?:and|so|then)\s+(?:i\s*)?(?:will|'ll)\s+(.+)$", re.IGNORECASE)


def future_part(commitment):
    """What a commitment says the person will do, without claims about what
    they already did. "I've approved T-1 and will provide input as needed" ->
    "I will provide input as needed"; "I've approved T-1" -> "" (not a
    commitment). In the run behind this, a claimed approval that never happened
    became a promise, a replan and a trip to the board to act on it."""
    text = (commitment or "").strip()
    if not _PAST_CLAIM.search(text):
        return text
    then = _THEN_WILL.search(text)
    return f"I will {then.group(1).strip()}" if then else ""


class Planner:
    def __init__(self, agent, memory_stream=None, memory_retriever=None):
        self.memory_stream = memory_stream or MemoryStream(agent["id"])
        self.memory_retriever = memory_retriever or MemoryRetriever(agent["id"], self.memory_stream)
        self._summary_cache = {}

    def _parse_numbered(self, text):
        items = []
        for line in text.splitlines():
            m = re.match(r"^\d+\.\s*(.+)$", line.strip())
            if m:
                items.append(m.group(1).replace("**", "").strip())
        return items


    def split_plan(self, chunks, current_time):
        """(done, remaining) around now. A chunk in progress is truncated to now."""
        now = self._parse_time(current_time)
        done, remaining = [], []
        for c in chunks:
            start = self._parse_time(c.get("start_time"))
            if start is None:
                continue
            if start < now:
                c = dict(c)
                c["duration"] = min(c.get("duration") or 0,
                                    int((now - start).total_seconds() // 60))
                done.append(c)
            else:
                remaining.append(c)
        return done, remaining

    @staticmethod
    def teammates_line(agent, world):
        """Who actually works here. Plans used to include "collaborate with a
        designer" or "a backend engineer": people who don't exist, so the
        block could never become a visit or a conversation."""
        if not world:
            return ""
        others = [a for a in world["agents"].values() if a["id"] != agent["id"]]
        if not others:
            return ""
        team = ", ".join(f"{a['name']} ({a.get('area') or 'teammate'})" for a in others)
        return (f"\n        {agent['name']}'s only teammates are {team}; there is no one "
                f"else at the company, so name them in any block that involves "
                f"working with someone.\n")

    def generate_initial_plan(self, agent, world=None):
        """Day plan in broad strokes, continuing the agent's profile (paper 4.3).

        Like the paper's prompt, the first item of the day is written in for the
        model ("1) wake up ..."), so the plan always starts at DAY_START instead
        of wherever the model decides the interesting part of the day begins."""
        name = agent["name"]
        prompt = f"""
        {AGENT_PROFILES[agent["id"]]}
{self.recent_project_context(agent)}{self.teammates_line(agent, world)}
        Today is Thursday.

        Here is {name}'s plan today in broad strokes:
        1. {FIRST_PLAN_ITEM}

        Continue the plan as a numbered list, starting at item 2. Cover the whole
        workday until {name} clocks out at {DAY_END}, with no gaps.

        Return the plan in broad-strokes time blocks, each written as
        "<activity> from H:MM am to H:MM pm".

        Example:
        {PLAN_EXAMPLES[agent["id"]]}
        """
        return self.ensure_day_start(self._parse_numbered(llm.generate(prompt)))

    def recent_project_context(self, agent, n=8):
        """The paper's initial plan uses the agent's summary description,
        including "a summary of their recent experiences". Here that is what the
        agent remembers about the project and the board: on the first day the
        seeded goal and habits, after that what they saw on the board and agreed
        with teammates. Retrieval only, no extra LLM call."""
        results = self.memory_retriever.retrieve(
            f"the company website project, the project board, and {agent['name']}'s tasks",
            n=n, touch=False,
        )
        lines = [f"        - {r['memory'].description}" for r in results
                 if r["memory"].memory_type != "plan"]
        if not lines:
            return ""
        return f"""
        What {agent['name']} remembers about the project:
""" + "\n".join(lines) + "\n"

    def ensure_day_start(self, items):
        """Put the first item back if the model didn't repeat it, and only if
        nothing already covers the start of the day."""
        if any(re.search(r"\bclock(?:ing|ed)? in\b", i, re.IGNORECASE) for i in items):
            return items
        day_start = self._parse_time(DAY_START)
        starts = [self._parse_time(self.extract_start_time(i)) for i in items]
        starts = [t for t in starts if t is not None]
        if starts and min(starts) <= day_start:
            return items
        return [FIRST_PLAN_ITEM] + items

    def regenerate_plan(self, agent, current_time, reaction, done_chunks, old_remaining,
                        end_of_day="5:00 pm", guidance="", world=None):
        name = agent["name"]
        done_text = "\n".join(f"- {c['activity']}" for c in done_chunks) or "- (nothing yet)"
        old_text = "\n".join(f"- {c['activity']}" for c in old_remaining) or "- (nothing left)"

        prompt = f"""
        {AGENT_PROFILES[agent["id"]]}

        Today is Thursday and it is now {current_time}.
        
        So far today, {name} has done:
        {done_text}

        {name}'s previous plan for the rest of the day:
        {old_text}

        What just happened: {reaction}
        {guidance}{self.teammates_line(agent, world)}
        Revise {name}'s plan from {current_time} until {end_of_day}.
        Return a numbered list of broad-strokes blocks, each written as
        "<activity> from H:MM am to H:MM pm". The first block must start at {current_time},
        each block must start when the previous one ends (no gaps), and the last
        one must end at {end_of_day}. Describe what {name} will do, not the reason.
        
        Example:
        1. work on the website's backend API from {current_time} to 1:00 pm"""

        response = llm.generate(prompt, options={**CTX, "num_predict": 600, "temperature": 0.3})

        return self._parse_numbered(response)

    def task_guidance(self, agent, world):
        """Planning instructions after the agent has seen the board: what the
        project needs from them, and how to schedule their tasks around it."""
        if world is None or "project" not in world:
            return ""
        name = agent["name"]
        open_tasks = project_board.open_tasks_for(world, agent["id"])
        status = "\n".join(f"        - {line}"
                            for line in project_board.status_for_agent(world, agent["id"]))
        startable = [t for t in open_tasks
                     if t["status"] != "todo" or not project_board.task_waits(world, t)]
        pick = startable[0] if startable else None
        # The fallback names the agent's own area: "the website's navigation
        # bar" was shown to Adam (backend) too.
        example = (f"work on {pick['id']} '{pick['title']}' in the office from ..." if pick
                   else f"work on the website's {agent.get('area') or 'part'} in the office from ...")
        owners = "".join(f"\n        - {line}"
                         for line in project_board.ownership_lines(world, agent["id"]))
        owners = f"\n        Who owns what:{owners}\n" if owners else ""
        known = [t["id"] for t in project_board.get_project(world)["tasks"]]
        ids = (f"The only tasks on the board are {', '.join(known)}; never name any other "
               f"task id (new work has to be added at the board first)."
               if known else "There are no tasks on the board yet, so don't name any task ids.")
        return f"""
        The project board, as {name} last saw it:
{status}
{owners}
        Plan the rest of the day around the project:
        - Name the task in each work block, e.g. "{example}". {ids}
        - Work blocks are for {name}'s own tasks. A teammate's task only appears
          as reviewing it with them, or working on it together with them (name
          them) when they asked for help or agreed to it.
        - Don't plan work on a task that can't start yet because it waits on
          another; plan other work instead.
        - If {name}'s area has no tasks, or nothing is waiting for {name}, add a
          block to add or take on tasks at the project board.
        - If a teammate's task is waiting for review, add a block to review it
          with them and approve it at the project board.
        - Starting and finishing tasks at the board is scheduled automatically, so
          the only other board visit to add is one short one before clocking out.
        """

    _TASK_ID = re.compile(r"\bT-\d+\b", re.IGNORECASE)
    _TIME_PREFIX = re.compile(r"^\d{1,2}:\d{2}\s*[ap]m:\s*", re.IGNORECASE)
    _SPAN = re.compile(r"from\s*\d{1,2}:\d{2}\s*[ap]m\s*to\s*\d{1,2}:\d{2}\s*[ap]m",
                       re.IGNORECASE)

    def _fmt(self, dt):
        return dt.strftime("%I:%M %p").lstrip("0").lower()

    def _retimed(self, chunk, start, minutes):
        """Copy of a chunk moved to [start, start + minutes), text included, so
        later decomposition sees the same times the chunk has."""
        c = dict(chunk)
        end = start + timedelta(minutes=minutes)
        span = f"from {self._fmt(start)} to {self._fmt(end)}"
        c["activity"] = (self._SPAN.sub(span, c["activity"], count=1)
                         if self._SPAN.search(c["activity"]) else f"{c['activity']} {span}")
        c["start_time"], c["duration"] = self._fmt(start), minutes
        return c

    def _board_step(self, agent, start, minutes, text):
        return {
            "activity": f"{text} on the project board from {self._fmt(start)} to "
                        f"{self._fmt(start + timedelta(minutes=minutes))}",
            "location": "conference room",
            "start_time": self._fmt(start),
            "duration": minutes,
            "target_agent": None,
            "board_work": True,
        }

    def add_task_checkpoints(self, chunks, agent, open_tasks, minutes=10):
        """Starting and finishing work go through the board.

        For each of the agent's open tasks named in a work block: if the task is
        still todo and nothing earlier in the plan visits the board for it, the
        block starts with a short board step to move it to in progress; and the
        last block that names it ends with a short board step to update it
        (in progress, review or blocked, decided at the board from what the
        agent actually did). Blocks are split, never moved, so the day's times
        stay as planned."""
        own = {t["id"]: t for t in open_tasks}
        if not own:
            return chunks

        def ids(chunk):
            return [i.upper() for i in dict.fromkeys(self._TASK_ID.findall(chunk["activity"]))
                    if i.upper() in own]

        last = {}
        for i, c in enumerate(chunks):
            if not c.get("board_work"):
                for tid in ids(c):
                    last[tid] = i

        out, visited = [], set()
        for i, c in enumerate(chunks):
            named = ids(c)
            start = self._parse_time(c.get("start_time"))
            minutes_total = c.get("duration") or 0
            if c.get("board_work"):
                visited.update(named)
                out.append(c)
                continue
            if not named or start is None or minutes_total < 3 * minutes or c.get("pinned"):
                out.append(c)               # an agreed meeting keeps its time
                continue

            # One task at a time: only the first todo task named here is started.
            to_start = [t for t in named if own[t]["status"] == "todo" and t not in visited][:1]
            to_wrap = [t for t in named if last.get(t) == i]
            if to_start:
                out.append(self._board_step(agent, start, minutes,
                                            f"start {to_start[0]} (move it to in progress)"))
                visited.update(to_start)
                start += timedelta(minutes=minutes)
                minutes_total -= minutes
            if to_wrap:
                minutes_total -= minutes
            out.append(self._retimed(c, start, minutes_total))
            if to_wrap:
                out.append(self._board_step(
                    agent, start + timedelta(minutes=minutes_total), minutes,
                    f"report progress on {', '.join(to_wrap)}"))
        return out

    BOARD_VISIT_GAP_MINUTES = 90   # a plain "update the board" visit this soon after another is dropped

    _PURPOSEFUL = re.compile(r"approv|review|assign|claim|hand\w* (?:off|over)|add\w* (?:new |any )?tasks?",
                             re.IGNORECASE)
    # Blocks a dropped board visit's minutes never go to: stretching a break or
    # a clock-in ("clock in 7:00-7:20", then split into "greet the receptionist
    # in office_1") is not what the plan meant.
    _BREAK = re.compile(r"\b(?:break|lunch|coffee|snack|rest|clock(?:ing|ed)? (?:in|out))\b",
                        re.IGNORECASE)

    def thin_board_visits(self, chunks):
        """Fewer, more useful trips to the board.

        A replan in the run behind this had nine "update the project board"
        visits on top of the automatic start/progress steps, cutting work on a
        task into 10-minute pieces (each visit is also an LLM call). So:
        - a plain visit ("update the project board", no task and nothing to
          approve, assign or add) is dropped when another board visit ended less
          than BOARD_VISIT_GAP_MINUTES before it or the next block is board work,
          unless it is the last board visit of the day (before clocking out);
        - a visit for the same task(s) as the board visit just before it, within
          that gap ("report progress on T-3" twice in a row), is dropped;
        - zero-length blocks are dropped.
        A dropped visit's time goes to the work block right before it (not to a
        break); otherwise the gap filler covers it."""
        timed = [c for c in chunks
                 if self._parse_time(c.get("start_time")) is None or (c.get("duration") or 0) > 0]

        def ids(c):
            return frozenset(i.upper() for i in self._TASK_ID.findall(c["activity"]))

        def plain(c):
            return (c.get("board_work") and not ids(c)
                    and not self._PURPOSEFUL.search(c["activity"]))

        def kind(c):
            """start / report / ... : a repeat is the same kind of step for the same task."""
            text = self._TIME_PREFIX.sub("", c["activity"]).strip().lower()
            return (text.split() or [""])[0]

        last_board = max((i for i, c in enumerate(timed) if c.get("board_work")), default=None)
        gap = timedelta(minutes=self.BOARD_VISIT_GAP_MINUTES)
        out, prev_board = [], None          # (end, ids, kind) of the last kept board visit
        for i, c in enumerate(timed):
            s = self._parse_time(c.get("start_time"))
            if c.get("board_work") and s is not None:
                nxt = timed[i + 1] if i + 1 < len(timed) else None
                end = s + timedelta(minutes=c.get("duration") or 0)
                next_is_board = (nxt is not None and nxt.get("board_work")
                                 and self._parse_time(nxt.get("start_time")) == end)
                recent = prev_board is not None and s - prev_board[0] < gap
                drop = (plain(c) and i != last_board and (recent or next_is_board)) \
                    or (ids(c) and recent and ids(c) <= prev_board[1] and kind(c) == prev_board[2])
                if drop:
                    if next_is_board and not recent and plain(c):
                        # Two board visits back to back: keep the one with a
                        # purpose, at the earlier time (check the board right
                        # after clocking in, not 15 minutes later).
                        timed[i + 1] = self._retimed(nxt, s, nxt.get("duration") or 0)
                    else:
                        self._absorb(out, c, s)
                    log.debug(f"dropped a board visit that repeats a recent one: {c['activity']}")
                    continue
                prev_board = (s + timedelta(minutes=c.get("duration") or 0), ids(c), kind(c))
            out.append(c)
        return out

    def _absorb(self, out, dropped, start):
        """Give a dropped block's minutes to the work block right before it."""
        prev = out[-1] if out else None
        if prev is None or prev.get("board_work") or prev.get("pinned") \
                or self._BREAK.search(prev["activity"]):
            return
        ps = self._parse_time(prev.get("start_time"))
        if ps is not None and ps + timedelta(minutes=prev.get("duration") or 0) == start:
            out[-1] = self._retimed(prev, ps, (prev.get("duration") or 0)
                                    + (dropped.get("duration") or 0))

    _CLOCK_IN = re.compile(r"\bclock(?:ing|ed)? in\b", re.IGNORECASE)
    _HAS_END = re.compile(r"\bto\s*\d{1,2}:\d{2}\s*[ap]m\b", re.IGNORECASE)
    BOARD_BLOCK_MAX_MINUTES = 15   # a board visit is a few minutes, not an hour
    LONG_BLOCK_MINUTES = 60        # longer blocks only follow a teammate when it's real collaboration

    def create_chunks(self, daily_plan, agent, world=None):
        chunks = []
        for activity in daily_plan:
            activity = self.drop_unknown_task_ids(activity.strip().rstrip("."), world)
            activity = self.own_tasks_only(activity, agent, world)
            if activity is None:
                continue
            duration = self.extract_duration(activity)
            if self._CLOCK_IN.search(activity) and not self._HAS_END.search(activity):
                duration = CLOCK_IN_MINUTES
            chunks.append({
                "activity": activity,
                "location": self.resolve_location(activity, agent),
                "start_time": self.extract_start_time(activity),
                "duration": duration,
                "target_agent": self._block_target(activity, agent, world, duration),
                "board_work": project_board.mentions_board_explicitly(activity),
            })

        return chunks

    def _block_target(self, activity, agent, world, duration):
        """The teammate a plan block is about. A long block only counts when it
        is real collaboration: "work on T-1 in the office with Eve Smith" from
        8:10 to 11:40 had Adam following Eve for three and a half hours."""
        target = self.extract_target_agent(activity, agent, world)
        if target and (duration or 0) > self.LONG_BLOCK_MINUTES and not self._TOGETHER.search(activity):
            return None
        return target

    def cap_board_blocks(self, chunks):
        """Board visits last at most BOARD_BLOCK_MAX_MINUTES. Plans had "check the
        project board 8:15-9:00" and "review and update frontend tasks on the
        board 9:00-10:00"; Eve spent most of the morning at the board while her
        task sat "in progress". The rest of a long board block is left open, and
        the gap filler puts the agent back on their own work."""
        out = []
        for c in chunks:
            start = self._parse_time(c.get("start_time"))
            if (c.get("board_work") and start is not None
                    and (c.get("duration") or 0) > self.BOARD_BLOCK_MAX_MINUTES):
                c = self._retimed(c, start, self.BOARD_BLOCK_MAX_MINUTES)
            out.append(c)
        return out

    @staticmethod
    def drop_unknown_task_ids(activity, world):
        """Remove task ids that aren't on the board ("work on T-5 'Implement
        auth'" -> "work on 'Implement auth'"). A replan in the run behind this
        made up T-5, T-6 and T-7; an agent working on a task that doesn't exist
        can never update it, and later prompts would treat it as real."""
        if not world or "project" not in world:
            return activity
        unknown = project_board.unknown_task_ids(world, activity)
        if unknown:
            activity = project_board.strip_unknown_task_ids(world, activity)
            log.debug(f"plan named tasks that aren't on the board ({', '.join(unknown)}): {activity}")
        return activity

    def is_joint(self, activity, owners):
        """Is this block done together with these teammates (named, and a
        working-together verb), or a review of their task? Those are the only
        ways a plan touches a teammate's task."""
        if project_board.is_review_of_task(activity):
            return True
        named = all(re.search(rf"\b{re.escape(o.split()[0])}\b", activity, re.IGNORECASE)
                    for o in owners)
        return named and bool(self._TOGETHER.search(activity) or self._HELP.search(activity))

    _HELP = re.compile(r"\bhelp(?:s|ed|ing)?\b", re.IGNORECASE)

    def own_tasks_only(self, activity, agent, world):
        """An agent's plan works on the agent's own tasks. Returns the block as
        it should be, or None to leave it out.

        In the run behind this, Adam's replan put Eve's T-2 into his day ("start
        T-2 (move it to in progress)", "work on T-2 ...") and he spent the
        morning sketching her home page. So, for each task id in a block:
        - an id next to another task's title is fixed or dropped (fix_task_refs);
        - a teammate's task is fine in a review of it, or in a block that works
          on it together with them (they're named: "pair with Eve on T-2");
        - otherwise a board step for it is left out (only the owner moves it),
          and a work block about it becomes the agent's own work, at the same
          time. A block that also names the agent's own tasks just loses the
          teammate's id."""
        if not world or "project" not in world:
            return activity
        fixed = project_board.fix_task_refs(world, activity)
        if fixed != activity:
            log.debug(f"plan named a task by another task's title: {activity!r} -> {fixed!r}")
            activity = fixed
        theirs = project_board.teammate_task_ids(world, agent["id"], activity)
        if not theirs:
            return activity
        tasks = {t["id"]: t for t in project_board.get_project(world)["tasks"]}
        owners = {project_board.owner_name(world, tasks[tid]) for tid in theirs}
        if self.is_joint(activity, owners):
            return activity
        mine = [tid for tid in project_board.task_ids_in(activity) if tid not in theirs]
        if mine:
            kept = project_board.strip_task_ids(activity, theirs, with_titles=True)
        elif project_board.mentions_board_explicitly(activity):
            kept = None
        else:
            span = self._SPAN.search(activity)
            at = self._TIME_PREFIX.match(activity)
            kept = ((at.group(0) if at else "") + self.own_work(agent, world)
                    + (f" {span.group(0)}" if span else ""))
        log.debug(f"{agent['name']}'s plan had a teammate's task ({', '.join(theirs)}): "
                  f"{activity!r} -> {kept!r}")
        return kept

    @staticmethod
    def own_work(agent, world):
        """What the agent works on when nothing more specific is planned: the
        task they have in progress, else the next one they can start, else
        their own work."""
        mine = project_board.open_tasks_for(world, agent["id"]) if world and "project" in world else []
        working = [t for t in mine if t["status"] == "in_progress"]
        ready = [t for t in mine if t["status"] == "todo" and not project_board.task_waits(world, t)]
        if working:
            return f"keep working on {working[0]['id']} '{working[0]['title']}'"
        if ready:
            # "focus on Adam Beck's own work" became "review Adam Beck's to-do list".
            # (Not "start T-1": starting is a board step, and those words would
            # send the agent to the board.)
            return f"prepare for {ready[0]['id']} '{ready[0]['title']}' (read up and sketch the approach)"
        return f"focus on {agent['name']}'s own work"

    def keep_blocks(self, chunks, kept):
        """Put back plans with teammates that a replan dropped.

        Replanning rewrites the whole rest of the day, and plans with a teammate
        got lost on the way: Eve's 8:00 stand-up with Adam and her 10:15 work with
        him on the contact form both disappeared, and the run had a single
        conversation in 30 steps. Each kept block is put back at its time unless
        the new plan already has that teammate at an overlapping time; whatever
        overlaps it is cut around it. A meeting the two agreed on ("agreed")
        always gets its slot: the new plan's own version of time with that
        teammate is cut around it, so it isn't there twice."""
        out = list(chunks)
        for k in kept:
            ks = self._parse_time(k.get("start_time"))
            if ks is None:
                continue
            ke = ks + timedelta(minutes=k.get("duration") or 0)
            if any(c.get("target_agent") == k["target_agent"] and self._overlaps(c, ks, ke)
                   and (c.get("agreed") or not k.get("agreed")) for c in out):
                continue
            out = self.insert_block(out, k)
            log.debug(f"kept a plan with a teammate the replan dropped: {k['activity']}")
        return out

    def _overlaps(self, c, ks, ke):
        cs = self._parse_time(c.get("start_time"))
        if cs is None:
            return False
        return cs < ke and ks < cs + timedelta(minutes=c.get("duration") or 0)

    def insert_block(self, chunks, block):
        """`block` at its own time; whatever overlaps it is cut around it
        (pieces shorter than 5 minutes are dropped)."""
        ks = self._parse_time(block.get("start_time"))
        if ks is None:
            return list(chunks)
        ke = ks + timedelta(minutes=block.get("duration") or 0)
        rebuilt = []
        for c in chunks:
            if not self._overlaps(c, ks, ke):
                rebuilt.append(c)
                continue
            cs = self._parse_time(c["start_time"])
            ce = cs + timedelta(minutes=c.get("duration") or 0)
            before = int((ks - cs).total_seconds() // 60)
            after = int((ce - ke).total_seconds() // 60)
            if before >= 5:
                rebuilt.append(self._retimed(c, cs, before))
            if after >= 5:
                rebuilt.append(self._retimed(c, ke, after))
        rebuilt.append(dict(block))
        return sorted(rebuilt, key=lambda c: self._parse_time(c.get("start_time")) or datetime.min)

    @staticmethod
    def work_focus(agent, world):
        """The task the agent's work time is for right now, as one sentence, or "".
        Breaking "review and update any frontend tasks" into 10-minute steps
        without it gave "prioritize task list", "update task status in project
        management tool" and "address high-priority bugs" (there were none)."""
        if not world or "project" not in world:
            return ""
        name = agent["name"]
        mine = project_board.open_tasks_for(world, agent["id"])
        busy = [t for t in mine if t["status"] == "in_progress"]
        if busy:
            t = busy[0]
            return f"{name} is working on {t['id']} '{t['title']}' (in progress)."
        ready = [t for t in mine if t["status"] == "todo" and not project_board.task_waits(world, t)]
        if ready:
            t = ready[0]
            return (f"{name}'s next task is {t['id']} '{t['title']}' (not started yet; "
                    f"starting it happens at the project board).")
        return ""

    def resolve_location(self, activity, agent, world=None, default=None):
        return self.resolve_location_keywords(activity, agent, default=default)

    def obvious_location(self, activity, agent, world):
        """The room for an activity when no judgment is needed, else None.

        In the run behind this every choose_location call (about 14s each)
        concluded "solo work, stay in your own office". So: if the activity
        doesn't involve a teammate and the keyword map says it is solo work
        (the agent's own office) or names the room the agent is already in,
        that is the answer, without an LLM call."""
        if self.extract_target_agent(activity, agent, world):
            return None
        valid = world["locations"]
        suggested = self.resolve_location_keywords(activity, agent).lower().replace(" ", "_")
        if suggested not in valid:
            return None
        if suggested == agent.get("office") or suggested == agent["current_location"]:
            return suggested
        return None

    def choose_location(self, activity, agent, world, observations=None, recent_dialogue=None):
        valid = list(world["locations"].keys())
        owners = {a["office"]: a["name"] for a in world["agents"].values() if a.get("office")}

        def describe(loc):
            if loc in owners:
                return f"- {loc}: {owners[loc]}'s private office, for solo focused work"
            purpose = (world["locations"][loc].get("description")
                       or ROOM_PURPOSES.get(loc, "general use"))
            return f"- {loc}: {purpose}"

        location_lines = "\n".join(describe(l) for l in valid)

        # The keyword map's suggestion, normalized to a real room key
        suggested = self.resolve_location_keywords(activity, agent).lower().replace(" ", "_")
        if suggested not in valid:
            suggested = agent["office"]

        obs_text = "\n".join(f"- {o}" for o in (observations or []))

        prompt = f"""
        {agent['name']} is about to: {activity}
        {agent['name']} is currently in {agent['current_location']}.
        The plan suggests: {suggested}

        Rooms:
        {location_lines}

        Recent observations:
        {obs_text}

        Which room should {agent['name']} go to in order to do this activity?

        Rules:
        - Choose the room whose purpose best matches the activity, even if that
          means leaving the current room.
        - Activities about meetings, group work, or collaborating belong in the
          conference room. Solo focused work belongs in {agent['name']}'s own office.
        - Leave the current room when the activity belongs somewhere else; staying
          is right when the current room already fits the activity.
        - Follow the plan's suggestion unless the observations clearly show a reason not to.
        """

        schema = {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "location": {"type": "string", "enum": valid},
            },
            "required": ["reason", "location"],
        }

        try:
            response = llm.generate(
                prompt, format=schema,
                options={**CTX, "num_predict": 300, "temperature": 0},
            )
            result = json.loads(response)
            log.debug(f"choose_location: {result['location']} ({result['reason']})")
            chosen = result["location"]
            if chosen in owners and chosen != agent.get("office") \
                    and not self.extract_target_agent(activity, agent, world):
                # A teammate's private office is only right when the activity is
                # about that teammate. Otherwise the agent "works" at Eve's desk
                # and then spends a reaction (and the cooldown) walking back.
                log.debug(f"choose_location: {chosen} is {owners[chosen]}'s office, "
                          f"using {suggested} instead")
                return suggested
            if chosen in valid:
                return chosen
        except Exception as e:
            log.warn(f"choose_location LLM failed ({e}), using keyword fallback")

        return suggested

    def resolve_location_keywords(self, activity, agent, default=None):
        """
        Map an activity description to a room. Order matters: explicit room
        names win, then specific activities, then the agent's default.
        `default` lets callers (e.g. reactions) fall back to the agent's current
        room instead of their office.
        """
        text = activity.lower()

        def has(*patterns):
            return any(re.search(p, text) for p in patterns)

        # 1. Explicit room names in the text
        if has(r"\breception\b", r"\bfront desk\b"):
            return "reception"
        if has(r"\bbreak room\b", r"\bkitchen\b"):
            return "break room"
        if has(r"\bconference room\b", r"\bmeeting room\b",
            r"\b(?:project|task) board\b", r"\bthe board\b"):
            return "conference room"

        # 2. Arriving / leaving
        if has(r"\bclock(?:ing)? (?:in|out)\b", r"\bsign(?:ing)?[ -](?:in|out)\b",
            r"\bbadge (?:in|out)\b", r"\bhead(?:ing)? home\b",
            r"\bleave (?:for the day|work)\b"):
            return "reception"

        # 3. Meals, rest and personal needs
        if has(r"\blunch\b", r"\bcoffee\b", r"\bsnack\b", r"\bhydrate\b",
            r"\bdrink(?:ing)? water\b", r"\bstretch(?:ing)?\b",
            r"\b(?:take|taking|short|quick|brief|coffee) break\b",
            r"\bbreak (?:for|and)\b", r"\brelax(?:ing)?\b", r"\beat(?:ing)?\b"):
            return "break room"

        # 4. Group work
        if has(r"\bmeetings?\b", r"\bstand-?up\b", r"\bsync\b", r"\bretro(?:spective)?\b",
            r"\bbrainstorm\w*\b", r"\bwhiteboard\w*\b", r"\bdemo\b",
            r"\bpresent(?:ing|ation)?\b", r"\bcollaborat\w+\b",
            r"\bpair(?:ing)? (?:up|programming)\b", r"\bdiscuss\w* (?:with )?(?:the )?team\b",
            r"\bteam (?:discussion|check-?in|huddle)\b"):
            return "conference room"

        # 5. Everything else is solo work
        return default or agent["office"]

    def extract_start_time(self, activity):
        activity = activity.lower().strip()
        
        # Matches:
        # "6:00 pm: start brainstorming"
        # "6:00 pm start brainstorming"
        match = re.match(
        r"^(\d{1,2}:\d{2}\s*(?:am|pm))(?::|\s|$)",
        activity
        )
        if match:
            return match.group(1)
        
        # Matches:
        # "work on project from 6:00 pm to 8:00 pm"
        match = re.search(
        r"from\s*(\d{1,2}:\d{2}\s*(?:am|pm))",
        activity
        )
        if match:
            return match.group(1)
        
        # Matches:
        # "meeting at 6:00 pm"
        match = re.search(
        r"at\s+(\d{1,2}:\d{2}\s*(?:am|pm))",
        activity
        )
        if match:
            return match.group(1)
        
        return None

    def extract_duration(self, activity):
        activity = activity.lower()

        match = re.search(
            r"from\s*(\d{1,2}):(\d{2})\s*(am|pm)\s*to\s*(\d{1,2}):(\d{2})\s*(am|pm)",
            activity
        )

        if not match:
            if "clock in" in activity:
                return 60

            if "clock out" in activity:
                return 60

            return 0

        start_hour = int(match.group(1))
        start_minute = int(match.group(2))
        start_period = match.group(3)

        end_hour = int(match.group(4))
        end_minute = int(match.group(5))
        end_period = match.group(6)

        if start_period == "pm" and start_hour != 12:
            start_hour += 12

        if start_period == "am" and start_hour == 12:
            start_hour = 0

        if end_period == "pm" and end_hour != 12:
            end_hour += 12

        if end_period == "am" and end_hour == 12:
            end_hour = 0

        start_total = start_hour * 60 + start_minute
        end_total = end_hour * 60 + end_minute

        return end_total - start_total

    def create_plan(self, chunks):
        description = "\n".join(
        f"{chunk['activity']} in {chunk['location']} for {chunk['duration']} minutes"
        for chunk in chunks
        )
        
        plan = self.memory_stream.add_plan(description)

        log.debug(f"plan saved:\n{plan.description}")

        return plan

    def _chunk_from_activity(self, activity, agent, parent, duration=None, world=None):
        """Build a plan chunk from one parsed line, inheriting the parent's target.
        None if the line is a board step for a teammate's task (own_tasks_only)."""
        activity = self.own_tasks_only(activity.strip().rstrip("."), agent, world)
        if activity is None:
            return None
        chunk = {
            "activity": activity,
            "location": self.resolve_location(activity, agent),
            "start_time": self.extract_start_time(activity),
            "target_agent": self._inherited_target(activity, parent),
            # A board visit stays a board visit when it is broken down, even if
            # a step reads "check all frontend tasks for updates".
            "board_work": bool(parent.get("board_work"))
                          or project_board.mentions_board_explicitly(activity),
        }
        if duration is not None:
            chunk["duration"] = duration
        return chunk

    # Verbs of working with someone. Nouns don't count: "summarize key points from
    # the discussion" is solo work, and so is "synchronize the data".
    _TOGETHER = re.compile(
        r"\b(?:discuss(?:es|ed|ing)?|pair(?:s|ed|ing)?\s+(?:with|up|on)|pair programming"
        r"|sync(?:s|ed|ing)?\s+(?:with|up)|meet(?:s|ing)?\s+with|talk(?:s|ed|ing)?\s+(?:with|to)"
        r"|ask(?:s|ed|ing)?|together|collaborat(?:e|es|ed|ing)|walk(?:s|ed|ing)?\s+\w+\s+through"
        r"|go(?:es|ing)? over\b.*\bwith|align(?:s|ed|ing)?\s+with|review\w*\b[^.]*?\b(?:with|together))\b",
        re.IGNORECASE)

    def _inherited_target(self, activity, parent):
        """A finer step keeps its block's teammate only if it is still about
        working with them. "Collaborate with Eve" broke down into steps like
        "set up the development environment", and every one of them chased Eve
        from room to room (Plan7: Adam followed her for an hour, unable to talk
        because they just had)."""
        target = parent.get("target_agent")
        if not target:
            return None
        # Agent ids are first names ("eve"), as extract_target_agent assumes too.
        named = re.search(rf"\b{re.escape(target)}\b", activity, re.IGNORECASE)
        return target if named or self._TOGETHER.search(activity) else None

    def _focus_text(self, chunk, focus):
        """Task context for breaking a work block down. Not for breaks, board
        visits or clocking in/out, which are about something else."""
        if not focus or chunk.get("board_work") or self._BREAK.search(chunk.get("activity") or ""):
            return ""
        return f"""
            Context: {focus}
            If the activity is general work, make the actions concrete steps of
            that task. Don't add actions about to-do lists, task statuses or
            project management tools: the project board is updated on separate visits.
"""

    def decompose_chunks(self, chunks, agent, focus="", world=None):
        """Broad blocks -> one chunk per hour (paper: hour-long chunks).
        `focus` (work_focus) says which task the agent's work time is for."""
        decomposed_chunks = []

        for chunk in chunks:
            if chunk["duration"] < 60:
                decomposed_chunks.append(chunk)
                continue

            # Round UP: a 10:05-12:00 block (115 min) needs two hour chunks, not one,
            # or 11:05-12:00 has nothing in it.
            hours = math.ceil(chunk["duration"] / 60)

            prompt = f"""
            The following activity is one entry from a daily plan.

            Create EXACTLY {hours} hourly actions.

            The first action starts at the activity's start time.
            Each following action occurs exactly one hour later.

            DO NOT create additional hours.
            DO NOT include the end time as another action.
            DO NOT create more than {hours} actions.

            Activity:
            {chunk["activity"]}

            Duration:
            {chunk["duration"]} minutes
{self._focus_text(chunk, focus)}
            Example:

            Activity:
            Collaborate with coworkers from 2:00 pm to 4:00 pm

            Duration:
            120 minutes

            Output:

            1. 2:00 pm: Discuss technical challenges
            2. 3:00 pm: Evaluate solutions and next steps

            Return only a numbered list.
            """

            hour_chunks = [c for c in (
                self._chunk_from_activity(activity, agent, chunk, duration=60, world=world)
                for activity in self._parse_numbered(llm.generate(prompt))[:hours]) if c]
            hour_chunks = self._drop_board_detours(hour_chunks, chunk)
            self._fit_to_block(hour_chunks, chunk)
            decomposed_chunks.extend(hour_chunks)

        return decomposed_chunks

    def _fit_to_block(self, hour_chunks, block):
        """Each hour chunk lasts until the next one starts or the broad block ends,
        and never more than an hour (so a 115-minute block ends with a 55-minute
        chunk). If the model returned too few hours, the rest is a gap that
        main.get_chunk_lazy fills from the broad block."""
        block_start = self._parse_time(block.get("start_time"))
        if block_start is None:
            return
        block_end = block_start + timedelta(minutes=block["duration"])
        for i, hc in enumerate(hour_chunks):
            start = self._parse_time(hc.get("start_time"))
            if start is None:
                continue
            end = block_end
            if i + 1 < len(hour_chunks):
                following = self._parse_time(hour_chunks[i + 1].get("start_time"))
                if following is not None and following > start:
                    end = min(following, block_end)
            minutes = min(60, int((end - start).total_seconds() // 60))
            hc["duration"] = minutes if minutes > 0 else 60

    def covers(self, chunk, current_time):
        """Is current_time inside this chunk (not just after its start)?"""
        now, start = self._parse_time(current_time), self._parse_time(chunk.get("start_time"))
        if now is None or start is None:
            return False
        return start <= now < start + timedelta(minutes=chunk.get("duration") or 0)

    _PM_TOOL = re.compile(r"\b(?:project management|task tracker|tracking tool|jira|trello|kanban)\b",
                          re.IGNORECASE)

    def _drop_board_detours(self, actions, parent):
        """Inside a work block, a step like "update task status in project
        management tool" sent Eve to the board for an unplanned visit (an LLM
        call, and a note about bugs that didn't exist). Board visits are
        planned on their own, so such steps are dropped; the step before
        absorbs the time, and if the first one goes, the next one starts in its place."""
        if parent.get("board_work") or not actions:
            return actions
        kept = [a for a in actions
                if not project_board.mentions_board_explicitly(a["activity"])
                and not self._PM_TOOL.search(a["activity"])]
        if len(kept) == len(actions):
            return actions
        log.debug(f"dropped board steps from a work block: "
                  f"{[a['activity'] for a in actions if a not in kept]}")
        if kept and kept[0] is not actions[0] and actions[0].get("start_time"):
            first, start = dict(kept[0]), actions[0]["start_time"]
            first["activity"] = self._TIME_PREFIX.sub(f"{start}: ", first["activity"], count=1)
            first["start_time"] = start
            kept[0] = first
        return kept

    def decompose_hour_chunks(self, hour_chunks, agent, focus="", world=None):
        """Hour chunks -> 5-15 minute actions (paper: finer-grained actions).
        `focus` (work_focus) says which task the agent's work time is for."""
        minute_chunks = []

        for chunk in hour_chunks:
            prompt = f"""
            The following activity is one entry from an hourly plan.

            Break it into actions lasting roughly 5 to 15 minutes.

            Rules:
            - Create actions lasting roughly 5 to 15 minutes.
            - Action times should be sequential.
            - Actions should contribute to the original activity.
            - Do not create actions that are unrelated to the activity.
            - Return only a numbered list.

            Activity:
            {chunk["activity"]}
{self._focus_text(chunk, focus)}
            Example Input:

            4:00 pm: Review and polish a music composition

            Example Output:

            1. 4:00 pm: Review the composition.
            2. 4:10 pm: Identify areas needing improvement.
            3. 4:20 pm: Refine weak sections.
            4. 4:35 pm: Listen through the revision.
            5. 4:50 pm: Organize notes and clean the workspace.
            """

            actions = [c for c in (
                self._chunk_from_activity(activity, agent, chunk, world=world)
                for activity in self._parse_numbered(llm.generate(prompt))) if c]
            actions = self._drop_board_detours(actions, chunk)
            self._fill_durations(actions, hour_start=chunk.get("start_time"))
            minute_chunks.extend(actions)

        return minute_chunks

    def _fill_durations(self, actions, hour_start, default=10):
        """Each action lasts until the next one starts; the last one fills the
        rest of the hour. Anything unparseable or non-positive gets `default`."""
        for i, action in enumerate(actions):
            start = self._parse_time(action["start_time"])
            minutes = default

            if start is not None:
                if i < len(actions) - 1:
                    following = self._parse_time(actions[i + 1]["start_time"])
                    if following is not None:
                        minutes = int((following - start).total_seconds() / 60)
                else:
                    anchor = self._parse_time(hour_start)
                    if anchor is not None:
                        elapsed = int((start - anchor).total_seconds() / 60)
                        minutes = max(5, 60 - elapsed)

            action["duration"] = minutes if minutes > 0 else default

    def get_current_status(self, agent, current_time):
        return (
            f"It is {current_time}. "
            f"{agent['name']} is currently at {agent['current_location']} and is {agent['current_action']}."
        )

    def _parse_time(self, value):
        try:
            return datetime.strptime(value.upper().strip(), "%I:%M %p")
        except (ValueError, AttributeError):
            return None

    def get_current_minute_chunk(self, current_time, minute_chunks):
        current_dt = self._parse_time(current_time)
        if current_dt is None:
            return None

        latest_started = None
        latest_start_dt = None

        for chunk in minute_chunks:
            start_dt = self._parse_time(chunk.get("start_time"))
            if start_dt is None:
                continue

            duration = chunk.get("duration") or 0
            end_dt = start_dt + timedelta(minutes=duration)

            # Exact match: current time falls inside this chunk
            if start_dt <= current_dt < end_dt:
                return chunk

            # Track the most recent chunk that has already started
            if start_dt <= current_dt and (
                latest_start_dt is None or start_dt >= latest_start_dt
            ):
                latest_started = chunk
                latest_start_dt = start_dt

        return latest_started

    def continue_plan_or_react(self, agent, status, current_chunk, observations, world):
        # `observations` should now be only the NEW external observations
        # (main.py passes `candidates`), matching the paper's "react to an observation".

        agent_summary = self.generate_agent_summary(agent)   # cached after the first call
        observations_text = "\n".join(
            f"Observation: {observation}" for observation in observations
        )

        room = agent["current_location"]

        # Build every entity in the room first...
        entities = [(room, world["locations"][room]["state"])]

        for object_name, object_data in world["locations"][room]["objects"].items():
            entities.append((object_name, object_data["state"]))

        for other_agent in world["agents"].values():
            if other_agent["id"] != agent["id"] and other_agent["current_location"] == room:
                entities.append((other_agent["name"], other_agent["current_action"]))

        # ...THEN keep only the ones mentioned in what we're reacting to.
        # (object keys use underscores, observations use spaces: "project board")
        entities = [(n, s) for n, s in entities
                    if any(n in o or n.replace("_", " ") in o for o in observations)]

        context_memories = []
        seen = set()

        for entity_name, entity_status in entities:
            queries = [
                f"What is {agent['name']}'s relationship with {entity_name}",
                f"{entity_name} is {entity_status}"
            ]

            for query in queries:
                for result in self.memory_retriever.retrieve(query, n=5):
                    description = result["memory"].description
                    if description not in seen:
                        seen.add(description)
                        context_memories.append(result)

        context_memories.sort(key=lambda r: r["retrieval_score"], reverse=True)
        context_memories = context_memories[:10]

        context = "\n".join(
            f"- {result['memory'].description}" for result in context_memories
        )

        prompt = f"""
        {agent_summary}

        {agent["name"]}'s status: {status}

        {observations_text}

        Current planned action: {current_chunk["activity"]}
        Current planned location: {current_chunk["location"]}

        Summary of relevant context from {agent["name"]}'s memory:
        {context}
{self._board_rules(agent, world, observations_text)}
        Should {agent["name"]} react to the observation, and if so, what would be an appropriate reaction?
        A reaction is something {agent["name"]} does (go somewhere, talk to someone, work on
        something), written as an action, not a line of dialogue. People and changes
        to the work are worth reacting to; the room's ordinary objects (a time clock,
        a desk, a coffee machine) being used as usual are not.

        Answer in exactly this format:
        DECISION: 1 (continue the current plan) or 2 (react)
        REACTION: <one short sentence, or "none">
        """

        return llm.generate(
            prompt, options={**CTX, "num_predict": 80, "temperature": 0}
        )

    def _board_rules(self, agent, world, mentioned=""):
        """How the board works, and where the agent's own tasks really stand, so
        a reaction can't be "approve my own T-1 to unblock Eve" (it was)."""
        if not world or "project" not in world:
            return ""
        mine = project_board.open_tasks_for(world, agent["id"])
        own = [t["id"] for t in mine]
        facts = project_board.task_facts(world, project_board.task_ids_in(mentioned) + own)
        facts = "".join(f"\n        - {f}" for f in facts)
        busy = [t["id"] for t in mine if t["status"] == "in_progress"]
        wip = (f"\n        {agent['name']} works on one task at a time: {busy[0]} is in "
               f"progress, so {agent['name']}'s other tasks wait until it goes to review."
               if busy else "")
        owners = "".join(f"\n        - {line}"
                         for line in project_board.ownership_lines(world, agent["id"]))
        return f"""
        {project_board.task_id_line(world)}
        How the project board works: tasks only change at the board in the
        conference room. The assignee moves their task todo -> in progress ->
        review; a teammate (never the assignee) approves it while it is in
        review; then the assignee moves it to done. A task that waits on another
        can't start until that one is done. Questions about a teammate's task
        go to that teammate.{wip}{owners}{facts}
"""

    def invalidate_summary(self):
        """Forget cached agent summaries (call after reflection adds new insights)."""
        self._summary_cache.clear()

    def generate_agent_summary(self, agent):
        """Paper sec. 4.1 agent summary: identity, occupation, recent progress.
        Cached per agent until invalidate_summary() is called."""
        cached = self._summary_cache.get(agent["id"])
        if cached is not None:
            return cached

        # touch=False: building a summary isn't the agent *using* those memories,
        # so it shouldn't refresh their recency.
        core_memories = self.memory_retriever.retrieve(
            f"{agent['name']}'s core characteristics", touch=False
        )

        occupation_memories = self.memory_retriever.retrieve(
            f"{agent['name']}'s current daily occupation", touch=False
        )

        progress_memories = self.memory_retriever.retrieve(
            f"{agent['name']}'s feeling about recent progress in life", touch=False
        )

        all_memories = (
            core_memories +
            occupation_memories +
            progress_memories
        )

        seen = set()
        unique_memories = []

        for result in all_memories:
            description = result["memory"].description

            if description not in seen:
                seen.add(description)
                unique_memories.append(description)

        memory_text = "\n".join(
            f"- {description}"
            for description in unique_memories
        )

        prompt = f"""
    How would one describe {agent['name']}'s core characteristics
    given the following statements?

    {memory_text}

    Return the summary as a paragraph with no formatting.
    """

        summary = llm.generate(prompt).strip()
        self._summary_cache[agent["id"]] = summary
        return summary

    def extract_target_agent(self, activity, agent, world):
        """Return the id of another agent named in the activity, else None."""
        if not world:
            return None
        text = activity.lower()
        for other in world["agents"].values():
            if other["id"] == agent["id"]:
                continue
            first = other["name"].split()[0].lower()
            if re.search(rf"\b{first}\b", text):   # \b so "evening" doesn't match "eve"
                return other["id"]
        return None

    def make_reaction_chunk(self, agent, current_time, reaction, world, minutes=10):
        reaction = reaction.strip().rstrip(".")
        return {
            "activity": f"{current_time}: {reaction}",
            "location": self.resolve_location(reaction, agent, default=agent["current_location"]),
            "start_time": current_time,
            "duration": minutes,
            "target_agent": self.extract_target_agent(reaction, agent, world),
            "is_reaction": True,
            "board_work": project_board.mentions_board_explicitly(reaction),
        }

    @staticmethod
    def _ownership_for_outcomes(agent, world):
        if not world or "project" not in world:
            return ""
        lines = [line for line in project_board.ownership_lines(world, agent["id"])
                 if line.endswith(".") and "tasks:" in line or line.startswith("Unassigned")]
        if not lines:
            return ""
        return ("Whose tasks are whose: " + " ".join(lines).replace(f"{agent['name']}'s own", agent["name"] + "'s")
                + " A person's commitment about someone else's task can only be to review it,"
                  " or to work on it together with its owner when the owner agreed to that;"
                  " moving it on the board or doing it alone is the owner's job, not theirs.")

    def own_commitment(self, text, agent, world):
        """What of a commitment is the agent's to keep. Adam "committed" to
        "update the board once there is progress on T-2, T-3, and T-4" and to
        "ensure T-2 is stable and meets the review criteria" (all Eve's): the
        first sent him to the board to nag her in a note, the second put T-2 in
        his plan. About a teammate's task, a commitment can be reviewing it or
        working on it together with them (they're named); anything else is
        dropped, or just loses that task when it is also about the agent's own."""
        theirs = project_board.teammate_task_ids(world, agent["id"], text)
        if not theirs:
            return text
        tasks = {t["id"]: t for t in project_board.get_project(world)["tasks"]}
        owners = {project_board.owner_name(world, tasks[tid]) for tid in theirs}
        if self.is_joint(text, owners):
            return text
        mine = [tid for tid in project_board.task_ids_in(text) if tid not in theirs]
        kept = self._own_clauses(text, agent, world, owners) if mine else ""
        log.debug(f"{agent['name']}'s commitment was about a teammate's task "
                  f"({', '.join(theirs)}): {text!r} -> {kept!r}")
        return kept

    _CLAUSE = re.compile(r"\s*;\s*|,?\s+and\s+(?:then\s+)?|,\s+(?=(?:then\s+)?(?:I\b|I'll|I'm|we\b))",
                         re.IGNORECASE)

    def _own_clauses(self, text, agent, world, owners):
        """The parts of a commitment that are the agent's: a part only about a
        teammate's task is dropped whole. Cutting just the task id out left
        "I will ensure is properly set up" and "keep T-6 in todo until is in
        review". A part about the agent's own task that waits on a teammate's
        ("start T-6 as soon as T-1 moves to review") is kept as it is."""
        kept = []
        for part in self._CLAUSE.split(text):
            ids = project_board.task_ids_in(part)
            theirs = project_board.teammate_task_ids(world, agent["id"], part)
            if not theirs or len(ids) > len(theirs) or self.is_joint(part, owners):
                if part.strip():
                    kept.append(part.strip())
        return " and ".join(kept)

    def _meeting_start(self, text, current_time):
        """When an agreed meeting starts: None for right afterwards ("now", no
        time, or a time that has already come), else the time ("10:30 am").
        A time at or after the end of the workday means no later meeting."""
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\b", text or "", re.IGNORECASE)
        when = (self._parse_time(f"{int(m.group(1))}:{m.group(2) or '00'} {m.group(3)}m")
                if m and 1 <= int(m.group(1)) <= 12 else None)
        now = self._parse_time(current_time) if current_time else None
        if when is None or now is None or when <= now + timedelta(minutes=5):
            return None
        if when >= self._parse_time(DAY_END):
            return None
        return self._fmt(when)

    def extract_outcomes(self, transcript, agent_a, agent_b, world, current_time=None):
        """One call: shared meeting (location-pinned) + each agent's own commitment."""
        valid = list(world["locations"].keys())
        owners = {a["office"]: a["name"] for a in world["agents"].values() if a.get("office")}
        loc_lines = "\n".join(
            f"- {l}" + (f" ({owners[l]}'s office)" if l in owners else "") for l in valid
        )
        names = [agent_a["name"], agent_b["name"]]

        prompt = f"""
        Conversation between {names[0]} and {names[1]}:
        {transcript}

        Locations:
        {loc_lines}

        1. Did they both agree to meet or work together at a specific place, either
        right afterwards or at a set time later today? ("your desk" means the office
        of the person being addressed.) Set meeting_time to "now" for right
        afterwards, or to the agreed time ("10:30 am"). Only count a meeting both
        of them agreed to, with a place; if not, set meeting_agreed to false.
        2. For each person, what did they commit to doing (a task, preparation, or
        a meeting at a certain time)? Write one short sentence from their point
        of view, including any time or place. Small talk is not a commitment, and
        neither is something they say they already did ("I've approved the task"):
        only what they will do next. Omit a person who committed to nothing.
        {self._ownership_for_outcomes(agent_a, world)}
        """
        schema = {
            "type": "object",
            "properties": {
                "meeting_agreed": {"type": "boolean"},
                "location": {"type": "string", "enum": valid},
                "activity": {"type": "string"},
                "meeting_time": {"type": "string"},
                "minutes": {"type": "integer"},
                "commitments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "person": {"type": "string", "enum": names},
                            "commitment": {"type": "string"},
                        },
                        "required": ["person", "commitment"],
                    },
                },
            },
            "required": ["meeting_agreed", "location", "activity", "minutes", "commitments"],
        }
        out = {"meeting": None, "commitments": {}}
        try:
            raw = llm.generate(prompt, format=schema,
                               options={**CTX, "num_predict": 300, "temperature": 0})
            data = json.loads(raw)
            if data["meeting_agreed"] and data["location"] in valid:
                out["meeting"] = {
                    "location": data["location"],
                    "activity": data["activity"],
                    "minutes": max(10, min(data["minutes"] or 30, 60)),
                    # "now" with "check-in at 8:00 am" as the activity (said at
                    # 7:45) was put at 7:50: the time named in the activity wins.
                    "start": (self._meeting_start(data.get("activity"), current_time)
                              or self._meeting_start(data.get("meeting_time"), current_time)),
                }
            for c in data["commitments"]:
                text = future_part(c["commitment"].strip().rstrip("."))
                if world and "project" in world:
                    text = project_board.strip_unknown_task_ids(world, text)
                    person = agent_a if c["person"] == agent_a["name"] else agent_b
                    text = self.own_commitment(text, person, world)
                if text:
                    out["commitments"][c["person"]] = text
        except Exception as e:
            log.warn(f"extract_outcomes failed ({e})")
        return out