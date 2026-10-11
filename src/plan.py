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

        The team keeps track of its work on the project board in the conference room. Anyone on the team can add tasks to the board, take on tasks, and update their status. Adam checks the project board at the start of each workday and adds any backend work the website still needs. When Adam starts, finishes, or gets stuck on a task, he walks to the project board to update it, and he updates his tasks there before lunch and before clocking out. A task is only done once a teammate has reviewed and approved it, so when a teammate's task is in review, Adam looks it over with them and approves it at the board if it is ready. When Adam hands a task to a teammate, he leaves a note on it saying what it needs.

        Adam Beck works closely with other engineers and enjoys collaborating on technical challenges.

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

        The team keeps track of its work on the project board in the conference room. Anyone on the team can add tasks to the board, take on tasks, and update their status. Eve checks the project board at the start of each workday and adds any frontend work the website still needs. When Eve starts, finishes, or gets stuck on a task, she walks to the project board to update it, and she updates her tasks there before lunch and before clocking out. A task is only done once a teammate has reviewed and approved it, so when a teammate's task is in review, Eve looks it over with them and approves it at the board if it is ready. When Eve hands a task to a teammate, she leaves a note on it saying what it needs.

        Eve Smith works with designers and other engineers, and prefers collaborating one-on-one or in writing on technical challenges.

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
FIRST_PLAN_ITEM = f"clock in and start work at {DAY_START}"

# The example line shown to the model in the initial-plan prompt.
PLAN_EXAMPLES = {
    "adam": "2. check the project board in the conference room from 8:00 am to 9:00 am",
    "eve": "2. check the project board in the conference room from 8:00 am to 9:00 am",
}


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

    def generate_initial_plan(self, agent):
        """Day plan in broad strokes, continuing the agent's profile (paper 4.3).

        Like the paper's prompt, the first item of the day is written in for the
        model ("1) wake up ..."), so the plan always starts at DAY_START instead
        of wherever the model decides the interesting part of the day begins."""
        name = agent["name"]
        prompt = f"""
        {AGENT_PROFILES[agent["id"]]}
{self.recent_project_context(agent)}
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
                        end_of_day="5:00 pm", guidance=""):
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
        {guidance}
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
        example = (f"work on {open_tasks[0]['id']} '{open_tasks[0]['title']}' in the office from ..."
                   if open_tasks else "work on T-3 'Set up the API' in the office from ...")
        return f"""
        The project board, as {name} last saw it:
{status}

        Plan the rest of the day around the project:
        - Name the task in each work block, e.g. "{example}".
        - If {name}'s area has no tasks, or nothing is waiting for {name}, add a
          block to add or take on tasks at the project board.
        - When {name} starts, finishes, or gets stuck on a task, add a short block
          to update it on the project board in the conference room.
        - If a teammate's task is waiting for review, add a block to review it
          with them and approve it at the project board.
        - Add a short block to update the project board before lunch and before
          clocking out.
        """

    _TASK_ID = re.compile(r"\bT-\d+\b", re.IGNORECASE)
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
            if not named or start is None or minutes_total < 3 * minutes:
                out.append(c)
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

    def create_chunks(self, daily_plan, agent, world=None):
        chunks = []
        for activity in daily_plan:
            activity = activity.strip().rstrip(".")
            chunks.append({
                "activity": activity,
                "location": self.resolve_location(activity, agent),
                "start_time": self.extract_start_time(activity),
                "duration": self.extract_duration(activity),
                "target_agent": self.extract_target_agent(activity, agent, world),
                "board_work": project_board.mentions_board_explicitly(activity),
            })

        return chunks

    def resolve_location(self, activity, agent, world=None, default=None):
        return self.resolve_location_keywords(activity, agent, default=default)

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
        - Do not stay in the current room just because it is the current room.
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

    def _chunk_from_activity(self, activity, agent, parent, duration=None):
        """Build a plan chunk from one parsed line, inheriting the parent's target."""
        activity = activity.strip().rstrip(".")
        chunk = {
            "activity": activity,
            "location": self.resolve_location(activity, agent),
            "start_time": self.extract_start_time(activity),
            "target_agent": parent.get("target_agent"),
            # A board visit stays a board visit when it is broken down, even if
            # a step reads "check all frontend tasks for updates".
            "board_work": bool(parent.get("board_work"))
                          or project_board.mentions_board_explicitly(activity),
        }
        if duration is not None:
            chunk["duration"] = duration
        return chunk

    def decompose_chunks(self, chunks, agent):
        """Broad blocks -> one chunk per hour (paper: hour-long chunks)."""
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

            hour_chunks = [
                self._chunk_from_activity(activity, agent, chunk, duration=60)
                for activity in self._parse_numbered(llm.generate(prompt))[:hours]
            ]
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

    def decompose_hour_chunks(self, hour_chunks, agent):
        """Hour chunks -> 5-15 minute actions (paper: finer-grained actions)."""
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

            Example Input:

            4:00 pm: Review and polish a music composition

            Example Output:

            1. 4:00 pm: Review the composition.
            2. 4:10 pm: Identify areas needing improvement.
            3. 4:20 pm: Refine weak sections.
            4. 4:35 pm: Listen through the revision.
            5. 4:50 pm: Organize notes and clean the workspace.
            """

            actions = [
                self._chunk_from_activity(activity, agent, chunk)
                for activity in self._parse_numbered(llm.generate(prompt))
            ]
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

        Should {agent["name"]} react to the observation, and if so, what would be an appropriate reaction?

        Answer in exactly this format:
        DECISION: 1 (continue the current plan) or 2 (react)
        REACTION: <one short sentence, or "none">
        """

        return llm.generate(
            prompt, options={**CTX, "num_predict": 80, "temperature": 0}
        )

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

    def extract_outcomes(self, transcript, agent_a, agent_b, world):
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

        1. Did they agree to meet or work together at a specific place right afterwards?
        ("your desk" means the office of the person being addressed.)
        If not, set meeting_agreed to false.
        2. For each person, what did they commit to doing (a task, preparation, or
        a meeting at a certain time)? Write one short sentence from their point
        of view, including any time or place. Small talk is not a commitment.
        Omit a person who committed to nothing.
        """
        schema = {
            "type": "object",
            "properties": {
                "meeting_agreed": {"type": "boolean"},
                "location": {"type": "string", "enum": valid},
                "activity": {"type": "string"},
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
                }
            for c in data["commitments"]:
                text = c["commitment"].strip().rstrip(".")
                if text:
                    out["commitments"][c["person"]] = text
        except Exception as e:
            log.warn(f"extract_outcomes failed ({e})")
        return out