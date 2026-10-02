from memory_stream import MemoryStream
from memory import Memory
from datetime import datetime

import ollama 
import re

class Planner:
    def __init__(self):
        self.memory_stream = MemoryStream()

    def generate_initial_plan(self):

        prompt = f"""
        Name: Adam Beck (age: 24)

        Innate traits: collaborative, technical, growth-oriented

        Adam Beck is a backend software engineer who primarily develops software using Python and TypeScript.

        He enjoys building reliable and scalable software systems and values clean code.

        Adam Beck is currently working on a memory-system project. He has recently been focused on memory retrieval, memory-stream architecture, testing, and implementation work.

        Adam Beck works closely with other engineers and enjoys collaborating on technical challenges.

        Adam Beck hopes to become a software architect. He wants to build software that positively impacts many people.

        Adam Beck is making good progress on the memory-system project and wants to continue improving it in the coming days.

        On Wednesday, Adam:
        1) clocked in to work at 7:00 am,
        2) attended engineering meetings from 9:00 am to 10:00 am,
        3) worked on memory-system development from 10:00 am to 1:00 pm,
        4) took a lunch break in the break room from 1:00 pm to 2:00 pm,
        5) collaborated with coworkers on technical challenges from 2:00 pm to 4:00 pm,
        6) reviewed project updates, documentation, and testing progress from 4:00 pm to 5:00 pm,
        7) clocked out and finished work at 5:00 pm.

        Today is Thursday.

        Here is Adam's plan today in broad strokes:

        Continue the plan as a numbered list.

        Return the plan in broad-strokes time blocks.

        Example:
        3. work on memory-system development from 10:00 am to 1:00 pm
        """

        response = ollama.generate(
            model="qwen3",
            prompt=prompt,
            think=False
        )

        plan_text = response["response"]

        plan_items = []

        for line in plan_text.splitlines():
            line = line.strip()

            match = re.match(r"^\d+\.\s*(.+)$", line)

            if not match:
                continue

            item = match.group(1)

            item = item.replace("**", "").strip()

            plan_items.append(item)
        return plan_items

    def create_chunks(self, daily_plan):
        chunks = []

        for activity in daily_plan:
            chunks.append({
                "activity": activity,
                "location": self.resolve_location(activity),
                "start_time": self.extract_start_time(activity),
                "duration": self.extract_duration(activity)
            })

        return chunks

    def resolve_location(self, activity):
        activity = activity.lower()

        if "clock in" in activity:
            return "Reception"

        if "clock out" in activity:
            return "Reception"

        if "meeting" in activity:
            return "Conference Room"

        if "collaborate" in activity:
            return "Conference Room - Whiteboard"

        if "break" in activity:
            return "Break Room"

        return "Cubicle" 

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
        r"from\s+(\d{1,2}:\d{2}\s*(?:am|pm))",
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
            r"from\s+(\d{1,2}):(\d{2})\s*(am|pm)\s+to\s+(\d{1,2}):(\d{2})\s*(am|pm)",
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
        memories = self.memory_stream.get_memory_stream()
        description = "\n".join(
        f"{chunk['activity']} in {chunk['location']} for {chunk['duration']} minutes"
        for chunk in chunks
        )
        
        plan = Memory(
            id=len(memories) + 1,
            description=description,
            memory_type="plan",
            embedding=self.memory_stream.generate_embedding(description),
            importance=5
        )

        memories.append(plan)

        self.memory_stream.save(memories)

        print(plan.description)

        return plan

    def decompose_chunks(self, chunks):
        decomposed_chunks = []

        for chunk in chunks:
            if chunk["duration"] < 60:
                continue

            prompt = f"""
            The following activity is one entry from a daily plan.

            Create EXACTLY {chunk["duration"] // 60} hourly actions.

            The first action starts at the activity's start time.
            Each following action occurs exactly one hour later.

            DO NOT create additional hours.
            DO NOT include the end time as another action.
            DO NOT create more than {chunk["duration"] // 60} actions.

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

            response = ollama.generate(
                model="qwen3",
                prompt=prompt,
                think=False
            )

            for line in response["response"].splitlines():
                line = line.strip()
                
                match = re.match(
                r"^\d+\.\s*(.+)$",
                line
                )
                
                if not match:
                    continue
                
                activity = match.group(1).strip()
                
                decomposed_chunks.append({
                "activity": activity,
                "location": chunk["location"],
                "start_time": self.extract_start_time(activity),
                "duration": 60
                })
                
        return decomposed_chunks

    def decompose_hour_chunks(self, hour_chunks):

        minute_chunks = []

        for chunk in hour_chunks:

            prompt = f"""
            The following activity is one entry from an hourly plan.

            Break it into actions lasting roughly 5 to 15 minutes.

            Rules:
            - Create actions lasting roughly 5 to 15 minutes.
            - Actions should be sequential.
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

            response = ollama.generate(
                model="qwen3",
                prompt=prompt,
                think=False
            )

            actions = []

            for line in response["response"].splitlines():

                line = line.strip()

                match = re.match(
                    r"^\d+\.\s*(.+)$",
                    line
                )

                if not match:
                    continue

                activity = match.group(1).strip()

                actions.append({
                    "activity": activity,
                    "location": chunk["location"],
                    "start_time": self.extract_start_time(activity),
                })

            #
            # Calculate durations from timestamps
            #
            for i, action in enumerate(actions):

                if not action["start_time"]:
                    action["duration"] = 10
                    continue

                if i < len(actions) - 1:

                    next_start = actions[i + 1]["start_time"]

                    try:
                        start_dt = datetime.strptime(
                            action["start_time"].upper(),
                            "%I:%M %p"
                        )

                        next_dt = datetime.strptime(
                            next_start.upper(),
                            "%I:%M %p"
                        )

                        action["duration"] = int(
                            (next_dt - start_dt).total_seconds() / 60
                        )

                    except Exception:
                        action["duration"] = 10

                else:
                    #
                    # Last action fills remaining hour
                    #
                    try:

                        start_dt = datetime.strptime(
                            action["start_time"].upper(),
                            "%I:%M %p"
                        )

                        hour_start = datetime.strptime(
                            chunk["start_time"].upper(),
                            "%I:%M %p"
                        )

                        elapsed = int(
                            (start_dt - hour_start).total_seconds() / 60
                        )

                        action["duration"] = max(
                            5,
                            60 - elapsed
                        )

                    except Exception:
                        action["duration"] = 10

            minute_chunks.extend(actions)

        return minute_chunks

    def minutes_between(self, start_time, end_time):
        fmt = "%I:%M %p"
        
        start = datetime.strptime(
        start_time.upper(),
        fmt
        )
        
        end = datetime.strptime(
        end_time.upper(),
        fmt
        )
        
        return int(
        (end - start).total_seconds() / 60
        )