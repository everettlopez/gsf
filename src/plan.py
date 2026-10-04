from memory_stream import MemoryStream
from memory_retriever import MemoryRetriever
from memory import Memory
from datetime import datetime, timedelta

import ollama 
import re


class Planner:
    def __init__(self, agent):
        self.memory_stream = MemoryStream(agent["id"])
        self.memory_retriever = MemoryRetriever(agent["id"])
        self.memories = self.memory_stream.load()

    def generate_initial_plan_adam(self):

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
        1) clock in and start work at 7:00 am,
        2) attended engineering meetings from 8:00 am to 9:00 am,
        3) worked on memory-system development from 9:00 am to 12:00 pm,
        4) took a lunch break in the break room from 12:00 pm to 1:00 pm,
        5) collaborated with coworkers on technical challenges from 1:00 pm to 2:00 pm,
        6) reviewed project updates, documentation, and testing progress from 2:00 pm to 5:00 pm,
        7) clock out and finished work at 5:00 pm.

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

    def generate_initial_plan_eve(self):
    
        prompt = f"""
        Name: Eve Smith (age: 21)

        Innate traits: shy, detail-oriented, thoughtful

        Eve Smith is a frontend software engineer who primarily develops software using TypeScript and React.

        She enjoys building polished and accessible user interfaces and values clean code.

        Eve Smith is currently working on a navigation menu redesign project. She has recently been focused on UI components, responsive layouts, accessibility, and implementation work.

        Eve Smith works with designers and other engineers, and prefers collaborating one-on-one or in writing on technical challenges.

        Eve Smith hopes to become a lead frontend engineer. She wants to build software that feels simple and welcoming for many people to use.

        Eve Smith is making good progress on the navigation menu redesign and wants to continue improving it in the coming days.

        On Wednesday, Eve:
        1) clock in and start work at 7:00 am,
        2) attended engineering meetings from 8:00 am to 9:00 am,
        3) worked on navigation menu development from 9:00 am to 12:00 pm,
        4) took a lunch break in the break room from 12:00 pm to 1:00 pm,
        5) collaborated with coworkers on technical challenges from 1:00 pm to 2:00 pm,
        6) reviewed project updates, documentation, and testing progress from 2:00 pm to 5:00 pm,
        7) clock out and finished work at 5:00 pm.

        Today is Thursday.

        Here is Eve's plan today in broad strokes:

        Continue the plan as a numbered list.

        Return the plan in broad-strokes time blocks.

        Example:
        3. work on navigation menu development from 10:00 am to 1:00 pm
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

    def regenerate_plan_adam(self, agent, current_time, current_chunk, observations, status, world):
        agent_summary = self.generate_agent_summary(agent)
        observations_text = "\n".join(
            f"Observation: {observation}" for observation in observations
        )

        context_memories = []
        seen = set()

        room = agent["current_location"]

        entities = []

        # ROOM ITSELF
        entities.append((room, world["locations"][room]["state"]))

        # OBJECTS IN ROOM
        for object_name, object_data in (world["locations"][room]["objects"].items()):
            entities.append((object_name, object_data["state"]))

        # agents in the room
        for other_agent in world["agents"].values():
            if(other_agent["id"] != agent["id"] and other_agent["current_location"] == room):
                entities.append((other_agent["name"], other_agent["current_action"]))

        for entity_name, entity_status in entities:
            queries = [
                f"What is {agent["name"]}'s relationship with {entity_name}",
                f"{entity_name} is {entity_status}"
            ]

            for query in queries:
                results = self.memory_retriever.retrieve(query)

                for result in results:
                    description = (result["memory"].description)

                    if description not in seen:
                        seen.add(description)
                        context_memories.append(result)



        context = "\n".join(
        f"- {result['memory'].description}"
        for result in context_memories
        )

        prompt = f"""
        Name: Adam Beck (age: 24)
        
        Innate traits: collaborative, technical, growth-oriented

        Adam Beck is a backend software engineer who primarily develops software using Python and TypeScript.

        He enjoys building reliable and scalable software systems and values clean code.

        Adam Beck is currently working on a memory-system project. He has recently been focused on memory retrieval, memory-stream architecture, testing, and implementation work.

        Adam Beck works closely with other engineers and enjoys collaborating on technical challenges.

        Adam Beck hopes to become a software architect. He wants to build software that positively impacts many people.

        Adam Beck is making good progress on the memory-system project and wants to continue improving it in the coming days.

        Summary of relevant context from Adam's memory:
        {context}

        On Wednesday, Adam:
        1) clocked in to work at 7:00 am,
        2) attended engineering meetings from 9:00 am to 10:00 am,
        3) worked on memory-system development from 10:00 am to 1:00 pm,
        4) took a lunch break in the break room from 1:00 pm to 2:00 pm,
        5) collaborated with coworkers on technical challenges from 2:00 pm to 4:00 pm,
        6) reviewed project updates, documentation, and testing progress from 4:00 pm to 5:00 pm,
        7) clocked out and finished work at 5:00 pm.

        Today is {current_time} on Thursday

        Here is Adam's plan today in broad strokes:

        Continue the plan as a numbered list.

        Return the plan in broad-strokes time blocks.

        Example:
        3. work on memory-system development from {current_time} to 1:00 pm
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

    def regenerate_plan_eve(self, agent, current_time, current_chunk, observations, status, world):
        agent_summary = self.generate_agent_summary(agent)
        observations_text = "\n".join(
            f"Observation: {observation}" for observation in observations
        )

        context_memories = []
        seen = set()

        room = agent["current_location"]

        entities = []

        # ROOM ITSELF
        entities.append((room, world["locations"][room]["state"]))

        # OBJECTS IN ROOM
        for object_name, object_data in (world["locations"][room]["objects"].items()):
            entities.append((object_name, object_data["state"]))

        # agents in the room
        for other_agent in world["agents"].values():
            if(other_agent["id"] != agent["id"] and other_agent["current_location"] == room):
                entities.append((other_agent["name"], other_agent["current_action"]))

        for entity_name, entity_status in entities:
            queries = [
                f"What is {agent["name"]}'s relationship with {entity_name}",
                f"{entity_name} is {entity_status}"
            ]

            for query in queries:
                results = self.memory_retriever.retrieve(query)

                for result in results:
                    description = (result["memory"].description)

                    if description not in seen:
                        seen.add(description)
                        context_memories.append(result)

        context = "\n".join(
        f"- {result['memory'].description}"
        for result in context_memories
        )

        prompt = f"""
        Name: Eve Smith (age: 21)
        
        Innate traits: shy, detail-oriented, thoughtful

        Eve Smith is a frontend software engineer who primarily develops software using TypeScript and React.

        She enjoys building polished and accessible user interfaces and values clean code.

        Eve Smith is currently working on a navigation menu redesign project. She has recently been focused on UI components, responsive layouts, accessibility, and implementation work.

        Eve Smith works with designers and other engineers, and prefers collaborating one-on-one or in writing on technical challenges.

        Eve Smith hopes to become a lead frontend engineer. She wants to build software that feels simple and welcoming for many people to use.

        Eve Smith is making good progress on the navigation menu redesign and wants to continue improving it in the coming days.

        Summary of relevant context from Adam's memory:
        {context}

        On Wednesday, Eve:
        1) clocked in to work at 7:00 am,
        2) attended engineering meetings from 9:00 am to 10:00 am,
        3) worked on navigation menu development from 10:00 am to 1:00 pm,
        4) took a lunch break in the break room from 1:00 pm to 2:00 pm,
        5) collaborated with coworkers on technical challenges from 2:00 pm to 4:00 pm,
        6) reviewed project updates, documentation, and testing progress from 4:00 pm to 5:00 pm,
        7) clocked out and finished work at 5:00 pm.

        Today is {current_time} on Thursday
    
        Here is Eve's plan today in broad strokes:

        Continue the plan as a numbered list.

        Return the plan in broad-strokes time blocks.

        Example:
        3. work on navigation menu development from {current_time} to 1:00 pm
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

    def create_chunks(self, daily_plan, agent):
        chunks = []

        for activity in daily_plan:
            chunks.append({
                "activity": activity,
                "location": self.resolve_location(activity, agent),
                "start_time": self.extract_start_time(activity),
                "duration": self.extract_duration(activity)
            })

        return chunks

    def resolve_location(self, activity, agent):
        activity = activity.lower()

        if "clock in" in activity:
            return "reception"

        if "reception" in activity:
            return "reception"

        if "clock out" in activity:
            return "reception"

        if "conference" in activity:
            return "conference room"

        if "lunch" in activity:
            return "break room"

        if "collaborate" in activity:
            return "conference room"

        if "break" in activity:
            return "break room"

        return agent["office"]

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
        description = "\n".join(
        f"{chunk['activity']} in {chunk['location']} for {chunk['duration']} minutes"
        for chunk in chunks
        )
        
        plan = Memory(
            id=len(self.memories) + 1,
            description=description,
            memory_type="plan",
            embedding=self.memory_stream.generate_embedding(description),
            importance=5
        )

        self.memories.append(plan)

        self.memory_stream.save(self.memories)

        print(plan.description)

        return plan

    def decompose_chunks(self, chunks, agent):
        decomposed_chunks = []

        for chunk in chunks:
            if chunk["duration"] < 60:
                decomposed_chunks.append(chunk)
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
                "location": self.resolve_location(activity, agent),
                "start_time": self.extract_start_time(activity),
                "duration": 60
                })
                
        return decomposed_chunks

    def decompose_hour_chunks(self, hour_chunks, agent):

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
                    "location": self.resolve_location(activity, agent),
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

    def get_current_status(self, agent, current_time):
        return (
            f"It is {current_time}. "
            f"{agent["name"]} is currently at {agent["current_location"]} and is {agent["current_action"]}."
        )

    def get_current_minute_chunk(
        self,
        current_time,
        minute_chunks
    ):
        current_dt = datetime.strptime(
            current_time.upper(),
            "%I:%M %p"
        )

        for chunk in minute_chunks:
            start_dt = datetime.strptime(
                chunk["start_time"].upper(),
                "%I:%M %p"
            )

            end_dt = start_dt + timedelta(
                minutes=chunk["duration"]
            )

            if start_dt <= current_dt < end_dt:
                return chunk

        return None

    def continue_plan_or_react(self, agent, status, current_chunk, observations, world):

        agent_summary = self.generate_agent_summary(agent)
        observations_text = "\n".join(
            f"Observation: {observation}" for observation in observations
        )
        
        context_memories = []
        seen = set()

        room = agent["current_location"]

        entities = []

        # Room itself
        entities.append((room, world["locations"][room]["state"]))

        # objects in room
        for object_name, object_data in (world["locations"][room]["objects"].items()):
            entities.append((object_name, object_data["state"]))

        # agenrs in the room
        for other_agent in world["agents"].values():
            if(other_agent["id"] != agent["id"] and other_agent["current_location"] == room):
                entities.append((other_agent["name"], other_agent["current_action"]))

        for entity_name, entity_status in entities:
            queries = [
                f"What is {agent["name"]}'s relationship with {entity_name}",
                f"{entity_name} is {entity_status}"
            ]

            for query in queries:
                results = self.memory_retriever.retrieve(query)

                for result in results:
                    description = (result["memory"].description)

                    if description not in seen:
                        seen.add(description)
                        context_memories.append(result)

        context = "\n".join(
        f"- {result['memory'].description}"
        for result in context_memories
        )

        

        prompt = f"""
        {agent_summary}

        {agent["name"]}'s status: {status}

        {observations_text}

        Current planned action: {current_chunk["activity"]}
        Current planned location: {current_chunk["location"]}

        Summary of relevant context from {agent["name"]}'s memory:
        {context}

        Should {agent["name"]} react to any of the observed entities or their current state, and if so, what would be an appropriate reaction?

        1. Continue the current planned action
        2. React to the current sitation

        Respond with an integer 1, or 2.
        """

        response = ollama.generate(
            model="qwen3",
            prompt=prompt,
            think=False
        )
    
        return response["response"]

    def generate_agent_summary(self, agent):
        core_memories = self.memory_retriever.retrieve(
            f"{agent['name']}'s core characteristics"
        )

        occupation_memories = self.memory_retriever.retrieve(
            f"{agent['name']}'s current daily occupation"
        )

        progress_memories = self.memory_retriever.retrieve(
            f"{agent['name']}'s feeling about recent progress in life"
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

        response = ollama.generate(
            model="qwen3",
            prompt=prompt,
            think=False
        )

        return response["response"]