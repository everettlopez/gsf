from dataclasses import asdict
from pathlib import Path
import json
import os
import re
import time

import llm
import log
from memory import Memory, Reflection

DEFAULT_IMPORTANCE = 5
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_importance(text, default=DEFAULT_IMPORTANCE):
    """Pull the first integer out of an LLM reply and clamp it to 1-10."""
    match = re.search(r"\d+", text or "")
    if not match:
        return default
    return max(1, min(10, int(match.group(0))))


class MemoryStream:
    def __init__(self, agent_name: str):
        self.path = DATA_DIR / "agents" / agent_name / "memory_stream.json"
        self._importance_cache = {}
        self.dirty = False
        self.memories = self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self):
        if not self.path.exists():
            return []

        with open(self.path, "r") as f:
            memories = json.load(f)

        loaded_memories = []

        for memory in memories:
            if memory.get("memory_type") == "reflection":
                loaded_memories.append(Reflection(**memory))
            else:
                loaded_memories.append(Memory(**memory))

        return loaded_memories

    def save(self, memories):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{self.path}.tmp"
        data = json.dumps([asdict(m) for m in memories], indent=2)
        for attempt in range(5):
            try:
                with open(tmp, "w") as f:
                    f.write(data)
                os.replace(tmp, self.path)
                self.dirty = False
                return
            except OSError as e:
                if attempt == 4:
                    raise
                log.warn(f"memory save failed ({e}), retrying")
                time.sleep(0.2 * (attempt + 1))

    def mark_dirty(self):
        self.dirty = True

    def flush(self):
        if self.dirty:
            self.save(self.memories)

    # ------------------------------------------------------------------
    # Adding memories
    # ------------------------------------------------------------------
    def _next_id(self):
        return max((m.id for m in self.memories), default=0) + 1

    def add_observation(self, description):
        memory = Memory(
            id=self._next_id(),
            description=description,
            importance=self.cached_importance(description),
            embedding=llm.embed(description),
        )
        self.memories.append(memory)
        self.mark_dirty()
        return memory

    def add_observations(self, descriptions):
        """Batched version of add_observation (one embedding call)."""
        if not descriptions:
            return []

        embeddings = llm.embed(list(descriptions))
        created = []

        for description, embedding in zip(descriptions, embeddings):
            memory = Memory(
                id=self._next_id(),
                description=description,
                importance=self.cached_importance(description),
                embedding=embedding,
            )
            self.memories.append(memory)
            created.append(memory)

        self.mark_dirty()
        return created

    def add_reflection(self, description, source_memory_ids):
        reflection = Reflection(
            id=self._next_id(),
            description=description,
            importance=self.cached_importance(description),
            embedding=llm.embed(description),
            source_memory_ids=source_memory_ids,
        )
        self.memories.append(reflection)
        self.mark_dirty()
        return reflection

    def add_plan(self, description):
        plan = Memory(
            id=self._next_id(),
            description=description,
            memory_type="plan",
            embedding=llm.embed(description),
            importance=DEFAULT_IMPORTANCE,
        )
        self.memories.append(plan)
        self.mark_dirty()
        return plan

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def get_memory_stream(self):
        return self.memories

    def last_reflection_id(self):
        """Id of the newest reflection (0 if the agent has never reflected).
        Reflections are appended after everything they considered, so any
        memory with a higher id has not been reflected on yet."""
        return max(
            (m.id for m in self.memories if m.memory_type == "reflection"),
            default=0,
        )

    def unreflected(self, limit=100):
        """The `limit` most recent observations since the last reflection
        (the paper reflects over the 100 most recent records)."""
        boundary = self.last_reflection_id()
        fresh = [
            m for m in self.memories
            if m.id > boundary and m.memory_type == "observation"
        ]
        return fresh[-limit:]

    # ------------------------------------------------------------------
    # Importance scoring
    # ------------------------------------------------------------------
    def cached_importance(self, description):
        if description not in self._importance_cache:
            try:
                self._importance_cache[description] = self.generate_importance_score(description)
            except Exception as e:
                log.debug(f"importance scoring failed ({e}); using {DEFAULT_IMPORTANCE}")
                self._importance_cache[description] = DEFAULT_IMPORTANCE
        return self._importance_cache[description]

    def generate_importance_score(self, description):
        prompt = f"""
        On the scale of 1 to 10, where

        1 is purely mundane
        (e.g., checking email, drinking coffee, attending a routine meeting, desk is idle)

        and

        10 is extremely poignant
        (e.g., critical production outage, employee termination, major promotion, major customer data breach)

        rate the likely poignancy of the following piece of memory.

        Memory:
        {description}

        Respond with only a single integer.
        """
        return parse_importance(llm.generate(prompt))

    def generate_embedding(self, text):
        return llm.embed(text)
