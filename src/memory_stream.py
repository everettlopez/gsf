from dataclasses import asdict
from pathlib import Path
from memory import Memory, Reflection
import json
import ollama
import re
import os, time 


class MemoryStream:
    def __init__(self, agent_name: str):
        root_dir = Path(__file__).parent.parent
        self.path = (root_dir / "data" / "agents" / agent_name / "memory_stream.json")
        self._importance_cache = {}
        self.dirty = False
        self.memories = self.load()

    def load(self):
        if not self.path.exists():
            return []

        with open(self.path, "r") as f:
            memories = json.load(f)

        loaded_memories = []

        for memory in memories:
            if memory.get("memory_type") == "reflection":
                loaded_memories.append(
                    Reflection(**memory)
                )
            else:
                loaded_memories.append(
                    Memory(**memory)
                )

        return loaded_memories


    def save(self, memories):
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
                print(f"    [save] {e}, retrying")
                time.sleep(0.2 * (attempt + 1))

    def mark_dirty(self):
        self.dirty = True

    def flush(self):
        if self.dirty:
            self.save(self.memories)

    # ADD SINGLE OBSERVATION
    def add_observation(self, description):

        importance = self.generate_importance_score(description)
        embedding = self.generate_embedding(description)
        next_id = len(self.memories) + 1

        memory = Memory(
            id=next_id,
            description=description,
            importance=importance,
            embedding=embedding
        )

        self.memories.append(memory)

        self.mark_dirty()

        return memory

    # BATCHED OBSERVATIONS CREATION
    def add_observations(self, descriptions):
        embeddings = ollama.embed(model="nomic-embed-text", input=descriptions)["embeddings"]
        created = []

        for description, embedding in zip(descriptions, embeddings):
            memory = Memory(
                id=len(self.memories) + 1,
                description=description,
                importance=self.cached_importance(description),
                embedding=embedding
            )

            self.memories.append(memory)
            created.append(memory)

        self.mark_dirty()
        return created

    def cached_importance(self, description):
        if description not in self._importance_cache:
            try:
                self._importance_cache[description] = self.generate_importance_score(description)
            except Exception:
                self._importance_cache[description] = 5
        return self._importance_cache[description]

    def add_reflection(self, description, source_memory_ids):
        embedding = self.generate_embedding(description)
        importance = self.generate_importance_score(description)
        next_id = len(self.memories) + 1

        reflection = Reflection(
            id=next_id,
            description=description,
            importance=importance,
            embedding=embedding,
            source_memory_ids=source_memory_ids
        )

        self.memories.append(reflection)
        self.mark_dirty()

        return reflection

    def get_memory_stream(self):
        return self.memories

    # Helper functions
    def generate_importance_score(self, description):

        importance_prompt = f"""
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

        response = ollama.generate(
            model="qwen3",
            prompt=importance_prompt,
            think=False
        )
    
        return int(response["response"].strip())

    def generate_embedding(self, text):
        response = ollama.embed(
            model="nomic-embed-text",
            input=text
        )

        return response["embeddings"][0]

    def add_plan(self, description):
        plan = Memory(
            id=len(self.memories) + 1,
            description=description,
            memory_type="plan",
            embedding=self.generate_embedding(description),
            importance=5,
        )
        self.memories.append(plan)
        self.mark_dirty()
        return plan