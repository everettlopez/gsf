from dataclasses import asdict
from pathlib import Path
from memory import Memory, Reflection
import json
import ollama


class MemoryStream:
    def __init__(self):
        root_dir = Path(__file__).parent.parent

        self.path = root_dir / "data" / "memory-stream.json"

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
        with open(self.path, "w") as f:
            json.dump(
                [asdict(memory) for memory in memories],
                f,
                indent=2
            )
    
    def add_observation(self, description):
        memories = self.load()

        importance = self.generate_importance_score(description)
        embedding = self.generate_embedding(description)
        next_id = len(memories) + 1

        memory = Memory(
            id=next_id,
            description=description,
            importance=importance,
            embedding=embedding
        )

        memories.append(memory)

        self.save(memories)

        return memory

    def add_reflection(self, description, source_memory_ids):
        memories = self.load()
        embedding = self.generate_embedding(description)
        importance = self.generate_importance_score(description)
        next_id = len(memories) + 1

        reflection = Reflection(
            id=next_id,
            description=description,
            importance=importance,
            embedding=embedding,
            source_memory_ids=source_memory_ids
        )

        memories.append(reflection)
        self.save(memories)

        return reflection

    def get_memory_stream(self):
        return self.load()

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