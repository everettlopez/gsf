from dataclasses import dataclass, field
from typing import List

import clock

@dataclass
class Memory:
    id: int
    description: str
    importance: int
    memory_type: str = "observation"
    embedding: List[float] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: clock.now().isoformat())
    last_accessed_at: str = field(default_factory=lambda: clock.now().isoformat())

@dataclass
class Reflection(Memory):
    source_memory_ids: List[int] = field(default_factory=list)
    memory_type: str = "reflection"