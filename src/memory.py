from dataclasses import dataclass, field
from datetime import datetime
from typing import List

@dataclass
class Memory:
    id: int
    description: str
    importance: int
    memory_type: str = "observation"
    embedding: List[float] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed_at: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Reflection(Memory):
    source_memory_ids: List[int] = field(default_factory=list)
    memory_type: str = "reflection"