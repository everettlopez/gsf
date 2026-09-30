from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from memory_stream import MemoryStream

stream = MemoryStream()

stream.add_memory("Desk is idle.")

stream.add_memory("Coffee mug is half full.")

stream.add_memory("Monitor is displaying Jira.")

stream.add_memory(
    "PM assigned task to Agent to implement memory retrieval for the agent platform."
)

stream.add_memory(
    "Agent is reading the requirements for the memory retrieval task."
)

stream.add_memory(
    "PM says the first draft of the memory retrieval design is needed by Friday."
)

stream.add_memory(
    "Coworker is discussing the memory retrieval design with Agent."
)

stream.add_memory(
    "Coworker says retrieval quality depends on relevance, recency, and importance."
)

stream.add_memory("Meeting room is occupied.")

stream.add_memory(
    "Agent is taking notes about the memory retrieval task."
)

print("Memories have been added.")