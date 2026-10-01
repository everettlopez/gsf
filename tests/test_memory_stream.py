from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from memory_stream import MemoryStream

seed_memories = [
    "Adam Beck is a backend software engineer.",
    "Adam Beck enjoys building reliable software systems.",
    "Adam Beck enjoys building scalable software systems.",
    "Adam Beck lives in Salt Lake City, Utah.",
    "Adam Beck enjoys discussing software architecture.",
    "Adam Beck enjoys discussing APIs and distributed systems.",
    "Adam Beck frequently collaborates with other engineers.",
    "Adam Beck enjoys helping others solve technical problems.",
    "Adam Beck values clean code.",
    "Adam Beck primarily develops software using Python and TypeScript.",
    "Adam Beck hopes to become a software architect.",
    "Adam Beck wants to build software that positively impacts many people.",
]

def addSeedMemories() -> MemoryStream:
    stream = MemoryStream()

    for memory in seed_memories:
        stream.add_observation(memory)

    return stream

def addNewMemories() -> MemoryStream:
    stream = MemoryStream()

    stream.add_observation("Adam Beck is reading a message about upcoming testing plans.")
    stream.add_observation("Adam Beck is attending a project meeting about the memory-system implementation.")
    stream.add_observation("A coworker is explaining a recent testing result to Adam Beck.")
    stream.add_observation("Adam Beck is reading documentation about agent memory behavior.")
    stream.add_observation("Adam Beck is discussing project timelines with a coworker.")
    stream.add_observation("Adam Beck is reviewing the latest project updates from the engineering team.")
    

if __name__ == "__main__":
    # addSeedMemories()
    addNewMemories()