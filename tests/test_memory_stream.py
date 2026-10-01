from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from memory_stream import MemoryStream

seed_memories = [
    "Adam Beck is a backend software engineer who enjoys building reliable and scalable software systems.",
    "Adam Beck is always looking for ways to make applications faster, easier to maintain, and more useful for their users.",
    "Adam Beck lives in Salt Lake City, Utah.",
    "Adam Beck enjoys discussing software architecture, APIs, and distributed systems with other engineers.",
    "Adam Beck frequently collaborates with his colleagues on technical projects.",
    "Adam Beck enjoys helping others solve challenging problems.",
    "Adam Beck values clean code, continuous learning, and professional growth.",
    "Adam Beck primarily develops software using Python and TypeScript.",
    "Adam Beck enjoys drinking coffee while working.",
    "Adam Beck prefers sunny days over rainy days.",
    "Adam Beck hopes to become a highly respected software architect.",
    "Adam Beck wants to build systems that positively impact many people."
]

def addSeedMemories() -> MemoryStream:
    stream = MemoryStream()

    for memory in seed_memories:
        stream.add_observation(memory)

    return stream

def addNewMemories() -> MemoryStream:
    stream = MemoryStream()

    stream.add_observation("Adam Beck clocks in for the workday.")
    stream.add_observation("Adam Beck places his backpack underneath his desk.")
    stream.add_observation("Adam Beck turns on his monitor.")
    stream.add_observation("Adam Beck unlocks his work laptop.")
    stream.add_observation("Adam Beck opens Microsoft Teams and reviews unread messages.")
    stream.add_observation("Adam Beck receives a message from his project manager assigning a new memory-system task.")
    stream.add_observation("Adam Beck reads the task description and adds it to his notes.")
    stream.add_observation("Adam Beck joins the morning engineering standup meeting.")
    stream.add_observation("A coworker tells Adam Beck a joke before the meeting begins.")
    stream.add_observation("Adam Beck laughs and briefly discusses weekend plans with the coworker.")
    stream.add_observation("Adam Beck returns to his desk and opens the memory-stream project.")
    stream.add_observation("Adam Beck begins reviewing memory retrieval results from yesterday's testing.")

if __name__ == "__main__":
    # addSeedMemories()
    addNewMemories()