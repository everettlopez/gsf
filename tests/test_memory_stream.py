from pathlib import Path
import sys
import time

sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from memory_stream import MemoryStream


adam_seed_memories = [
    "Adam Beck is a 24 year old male.",
    "Adam Beck is a backend software engineer.",
    "Adam Beck enjoys building reliable software systems.",
    "Adam Beck enjoys building scalable software systems.",
    "Adam Beck lives in Salt Lake City, Utah.",
    "Adam Beck enjoys discussing APIs and distributed systems.",
    "Adam Beck frequently collaborates with other engineers.",
    "Adam Beck enjoys helping others solve technical problems.",
    "Adam Beck values clean code.",
    "Adam Beck primarily develops software using Python and TypeScript.",
    "Adam Beck hopes to become a software architect.",
    "Adam Beck wants to build software that positively impacts many people.",
    "Adam Beck works on the same team as Eve Smith.",
    "Adam Beck builds the backend APIs that Eve Smith's interfaces use.",
    "Adam Beck knows Eve Smith is shy in large groups.",
    "Adam Beck thinks Eve Smith's ideas are thoughtful.",
    "Adam Beck prefers to check in with Eve Smith one-on-one.",
]

eve_seed_memories = [
    "Eve Smith is a 21 year old female.",
    "Eve Smith is a frontend software engineer.",
    "Eve Smith is shy and tends to stay quiet in large group settings.",
    "Eve Smith enjoys building polished, accessible user interfaces.",
    "Eve Smith lives in Salt Lake City, Utah.",
    "Eve Smith enjoys quiet, focused work on UI details like animations and spacing.",
    "Eve Smith prefers collaborating one-on-one or in writing rather than in big meetings.",
    "Eve Smith opens up and shares ideas more once she feels comfortable with her teammates.",
    "Eve Smith values clean, well-organized code.",
    "Eve Smith primarily develops software using TypeScript and React.",
    "Eve Smith hopes to become a confident lead frontend engineer.",
    "Eve Smith wants to build software that feels simple and welcoming to use.",
    "Eve Smith works on the same team as Adam Beck.",
    "Eve Smith builds the interfaces that use Adam Beck's backend APIs.",
    "Eve Smith finds Adam Beck approachable.",
    "Eve Smith feels comfortable asking Adam Beck for help.",
    "Eve Smith is building a feature with Adam Beck.",
]

def addAdamSeedMemories() -> MemoryStream:
    stream = MemoryStream("adam")

    for memory in adam_seed_memories:
        stream.add_observation(memory)
        print("SEED MEMORY ADDED")

    return stream

def addEveSeedMemories() -> MemoryStream:
    stream = MemoryStream("eve")

    for memory in eve_seed_memories:
        stream.add_observation(memory)
        print("SEED MEMORY ADDED")

    return stream

if __name__ == "__main__":
    addAdamSeedMemories()
    addEveSeedMemories()