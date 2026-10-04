from pathlib import Path
import sys


sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from perceive import *
from world import *

world = load_world()
adam = load_agent("adam")

observations = perceive_environment(adam, world)

print(observations)