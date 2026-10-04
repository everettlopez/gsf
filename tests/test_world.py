from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from world import *
from perceive import *

world = load_world()
adam = load_agent("adam")

observations = perceive_environment(adam, world)

adam, world = generate_interactions(adam, world, observations)

save_agent(adam)
save_world(world)

print("INTERACTING WITH:")
print(adam["interacting_with"])
print(world)