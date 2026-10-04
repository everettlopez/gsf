from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from src.simulation.environment_tree import EnvironmentTree

tree = EnvironmentTree.load("data/world/environment_tree.json")

print(tree.get_children("cubicles"))