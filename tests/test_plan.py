from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from plan import Planner
from perceive import *
from world import *

planner = Planner()
world = load_world()
adam = load_agent("adam")

current_time = "7:10 am"
print(adam)


daily_plan = planner.generate_initial_plan()
chunks = planner.create_chunks(daily_plan)
decomposed_chunks = planner.decompose_chunks(chunks)
minute_chunks = planner.decompose_hour_chunks(decomposed_chunks)


observations = perceive_environment(adam, world)

for observation in observations:
    print(observation)

current_chunk = planner.get_current_minute_chunk(current_time, minute_chunks)

new_plan = planner.regenerate_plan(adam, current_time, current_chunk, observations)

print(new_plan)


# planner.create_plan(minute_chunks)

# for chunk in decomposed_chunks:
#     print(chunk)

# for chunk in minute_chunks:
#     print(chunk)