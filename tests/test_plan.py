from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from plan import Planner

planner = Planner()

daily_plan = planner.generate_initial_plan()

chunks = planner.create_chunks(daily_plan)

planner.create_plan(chunks)

print("\n")

decomposed_chunks = planner.decompose_chunks(chunks)

planner.create_plan(decomposed_chunks)

print("\n")

minute_chunks = planner.decompose_hour_chunks(decomposed_chunks)

planner.create_plan(minute_chunks)

# for chunk in decomposed_chunks:
#     print(chunk)

# for chunk in minute_chunks:
#     print(chunk)