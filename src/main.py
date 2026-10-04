from world import *
from plan import Planner
from datetime import datetime, timedelta
from perceive import perceive_environment
import time 

adam = load_agent("adam")
adam_planner = Planner(adam)

eve = load_agent("eve")
eve_planner = Planner(eve)

world = load_world()
current_time = "7:00 am"
STEP_DELAY_SECONDS = 120

# Initial plan
adam_daily_plan = adam_planner.generate_initial_plan_adam()
adam_chunks = adam_planner.create_chunks(adam_daily_plan, adam)
print("ADAM'S BROAD PLAN")
adam_planner.create_plan(adam_chunks)
time.sleep(45)

eve_daily_plan = eve_planner.generate_initial_plan_eve()
eve_chunks = eve_planner.create_chunks(eve_daily_plan, eve)
print("EVE'S BROAD PLAN")
eve_planner.create_plan(eve_chunks)
time.sleep(45)

# DECOMPOSE CHUNKS (HOUR)
adam_decomposed_chunks = adam_planner.decompose_chunks(adam_chunks, adam)
print("ADAM DECOMPOSED CHUNKS")
adam_planner.create_plan(adam_decomposed_chunks)
time.sleep(45)

eve_decomposed_chunks = eve_planner.decompose_chunks(eve_chunks, eve)
print("EVE DECOMPOSED CHUNKS")
eve_planner.create_plan(eve_decomposed_chunks)
time.sleep(45)

# DECOMPOSE HOUR CHUNKS (MINUTES)

adam_minute_chunks = adam_planner.decompose_hour_chunks(adam_decomposed_chunks, adam)
print("ADAM MINUTE CHUNKS")
adam_planner.create_plan(adam_minute_chunks)
time.sleep(45)

eve_minute_chunks = eve_planner.decompose_hour_chunks(eve_decomposed_chunks, eve)
print("EVE MINUTE CHUNKS")
eve_planner.create_plan(eve_minute_chunks)
time.sleep(45)

def step_time(current_time, minutes=5):
    current_dt = datetime.strptime(
    current_time.upper(),
    "%I:%M %p"
    )
    
    new_dt = current_dt + timedelta(
    minutes=minutes
    )
    
    return new_dt.strftime(
    "%I:%M %p"
    ).lower()

def process_step(self, agent, current_time, minute_chunks):
    print("PROCESSING STEP")
    status = self.get_current_status(agent, current_time)
    print("Status: ", status)

    current_chunk = self.get_current_minute_chunk(current_time, minute_chunks)
    print("Current Chunk: ", current_chunk)

    decision = self.continue_plan_or_react(agent, status, current_chunk)
    print("Decision: ", decision)

    if decision.strip() == "1":
        print("Decision is 1")
        agent = self.continue_plan(agent, current_chunk)

    print("\n")
    return agent

for _ in range(30):
    start = time.time()
    print("\n" + "=" * 50)
    print(f"TIME: {current_time}")

    # PERCEIVE
    adam_observations = perceive_environment(adam, world)
    eve_observations = perceive_environment(eve, world)

    print("ADAM'S OBSERVATIONS")
    for observation in adam_observations:
        print(observation)

    print("EVE'S OBSERVATIONS")
    for observation in eve_observations:
        print(observation)

    
    # Status
    adam_status = adam_planner.get_current_status(adam, current_time)
    eve_status = eve_planner.get_current_status(eve, current_time)

    # AGENTS CURRENT CHUNKS
    current_chunk_adam = adam_planner.get_current_minute_chunk(current_time, adam_minute_chunks)
    current_chunk_eve = eve_planner.get_current_minute_chunk(current_time, eve_minute_chunks)

    print("CURRENT CHUNK (ADAM): ", current_chunk_adam)
    print("CURRENT CHUNK (EVE): ", current_chunk_eve)

    print("\n")

    # AGENTS STATUS
    print(f"STATUS (ADAM): {adam_status}")
    print(f"STATUS (EVE): {eve_status}")

    print("\n")

    # AGENTS PLANNED ACTIONS
    print(f"PLANNED ACTION (ADAM): {current_chunk_adam['activity']}")
    print(f"PLANNED ACTION (EVE): {current_chunk_eve['activity']}")

    print("\n")

    # AGENTS PLANNED LOCATIONS
    print(f"PLANNED LOCATION (ADAM): {current_chunk_adam['location']}")
    print(f"PLANNED LOCATION (EVE): {current_chunk_eve['location']}")

    print("\n")

    # DECISIONS
    adam_decision = adam_planner.continue_plan_or_react(adam, adam_status, current_chunk_adam, adam_observations, world)
    adam_decision = adam_decision.strip().replace(".", "")

    eve_decision = eve_planner.continue_plan_or_react(eve, eve_status, current_chunk_eve, eve_observations, world)
    eve_decision = eve_decision.strip().replace(".", "")

    print(f"DECISION (ADAM): {adam_decision}")
    print(f"DECISION (EVE): {eve_decision}")

    # ADAMS CHOICE
    if adam_decision.strip() == "1":
        adam = update_agent_state(
        adam,
        current_chunk_adam
        )

        adam, world = generate_interactions(adam, world, adam_observations)

        world["agents"]["adam"] = adam

        save_world(world)

        print(
            f"NEW ACTION (ADAM): "
            f"{adam['current_action']}"
        )

        print(
            f"NEW ROOM (ADAM): "
            f"{adam['current_location']}"
        )

        print(
            f"INTERACTING WITH (ADAM): "
            f"{adam['interacting_with']}"
        )
    elif adam_decision.strip() == "2":
        adam_daily_plan = adam_planner.regenerate_plan_adam(adam, current_time, current_chunk_adam, adam_observations, adam_status, world)
        adam_chunks = adam_planner.create_chunks(adam_daily_plan, adam)
        print("ADAM DAILY PLAN")
        adam_planner.create_plan(adam_chunks)
        time.sleep(45)

        adam_decomposed_chunks = adam_planner.decompose_chunks(adam_chunks, adam)
        print("ADAM DECOMPOSED (HOUR)")
        adam_planner.create_plan(adam_decomposed_chunks)
        time.sleep(45)

        adam_minute_chunks = adam_planner.decompose_hour_chunks(adam_decomposed_chunks, adam)
        print("ADAM DECOMPOSED (MINUTE)")
        adam_planner.create_plan(adam_minute_chunks)
        time.sleep(45)

        print("PLAN GENERATED (ADAM)")

        current_chunk = adam_planner.get_current_minute_chunk(current_time, adam_minute_chunks)
        print("NEW CURRENT CHUNK (ADAM)")
        print(current_chunk)

        adam = update_agent_state(adam, current_chunk)

        adam, world = generate_interactions(adam, world, adam_observations)

        world["agents"]["adam"] = adam

        save_world(world)

        print(
            f"NEW ACTION: "
            f"{adam['current_action']}"
            )
            
        print(
        f"NEW ROOM: "
        f"{adam['current_location']}"
        )

    if eve_decision.strip() == "1":
        eve = update_agent_state(
        eve,
        current_chunk_eve
        )

        eve, world = generate_interactions(eve, world, eve_observations)

        world["agents"]["eve"] = eve

        save_world(world)

        print(
            f"NEW ACTION (EVE): "
            f"{eve['current_action']}"
        )

        print(
            f"NEW ROOM (EVE): "
            f"{eve['current_location']}"
        )

        print(
            f"INTERACTING WITH (EVE): "
            f"{eve['interacting_with']}"
        )
    elif eve_decision.strip() == "2":
        eve_daily_plan = eve_planner.regenerate_plan_eve(eve, current_time, current_chunk_eve, eve_observations, eve_status, world)
        eve_chunks = eve_planner.create_chunks(eve_daily_plan, eve)
        print("EVE DAILY PLAN")
        eve_planner.create_plan(eve_chunks)
        time.sleep(45)

        eve_decomposed_chunks = eve_planner.decompose_chunks(eve_chunks, eve)
        print("EVE DECOMPOSED CHUNKS (HOUR)")
        eve_planner.create_plan(eve_decomposed_chunks)
        time.sleep(45)

        eve_minute_chunks = eve_planner.decompose_hour_chunks(eve_decomposed_chunks, eve)
        print("EVE DECOMPOSED CHUNKS (MINUTE)")
        eve_planner.create_plan(eve_minute_chunks)
        time.sleep(45)

        print("PLAN GENERATED (EVE)")

        current_chunk = eve_planner.get_current_minute_chunk(current_time, eve_minute_chunks)
        print("NEW CURRENT CHUNK (EVE):")
        print(current_chunk)

        eve = update_agent_state(eve, current_chunk)

        eve, world = generate_interactions(eve, world, eve_observations)

        world["agents"]["eve"] = eve
        
        save_world(world)

        print(
            f"NEW ACTION (EVE): "
            f"{eve['current_action']}"
            )
            
        print(
        f"NEW ROOM (EVE): "
        f"{eve['current_location']}"
        )

    current_time = step_time(current_time)

    elapsed = time.time() - start
    sleep_time = max(0, STEP_DELAY_SECONDS - elapsed)

    print(f"Sleeping for {sleep_time:.1f} seconds...")
    time.sleep(sleep_time)