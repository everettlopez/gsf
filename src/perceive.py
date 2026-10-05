from memory_stream import MemoryStream

# perceive.py
def build_observations(agent, world):
    room = agent["current_location"]
    room_data = world["locations"][room]
    obs = [
        f"{agent['name']} is currently at {room}.",
        f"{agent['name']} is currently {agent['current_action'].lower()}.",
    ]

    if agent["interacting_with"] is not None:
        obs.append(
            f"{agent['name']} is interacting with the {agent['interacting_with']}."
        )

    for other in world["agents"].values():
        if other["id"] == agent["id"] or other["current_location"] != room:
            continue
        obs.append(f"{other['name']} is currently {other['current_action'].lower()}.")
        if other["interacting_with"] is not None:
            obs.append(
                f"{other['name']} is interacting with the {other['interacting_with']}."
            )

    obs.append(f"The {room} is currently {room_data['state']}.")
    for name, data in room_data["objects"].items():
        obs.append(f"The {name} in the {room} is {data['state']}.")

    return obs


def perceive_environment(agent, world, memory_stream, previous_observations):
    observations = build_observations(agent, world)

    previous = set(previous_observations)
    new_observations = [o for o in observations if o not in previous]

    if new_observations:
        memory_stream.add_observations(new_observations)

    return observations, new_observations