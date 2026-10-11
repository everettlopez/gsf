"""perceive.py - turn the world state around an agent into observation strings."""

import project_board


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
        if name == project_board.BOARD_OBJECT:
            continue                      # read off the board itself, below
        obs.append(f"The {name} in the {room} is {data['state']}.")

    # Anyone in the room can read the project board (it's a real object, so
    # an agent only knows what's on it after standing in front of it).
    if project_board.BOARD_OBJECT in room_data["objects"]:
        obs.extend(project_board.board_observations(world))

    return obs


def perceive_environment(agent, world, memory_stream, previous_observations):
    observations = build_observations(agent, world)

    previous = set(previous_observations)
    new_observations = [o for o in observations if o not in previous]

    if new_observations:
        memory_stream.add_observations(new_observations)

    return observations, new_observations