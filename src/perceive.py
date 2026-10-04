from memory_stream import MemoryStream

def perceive_environment(agent, world):

    memory_stream = MemoryStream(agent["id"])

    observations = []

    room = agent["current_location"]

    room_data = (
        world["locations"][room]
    )

    # PERCEIVE AGENT SELF ACTIONS / LOCATION

    location_observation = (
        f"{agent['name']} is currently at {room}."
    )

    observations.append(
        location_observation
    )

    memory_stream.add_observation(
        location_observation
    )

    action_observation = (
        f"{agent["name"]} is currently {agent["current_action"].lower()}."
    )

    observations.append(action_observation)

    memory_stream.add_observation(
        action_observation
    )

    if agent["interacting_with"] is not None:
        interaction_observation = (
            f"{agent['name']} is interacting with the {agent['interacting_with']}"
        )

        observations.append(interaction_observation)

        memory_stream.add_observation(interaction_observation)

    # PERCIEVCE OTHER AGENTS
    for other_agent in world["agents"].values():
        if other_agent["id"] == agent["id"]:
            continue

        if (other_agent["current_location"] == room):

            observation = (
                f"{other_agent["name"]} is currently {other_agent["current_action"].lower()}."
            )

            observations.append(observation)

            memory_stream.add_observation(observation)

            if other_agent["interacting_with"] is not None:
                interaction_observation = (
                    f"{other_agent["name"]} is interacting with the {other_agent["interacting_with"]}"
                )

                observations.append(interaction_observation)

                memory_stream.add_observation(interaction_observation)

    # PERCEIVE ROOM
    room_observation = (
        f"The {room} is currently {room_data["state"]}."
    )

    observations.append(room_observation)

    memory_stream.add_observation(room_observation)

    for obj, data in room_data["objects"].items():

        state = data["state"]

        observation = (
            f"The {obj} in the {room} is {state}."
        )

        observations.append(
            observation
        )

        memory_stream.add_observation(
            observation
        )

    return observations