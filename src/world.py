import json
import re
from lemminflect import getInflection
import ollama 

from plan import Planner

def load_world():
    with open("data/world/world.json", "r") as f:
        return json.load(f)

def save_world(world):
    with open(
        "data/world/world.json",
        "w"
    ) as f:
        json.dump(
            world,
            f,
            indent=2
        )

def load_agent(agent_id):
    world = load_world()
    return world["agents"][agent_id]

def save_agent(agent):
    world = load_world()
    world["agents"][agent["id"]] = agent
    
    save_world(world)

def update_agent_state(agent, current_chunk):

    print("Updating Agent Info")

    activity = current_chunk["activity"]

    if ": " in activity:
        activity = activity.split(": ", 1)[1]

    activity = activity.strip().rstrip(".")

    agent["current_action"] = action_to_present(activity)

    agent["current_location"] = (
        current_chunk["location"].lower().replace(" ", "_")
    )

    return agent

def generate_interactions(agent, world, observations):
    planner = Planner(agent)

    agent_summary = planner.generate_agent_summary(agent)

    observations_text = "\n".join(
        f"Observation: {observation}" for observation in observations
    )

    room = agent["current_location"]

    room_data = (world["locations"][room])

    room_objects = (room_data["objects"])

    print("ROOM: ", room)
    print("ROOM OBJECTS: ", room_objects)
    print("CURRENT ACTION: ", agent["current_action"])

    object_prompt = f"""
    {agent_summary}

    Current Action: {agent["current_action"]}

    Current Interaction: {agent["interacting_with"]}

    {observations_text}

    Available objects: 
    {json.dumps(room_objects, indent=2)}

    Determine which object the agent is interacting with.

    Rules:

    - The object MUST exist in the room.
    - Prefer continuing the current interaction.
    - Select the most relevant object for the action.
    - Return null if no object interaction is needed.
    - Only include state changes when necessary.

    Return ONLY JSON.

    Example:

    {{
        "interacting_with": "computer"
    }}

    Example:

    {{
        "interacting_with": null
    }}
    """

    response = ollama.generate(
        model="qwen3",
        prompt=object_prompt,
        think=False
    )

    object_result = json.loads(response["response"])

    agent["interacting_with"] = (object_result["interacting_with"])

    room_state_prompt = f"""
    {agent_summary}
    
    Current Action: {agent["current_action"]}

    Current Interaction: {agent["interacting_with"]}
    
    Current Room: {room}

    Current Room State: {room_data["state"]}

    Current Objects: 
    {json.dumps(room_objects, indent=2)}

    Observations: 
    {observations_text}

    Determine:

    1. The overall room state.
    2. Any object state changes.
    
    Rules:
    - The room state should describe what an observer would notice.
    - Most actions should NOT change object states.
    - Only modify objects that exist.
    - Preserve realism.
    - If nothing changes return an empty list.
    
    Return ONLY JSON.
    
    Example:
    {{
        "room_state": "active software testing",
        "changes": [
            {{
                "object": "computer",
                "new_state": "running testing tools"
            }}
        ]
    }}

    Example:
    {{
        "room_state": "empty",
        "changes": []
    }}

    """

    response = ollama.generate(
        model="qwen3",
        prompt=room_state_prompt,
        think=False)

    room_result = json.loads(response["response"])

    room_data["state"] = (room_result["room_state"])

    for change in room_result["changes"]:
        obj = change["object"]

        if obj in room_objects:
            room_objects[obj]["state"] = (change["new_state"])

    print("NEW ROOM STATE: ", room_data["state"])

    return agent, world

def action_to_present(action):
    words = action.split()
    if not words:
        return action
    
    verb = words[0].lower()

    inflection = getInflection(verb, tag="VBG")

    if inflection:
        words[0] = inflection[0]

    return " ".join(words)