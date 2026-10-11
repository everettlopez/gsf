import json
import re
from pathlib import Path

from lemminflect import getInflection

import llm
import log
import project_board

WORLD_PATH = Path(__file__).resolve().parent.parent / "data" / "world" / "world.json"


def load_world():
    with open(WORLD_PATH, "r") as f:
        return json.load(f)


def save_world(world):
    with open(WORLD_PATH, "w") as f:
        json.dump(world, f, indent=2)


def load_agent(agent_id):
    world = load_world()
    return world["agents"][agent_id]


def save_agent(agent):
    world = load_world()
    world["agents"][agent["id"]] = agent
    save_world(world)


def update_agent_state(agent, current_chunk, world, location=None):
    activity = current_chunk["activity"]
    if ": " in activity:
        activity = activity.split(": ", 1)[1]
    activity = activity.strip().rstrip(".")

    target = current_chunk.get("target_agent")

    if location:                                   # resolver output (includes agreements)
        new_loc = location
    elif target and target in world["agents"]:     # "go find Eve"
        new_loc = world["agents"][target]["current_location"]
    elif current_chunk.get("location"):            # legacy chunk location
        new_loc = current_chunk["location"].lower().replace(" ", "_")
    else:
        new_loc = agent["current_location"]

    if new_loc not in world["locations"]:
        log.warn(f"unknown location {new_loc!r} for {agent['name']}, staying put")
        new_loc = agent["current_location"]

    if new_loc != agent["current_location"]:
        agent["interacting_with"] = None

    agent["current_location"] = new_loc
    agent["current_action"] = action_to_present(activity)

    world["agents"][agent["id"]] = agent           # keep world in sync (see #2)
    return agent


def refresh_room_states(world):
    """Deterministic room state: who is there, or 'empty'. No LLM needed."""
    for room_name, room_data in world["locations"].items():
        occupants = [
            a["name"] for a in world["agents"].values()
            if a["current_location"] == room_name
        ]
        room_data["state"] = (
            "occupied by " + " and ".join(occupants) if occupants else "empty"
        )
    project_board.sync_board_object(world)


def _parse_json(text):
    """Tolerant JSON parse (handles code fences / stray text)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def generate_interactions(agent, world, observations, planner, chunk=None):
    """
    One LLM call (was two): pick the object being used and any object state
    changes. Room state is computed without the LLM.

    Pass `chunk` to skip the LLM when the agent is talking to another agent.
    """
    room = agent["current_location"]
    room_data = world["locations"][room]
    room_objects = room_data["objects"]

    talking_to_agent = bool(chunk and chunk.get("target_agent"))
    activity = (chunk or {}).get("activity") or agent["current_action"]
    board_here = project_board.BOARD_OBJECT in room_objects

    # Only board work can pick the board; otherwise "design the homepage" in the
    # conference room ends up "using" the board and editing tasks at random.
    usable = {name: data for name, data in room_objects.items()
              if name != project_board.BOARD_OBJECT}

    if board_here and ((chunk or {}).get("board_work") or mentions_board(activity)):
        # Plainly board work (even "review the board with Eve"): no LLM needed.
        agent["interacting_with"] = project_board.BOARD_OBJECT
    elif talking_to_agent or not usable:
        # Conversations don't involve objects, and empty rooms have none.
        agent["interacting_with"] = None
    else:
        agent_summary = planner.generate_agent_summary(agent)  # cached
        observations_text = "\n".join(
            f"Observation: {o}" for o in observations
        )

        prompt = f"""
    {agent_summary}

    Current Action: {agent["current_action"]}

    Current Interaction: {agent["interacting_with"]}

    Current Room: {room}

    {observations_text}

    Available objects:
    {json.dumps(usable, indent=2)}

    Determine:
    1. Which object the agent is interacting with.
    2. Any object state changes caused by the action.

    Rules:
    - The object MUST exist in the room.
    - Prefer continuing the current interaction if it still fits.
    - Return null if no object interaction is needed.
    - Most actions should NOT change object states; return an empty list then.

    Return ONLY JSON.

    Example:
    {{
        "interacting_with": "computer",
        "changes": [
            {{"object": "computer", "new_state": "running testing tools"}}
        ]
    }}

    Example:
    {{
        "interacting_with": null,
        "changes": []
    }}
    """

        result = _parse_json(llm.generate(prompt, format="json"))

        choice = result.get("interacting_with")
        agent["interacting_with"] = choice if choice in usable else None

        for change in result.get("changes") or []:
            obj = change.get("object")
            # (the board only changes through project_board actions)
            if obj in usable and change.get("new_state"):
                room_objects[obj]["state"] = change["new_state"]

    if agent["interacting_with"] == project_board.BOARD_OBJECT:
        # Using the board means reading it and (maybe) editing tasks.
        project_board.interact(agent, world, activity, planner=planner)

    refresh_room_states(world)
    log.debug(f"{room}: {room_data['state']}; {agent['name']} using {agent['interacting_with']}")

    return agent, world


_BOARD_WORDS = re.compile(
    r"\b(?:project board|task board|board|tasks?|backlog)\b", re.IGNORECASE
)


def mentions_board(activity):
    """Does this activity plainly involve the project board?"""
    return bool(_BOARD_WORDS.search(activity or "")
                or project_board.mentions_board_explicitly(activity))


def action_to_present(action):
    words = action.split()
    if not words:
        return action

    verb = words[0].lower()
    inflection = getInflection(verb, tag="VBG")
    if inflection:
        words[0] = inflection[0]

    return " ".join(words)