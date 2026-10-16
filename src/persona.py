"""
persona.py - who each agent is, from data/agents/<id>/persona.json.

The paper seeds each agent with a natural-language description: identity and
innate traits, occupation, lifestyle and relationships, and what they did the
day before (section 4.3: the previous day's summary starts the next day's
plan). Each of those is a field here, kept as the sentences the prompts use.

work_style and communication_style describe how the agent goes about work and
talk. They are part of the profile, and style() also puts them into the
prompts that make decisions (react or not, board actions, what to say), not
only the ones that set the tone.
"""

import json
from functools import lru_cache
from pathlib import Path

PERSONA_DIR = Path(__file__).resolve().parent.parent / "data" / "agents"

# The profile is pasted into prompts at this indentation.
_INDENT = "\n\n        "


@lru_cache(maxsize=None)
def load(agent_id):
    with open(PERSONA_DIR / agent_id / "persona.json", encoding="utf-8") as f:
        return json.load(f)


def previous_day(persona):
    """'On Wednesday, Adam:\n 1) ..., 2) ... .' as the paper's prompt reads."""
    first = persona["name"].split()[0]
    day = persona["previous_day"]
    items = ",\n        ".join(f"{i}) {a}" for i, a in enumerate(day["activities"], 1))
    return f"On {day['day']}, {first}:\n        {items}."


def profile(agent_id):
    """The agent's description for planning prompts."""
    p = load(agent_id)
    paragraphs = [
        f"Name: {p['name']} (age: {p['age']})",
        f"Innate traits: {', '.join(p['innate_traits'])}",
        *p["occupation"], *p["lifestyle"], *p["relationships"], *p["aspirations"],
        " ".join(p.get("work_style", [])), " ".join(p.get("communication_style", [])),
        previous_day(p),
    ]
    return _INDENT.join(x for x in paragraphs if x)


def style(agent_id, indent="        "):
    """How the agent works and talks, for the prompts that decide what they do:
    react or carry on, what to do at the board, what to say. The summary those
    prompts start with comes from memories and says little about habits, so
    Adam and Eve made the same choices in different words."""
    try:
        p = load(agent_id)
    except FileNotFoundError:
        return ""
    name = p["name"].split()[0]
    lines = []
    if p.get("work_style"):
        lines.append(f"How {name} works: {' '.join(p['work_style'])}")
    if p.get("communication_style"):
        lines.append(f"How {name} communicates: {' '.join(p['communication_style'])}")
    if not lines:
        return ""
    lines.append(f"{name} decides the way this person would.")
    return "\n" + "\n".join(indent + line for line in lines) + "\n"
