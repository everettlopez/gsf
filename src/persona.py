"""
persona.py - who each agent is, from data/agents/<id>/persona.json.

The paper seeds each agent with a natural-language description: identity and
innate traits, occupation, lifestyle and relationships, and what they did the
day before (section 4.3: the previous day's summary starts the next day's
plan). Each of those is a field here, kept as the sentences the prompts use.

work_style and communication_style describe how the agent goes about work and
talk. They are read by the prompts that make decisions (see describe_style),
not only the ones that set the tone.
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
        previous_day(p),
    ]
    return _INDENT.join(paragraphs)
