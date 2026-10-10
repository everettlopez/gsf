"""
Forces Adam to react to Eve so the whole social path runs for real:
react -> conversation -> outcomes -> replan -> both agents head to the agreed room.
"""

import json
import re

import llm
import main
from conftest import fake_generate


def scripted_generate(prompt, options=None, format=None):
    # Check the replan prompt first: it also mentions "decided to react".
    if "Revise" in prompt:
        start = re.search(r"must start at (\d{1,2}:\d{2} [ap]m)", prompt).group(1)
        return f"1. sync with the team from {start} to 9:00 am"
    if "DECISION" in prompt:
        # Only Adam reacts, and only the first time he is asked.
        if "Adam Beck's status" in prompt and not scripted_generate.reacted:
            scripted_generate.reacted = True
            return "DECISION: 2\nREACTION: greet Eve and suggest syncing"
        return "DECISION: 1\nREACTION: none"

    if "decided to react" in prompt:
        return "Hi Eve, want to sync in the conference room?"
    if "Here is the dialogue history" in prompt:
        # Transcript lines start with the speaker; retrieved memories don't.
        eve_has_spoken = re.search(r"^\s*Eve Smith: ", prompt, re.MULTILINE)
        return "END" if eve_has_spoken else "SAY: Sounds good, see you there."

    if isinstance(format, dict) and "meeting_agreed" in format["properties"]:
        return json.dumps({
            "meeting_agreed": True, "location": "conference_room",
            "activity": "Sync on the project", "minutes": 30,
            "commitments": [
                {"person": "Adam Beck", "commitment": "Walk Eve through the API"},
                {"person": "Eve Smith", "commitment": "Bring the navigation designs"},
            ],
        })

    return fake_generate(prompt, options, format)


scripted_generate.reacted = False


def test_reaction_becomes_conversation_commitments_and_a_meeting(
    isolated_data, monkeypatch, capsys
):
    scripted_generate.reacted = False
    monkeypatch.setattr(llm, "generate", scripted_generate)
    monkeypatch.setattr(llm, "embed", __import__("conftest").fake_embed)
    monkeypatch.setattr(main, "NUM_STEPS", 5)
    monkeypatch.setattr(main, "REFLECTION_TIMES", [])
    main.timings.clear()

    main.main()
    out = capsys.readouterr().out

    assert "💬 ADAM ↔ EVE" in out
    assert 'ADAM : "Hi Eve, want to sync in the conference room?"' in out
    assert 'EVE  : "Sounds good, see you there."' in out
    assert "agreed to meet in conference_room" in out
    assert "ADAM committed: Walk Eve through the API" in out
    assert "EVE committed: Bring the navigation designs" in out
    assert "ADAM replanned the day around" in out

    # After the talk, the agreed meeting sends both agents to the conference room.
    final = json.loads((isolated_data / "world" / "world.json").read_text())
    assert final["agents"]["adam"]["current_location"] == "conference_room"
    assert final["agents"]["eve"]["current_location"] == "conference_room"
