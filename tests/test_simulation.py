"""
End-to-end runs of the real step loop (main.main) against a scripted fake LLM:
planning, perceiving, deciding, conversations, acting and reflection together.
"""

import json
import re

import agent as ag
import llm
import log
import main
import world as world_module
from conftest import fake_generate, seed_module
from memory_stream import MemoryStream


def _run(monkeypatch, capsys, steps, threshold):
    monkeypatch.setattr(main, "NUM_STEPS", steps)
    monkeypatch.setattr(ag, "REFLECTION_THRESHOLD", threshold)
    log.timings.clear()
    main.main()
    return capsys.readouterr().out


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


TASK_ID = re.compile(r"\bT-\d+\b")


def test_simulation_runs_and_reflects_when_importance_adds_up(
    fake_llm, isolated_data, monkeypatch, capsys
):
    """Paper 4.2: reflection fires when the importance of what was perceived
    since the last reflection reaches the threshold (the fake LLM rates every
    memory 5), then the count starts over."""
    # The fake world is quiet (about 12 perceived events in 8 steps), so the
    # threshold is scaled down from the paper's 150.
    out = _run(monkeypatch, capsys, steps=8, threshold=30)

    # Readable step headers and one summary line per agent per step
    assert "STEP 1/8" in out and "STEP 8/8" in out
    assert out.count("ADAM ") >= 8 and out.count("EVE ") >= 8

    for tag, agent_id in (("ADAM", "adam"), ("EVE", "eve")):
        passes = out.count(f"{tag} reflecting")
        stream = MemoryStream(agent_id)
        perceived = [m for m in stream.memories if m.memory_type == "observation"]
        # Every 30 points of perceived importance (6 memories at 5) is one pass.
        assert 1 <= passes <= len(perceived) * 5 // 30
        reflections = [m for m in stream.memories if m.memory_type == "reflection"]
        assert len(reflections) == 2 * passes


def test_seeded_memories_dont_count_toward_reflection(fake_llm, isolated_data, monkeypatch, capsys):
    """The seeds alone are worth more than 150 (34 memories at 5), but they
    aren't events the agent perceived today."""
    seeds = seed_module()
    seeds.seed("adam", seeds.adam_seed_memories, reset=True)
    seeds.seed("eve", seeds.eve_seed_memories, reset=True)
    assert len(seeds.adam_seed_memories) * 5 >= 150
    out = _run(monkeypatch, capsys, steps=2, threshold=150)
    assert "reflecting" not in out


def test_no_reflection_when_turned_off(fake_llm, isolated_data, monkeypatch, capsys):
    out = _run(monkeypatch, capsys, steps=8, threshold=None)
    assert "reflecting" not in out


def test_world_state_is_saved_to_the_isolated_copy(fake_llm, isolated_data, monkeypatch, capsys):
    _run(monkeypatch, capsys, steps=3, threshold=None)
    saved = json.loads((isolated_data / "world" / "world.json").read_text())
    assert saved["agents"]["adam"]["current_action"] != "idle"


def test_reaction_becomes_conversation_commitments_and_a_meeting(
    isolated_data, monkeypatch, capsys
):
    scripted_generate.reacted = False
    monkeypatch.setattr(llm, "generate", scripted_generate)
    monkeypatch.setattr(llm, "embed", __import__("conftest").fake_embed)
    monkeypatch.setattr(main, "NUM_STEPS", 5)
    monkeypatch.setattr(ag, "REFLECTION_THRESHOLD", None)
    log.timings.clear()

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


def test_no_prompt_mentions_a_task_id_while_the_board_is_empty(isolated_data, monkeypatch):
    """Run the loop on an empty board (board visits change nothing) with one
    reaction that starts a conversation, and look at every prompt sent."""
    prompts = []
    state = {"reacted": False}

    def gen(prompt, options=None, format=None):
        prompts.append(prompt)
        if isinstance(format, dict) and "actions" in format["properties"]:
            return json.dumps({"actions": [], "reason": "nothing to change"})
        if "DECISION" in prompt and "Adam Beck's status" in prompt and not state["reacted"]:
            state["reacted"] = True
            return "DECISION: 2\nREACTION: Greet Eve Smith and start reviewing her tasks together"
        if "decided to react" in prompt:
            return "Morning Eve, what are you starting on today?"
        if "Here is the dialogue history" in prompt:
            return "END"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    monkeypatch.setattr(llm, "embed", __import__("conftest").fake_embed)
    monkeypatch.setattr(main, "NUM_STEPS", 6)
    monkeypatch.setattr(ag, "REFLECTION_THRESHOLD", None)
    log.timings.clear()
    main.main()

    assert any("is talking with" in p for p in prompts)          # a conversation happened
    offenders = [TASK_ID.findall(p) for p in prompts if TASK_ID.search(p)]
    assert offenders == []


def test_reset_clears_board_notes_and_puts_everyone_back_at_reception(isolated_data, monkeypatch):
    seed = seed_module()

    data = world_module.load_world()
    data["agents"]["adam"]["current_location"] = "office_1"
    data["project"]["notes"] = [{"time": "9:10 am", "agent": "adam", "note": "T-1 is complete"}]
    world_module.save_world(data)

    seed.clear_board()
    after = world_module.load_world()
    assert after["project"]["notes"] == [] and after["project"]["tasks"] == []
    assert {a["current_location"] for a in after["agents"].values()} == {"reception"}
