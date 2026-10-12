"""
End-to-end run of the real simulation loop against a fake LLM: planning,
perceiving, deciding, acting and reflection all execute for a few steps.
"""

import json

import main
from memory_stream import MemoryStream


def _run(monkeypatch, capsys, steps, threshold):
    monkeypatch.setattr(main, "NUM_STEPS", steps)
    monkeypatch.setattr(main, "REFLECTION_THRESHOLD", threshold)
    main.timings.clear()
    main.main()
    return capsys.readouterr().out


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


def _seed_module():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "scripts" / "seed_memories.py"
    spec = importlib.util.spec_from_file_location("seed_memories", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seeded_memories_dont_count_toward_reflection(fake_llm, isolated_data, monkeypatch, capsys):
    """The seeds alone are worth more than 150 (34 memories at 5), but they
    aren't events the agent perceived today."""
    seeds = _seed_module()
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
