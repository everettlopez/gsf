"""
End-to-end run of the real simulation loop against a fake LLM: planning,
perceiving, deciding, acting and reflection all execute for a few steps.
"""

import json

import main
from memory_stream import MemoryStream


def _run(monkeypatch, capsys, steps, reflection_times):
    monkeypatch.setattr(main, "NUM_STEPS", steps)
    monkeypatch.setattr(main, "REFLECTION_TIMES", reflection_times)
    main.timings.clear()
    main.main()
    return capsys.readouterr().out


def test_simulation_runs_and_reflects_once_per_scheduled_time(
    fake_llm, isolated_data, monkeypatch, capsys
):
    out = _run(monkeypatch, capsys, steps=6, reflection_times=["7:15 am"])

    # Readable step headers and one summary line per agent per step
    assert "STEP 1/6" in out and "STEP 6/6" in out
    assert out.count("ADAM ") >= 6 and out.count("EVE ") >= 6

    # Fires once (at the first step at/after 7:15), not on every later step
    assert out.count("ADAM reflecting") == 1
    assert out.count("EVE reflecting") == 1

    for agent_id in ("adam", "eve"):
        stream = MemoryStream(agent_id)
        reflections = [m for m in stream.memories if m.memory_type == "reflection"]
        assert len(reflections) == 2
        assert any(m.memory_type == "observation" for m in stream.memories)


def test_reflection_times_before_start_never_fire(fake_llm, isolated_data, monkeypatch, capsys):
    out = _run(monkeypatch, capsys, steps=3, reflection_times=["6:00 am"])
    assert "reflecting" not in out


def test_world_state_is_saved_to_the_isolated_copy(fake_llm, isolated_data, monkeypatch, capsys):
    _run(monkeypatch, capsys, steps=3, reflection_times=[])
    saved = json.loads((isolated_data / "world" / "world.json").read_text())
    assert saved["agents"]["adam"]["current_action"] != "idle"
