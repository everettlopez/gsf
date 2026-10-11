"""A broad block whose hour decomposition comes back short must not leave the
agent replaying the last minute action of an hour that already ended."""

import llm
import main
from conftest import fake_generate


def short_hours(prompt, options=None, format=None):
    if "Create EXACTLY" in prompt:          # the model returns one hour, whatever we ask
        return "1. 10:05 am: Review progress and adjust task order"
    return fake_generate(prompt, options, format)


def test_gap_after_a_short_hour_decomposition_is_filled(
    fake_llm, isolated_data, world, monkeypatch
):
    monkeypatch.setattr(llm, "generate", short_hours)
    rt = main.make_runtime("adam", world)
    rt.broad_chunks = [{"activity": "work on T-1 'Set up the API' from 10:05 am to 12:00 pm",
                        "location": "office_1", "start_time": "10:05 am", "duration": 115,
                        "target_agent": None}]

    early = main.get_chunk_lazy(rt, "10:30 am")
    late = main.get_chunk_lazy(rt, "11:30 am")
    later = main.get_chunk_lazy(rt, "11:50 am")

    assert rt.planner.covers(early, "10:30 am")
    assert rt.planner.covers(late, "11:30 am")         # not a stale 10:xx action
    assert rt.planner.covers(later, "11:50 am")
    gap_hours = [k for k in rt.minute_cache if k[0] == "11:05 am"]
    assert len(gap_hours) == 1                          # the gap is expanded once
