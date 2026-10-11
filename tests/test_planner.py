import pytest

from plan import Planner


@pytest.fixture
def planner(fake_llm, isolated_data):
    agent = {"id": "adam", "name": "Adam Beck"}
    return Planner(agent)


@pytest.fixture
def adam(world):
    return world["agents"]["adam"]


def test_extract_start_time(planner):
    assert planner.extract_start_time("6:00 pm: start brainstorming") == "6:00 pm"
    assert planner.extract_start_time("work on x from 6:00 pm to 8:00 pm") == "6:00 pm"
    assert planner.extract_start_time("meeting at 6:30 pm") == "6:30 pm"
    assert planner.extract_start_time("just vibes") is None


def test_extract_duration(planner):
    assert planner.extract_duration("work from 9:00 am to 12:00 pm") == 180
    assert planner.extract_duration("lunch from 12:00 pm to 1:00 pm") == 60
    assert planner.extract_duration("clock in and start work") == 60
    assert planner.extract_duration("something undated") == 0


@pytest.mark.parametrize("activity,room", [
    ("clock in and start work at 7:00 am", "reception"),
    ("take a lunch break", "break room"),
    ("attend engineering meetings from 8:00 am to 9:00 am", "conference room"),
    ("work on memory-system development", "office_1"),
])
def test_keyword_location(planner, adam, activity, room):
    assert planner.resolve_location_keywords(activity, adam) == room


def test_current_chunk_picks_the_one_covering_now(planner):
    chunks = [
        {"activity": "a", "start_time": "7:00 am", "duration": 30},
        {"activity": "b", "start_time": "7:30 am", "duration": 30},
    ]
    assert planner.get_current_minute_chunk("7:10 am", chunks)["activity"] == "a"
    assert planner.get_current_minute_chunk("7:45 am", chunks)["activity"] == "b"
    assert planner.get_current_minute_chunk("bogus", chunks) is None


def test_split_plan_truncates_chunk_in_progress(planner):
    chunks = [
        {"activity": "a", "start_time": "7:00 am", "duration": 60},
        {"activity": "b", "start_time": "8:00 am", "duration": 60},
    ]
    done, remaining = planner.split_plan(chunks, "7:20 am")
    assert [c["activity"] for c in done] == ["a"] and done[0]["duration"] == 20
    assert [c["activity"] for c in remaining] == ["b"]


def test_fill_durations_uses_gaps_and_fills_the_hour(planner):
    actions = [{"start_time": "4:00 pm"}, {"start_time": "4:10 pm"}, {"start_time": "4:35 pm"}]
    planner._fill_durations(actions, hour_start="4:00 pm")
    assert [a["duration"] for a in actions] == [10, 25, 25]


def test_fill_durations_never_produces_nonpositive_values(planner):
    actions = [{"start_time": "4:30 pm"}, {"start_time": "4:10 pm"}, {"start_time": None}]
    planner._fill_durations(actions, hour_start="4:00 pm")
    assert all(a["duration"] > 0 for a in actions)


def test_extract_target_agent_matches_whole_words_only(planner, world, adam):
    assert planner.extract_target_agent("go talk to Eve", adam, world) == "eve"
    assert planner.extract_target_agent("enjoy the evening", adam, world) is None


def test_initial_plan_is_generated_for_any_agent(planner, world):
    plan = planner.generate_initial_plan(world["agents"]["eve"])
    assert len(plan) == 3 and "from 7:00 am to 9:00 am" in plan[0]


def test_hour_and_minute_decomposition(planner, adam):
    broad = planner.create_chunks(["work on development from 7:00 am to 9:00 am"], adam)
    hours = planner.decompose_chunks(broad, adam)
    assert [h["start_time"] for h in hours] == ["7:00 am", "8:00 am"]

    minutes = planner.decompose_hour_chunks([hours[0]], adam)
    assert minutes[0]["start_time"] == "7:00 am" and len(minutes) == 6
    assert sum(m["duration"] for m in minutes) == 60


def test_agent_summary_is_cached_and_invalidated(planner, adam, monkeypatch):
    import llm

    planner.memory_stream.add_observation("Adam is a backend engineer.")
    calls = []
    original = llm.generate
    monkeypatch.setattr(llm, "generate", lambda *a, **k: calls.append(1) or original(*a, **k))

    planner.generate_agent_summary(adam)
    planner.generate_agent_summary(adam)
    assert len(calls) == 1

    planner.invalidate_summary()
    planner.generate_agent_summary(adam)
    assert len(calls) == 2


def test_time_parsing_tolerates_missing_spaces(planner):
    activity = "sketch page layouts from9:00 am to12:00 pm"
    assert planner.extract_start_time(activity) == "9:00 am"
    assert planner.extract_duration(activity) == 180


def test_plan_gets_its_clock_in_back_when_the_model_skips_it(planner):
    items = ["check the project board from 8:00 am to 9:00 am"]
    assert planner.ensure_day_start(items) == [
        "clock in and start work at 7:00 am",
        "check the project board from 8:00 am to 9:00 am",
    ]


def test_plan_that_already_starts_the_day_is_left_alone(planner):
    covered = ["work on development from 7:00 am to 9:00 am"]
    clocked = ["Clock in at 7:00 am", "check the project board from 8:00 am to 9:00 am"]
    assert planner.ensure_day_start(covered) == covered
    assert planner.ensure_day_start(clocked) == clocked


def test_hour_decomposition_covers_blocks_that_are_not_whole_hours(planner, adam):
    block = {"activity": "work on T-1 'Set up the API' from 10:05 am to 12:00 pm",
             "start_time": "10:05 am", "duration": 115, "target_agent": None}
    hours = planner.decompose_chunks([block], adam)
    assert [(h["start_time"], h["duration"]) for h in hours] == [("10:05 am", 60), ("11:05 am", 55)]
    assert planner.covers(hours[1], "11:50 am")
    assert not planner.covers(hours[0], "11:50 am")
