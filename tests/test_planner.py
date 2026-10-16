"""
plan.py: the Planner. Parsing the model's plans, decomposition, the cleanup
every plan gets (clock-in, board visits, checkpoints, teammates, task ids,
own lanes), where each activity happens, and the planning prompts.
"""

import json
import re

import pytest

import agent as ag
import llm
import project_board as pb
from conftest import board_api_and_pages, board_api_pages_and_db, board_pages_wait_on_api, fake_generate
from memory_stream import MemoryStream
from plan import Planner


@pytest.fixture
def planner(fake_llm, isolated_data):
    agent = {"id": "adam", "name": "Adam Beck"}
    return Planner(agent)


@pytest.fixture
def adam(world):
    return world["agents"]["adam"]


def _plan(p, lines, agent, w):
    return p.create_chunks(lines, agent, w)


EVE_INITIAL = [   # a model-written day: clock-in with no end, long board blocks
    "clock in and start work at 7:00 am",
    "attend the daily stand-up with the whole team from 8:00 am to 8:15 am",
    "check the project board in the conference room from 8:15 am to 9:00 am",
    "review and update frontend tasks on the project board from 9:00 am to 10:00 am",
    "collaborate with Adam Beck on technical challenges from 10:00 am to 11:00 am",
    "work on implementing a new page layout for the company website from 11:00 am to 12:00 pm",
]


ADAM_INITIAL = [   # the first board check written as two blocks
    "clock in at reception from 7:00 am to 7:05 am",
    "check the project board in the conference room from 7:05 am to 7:20 am",
    "review any new backend tasks that need to be added to the project board from 7:20 am to 7:35 am",
    "collaborate with Eve Smith on technical challenges related to the website from 7:40 am to 8:30 am",
]


EVE_910 = [   # a replan with a board visit around every block, before checkpoints
    "update the project board in the conference room from 9:10 am to 9:20 am",
    "work on T-2 'Design and implement homepage layout' from 9:20 am to 9:30 am",
    "update the project board in the conference room from 9:30 am to 9:40 am",
    "work on accessibility improvements for the homepage from 9:40 am to 11:00 am",
    "update the project board in the conference room from 11:00 am to 11:10 am",
    "work on accessibility improvements for the homepage from 11:10 am to 12:00 pm",
    "take a short break from 12:00 pm to 12:15 pm",
    "review and approve a teammate's task on the project board from 12:15 pm to 1:00 pm",
    "start T-3 'Design and implement about page layout' from 1:00 pm to 1:10 pm",
    "update the project board in the conference room from 1:10 pm to 1:20 pm",
    "work on T-3 'Design and implement about page layout' from 1:20 pm to 2:10 pm",
    "report progress on T-3 on the project board from 2:10 pm to 2:20 pm",
    "update the project board in the conference room from 2:20 pm to 2:30 pm",
    "work on T-4 'Implement contact form with validation' from 2:30 pm to 4:00 pm",
    "update the project board in the conference room from 4:00 pm to 4:10 pm",
    "report progress on T-4 on the project board from 4:10 pm to 4:20 pm",
    "update the project board in the conference room from 4:20 pm to 4:30 pm",
    "review documentation and test the frontend build from 4:30 pm to 4:50 pm",
    "update the project board in the conference room from 4:50 pm to 5:00 pm",
    "update the project board in the conference room from 5:00 pm to 5:00 pm",
]


def _work_block(activity, start, minutes, **extra):
    return {"activity": activity, "location": "office_1", "start_time": start,
            "duration": minutes, "target_agent": None, "board_work": False, **extra}


def _adam(w):
    return w["agents"]["adam"]


def _capture(prompts, key):
    def gen(prompt, options=None, format=None):
        if key in prompt:
            prompts.append(prompt)
        return fake_generate(prompt, options, format)
    return gen


def _runtime(world, monkeypatch, prompts):
    def capture(prompt, options=None, format=None):
        if "Revise" in prompt:
            prompts.append(prompt)
            return ("1. work on T-1 'Set up the API' in the office from 9:05 am to 11:00 am\n"
                    "2. update T-1 on the project board from 11:00 am to 11:15 am\n"
                    "3. clock out at 5:00 pm")
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", capture)
    rt = ag.Agent.load("adam", world)
    ag.initial_planning(rt, world)
    return rt


TASK_ID = re.compile(r"\bT-\d+\b")


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


def test_time_parsing_tolerates_missing_spaces(planner):
    activity = "sketch page layouts from9:00 am to12:00 pm"
    assert planner.extract_start_time(activity) == "9:00 am"
    assert planner.extract_duration(activity) == 180


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


def test_plan_gets_its_clock_in_back_when_the_model_skips_it(planner):
    items = ["check the project board from 8:00 am to 9:00 am"]
    assert planner.ensure_day_start(items) == [
        "clock in at reception from 7:00 am to 7:05 am",
        "check the project board from 8:00 am to 9:00 am",
    ]


def test_plan_that_already_starts_the_day_is_left_alone(planner):
    covered = ["work on development from 7:00 am to 9:00 am"]
    clocked = ["Clock in at 7:00 am", "check the project board from 8:00 am to 9:00 am"]
    assert planner.ensure_day_start(covered) == covered
    assert planner.ensure_day_start(clocked) == clocked


def test_hour_and_minute_decomposition(planner, adam):
    broad = planner.create_chunks(["work on development from 7:00 am to 9:00 am"], adam)
    hours = planner.decompose_chunks(broad, adam)
    assert [h["start_time"] for h in hours] == ["7:00 am", "8:00 am"]

    minutes = planner.decompose_hour_chunks([hours[0]], adam)
    assert minutes[0]["start_time"] == "7:00 am" and len(minutes) == 6
    assert sum(m["duration"] for m in minutes) == 60


def test_hour_decomposition_covers_blocks_that_are_not_whole_hours(planner, adam):
    block = {"activity": "work on T-1 'Set up the API' from 10:05 am to 12:00 pm",
             "start_time": "10:05 am", "duration": 115, "target_agent": None}
    hours = planner.decompose_chunks([block], adam)
    assert [(h["start_time"], h["duration"]) for h in hours] == [("10:05 am", 60), ("11:05 am", 55)]
    assert planner.covers(hours[1], "11:50 am")
    assert not planner.covers(hours[0], "11:50 am")


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


def test_breaks_and_board_visits_get_no_task_focus(fake_llm, isolated_data, w):
    board_api_pages_and_db(w)
    p = Planner(w["agents"]["eve"])
    focus = p.work_focus(w["agents"]["eve"], w)
    assert p._focus_text({"activity": "take a lunch break in the break room"}, focus) == ""
    assert p._focus_text({"activity": "check the board", "board_work": True}, focus) == ""
    assert "T-2" in p._focus_text({"activity": "work on the homepage"}, focus)


def test_board_blocks_keep_their_board_steps(fake_llm, isolated_data, w):
    p = Planner(w["agents"]["eve"])
    steps = [{"activity": "8:00 am: Update T-2 on the project board", "start_time": "8:00 am"}]
    assert p._drop_board_detours(steps, {"board_work": True}) == steps


def test_board_work_is_inherited_by_finer_chunks(fake_llm, isolated_data, w):
    planner = Planner(w["agents"]["adam"])
    parent = {"activity": "update the project board from 7:10 am to 8:00 am",
              "target_agent": None, "board_work": True}
    child = planner._chunk_from_activity("7:20 am: check all frontend tasks for updates",
                                         w["agents"]["adam"], parent)
    assert child["board_work"] is True


def test_clock_in_without_an_end_time_takes_five_minutes(w):
    eve = w["agents"]["eve"]
    chunks = _plan(Planner(eve), EVE_INITIAL, eve, w)
    assert chunks[0]["duration"] == 5                     # was 60: the whole first hour


def test_board_blocks_are_capped_at_fifteen_minutes(w):
    eve = w["agents"]["eve"]
    p = Planner(eve)
    chunks = p.cap_board_blocks(_plan(p, EVE_INITIAL, eve, w))
    board = [(c["activity"], c["duration"]) for c in chunks if c.get("board_work")]
    assert board == [
        ("check the project board in the conference room from 8:15 am to 8:30 am", 15),
        ("review and update frontend tasks on the project board from 9:00 am to 9:15 am", 15)]


def test_clock_in_is_not_stretched_and_the_board_check_moves_up(w):
    adam = w["agents"]["adam"]
    p = Planner(adam)
    chunks = p.thin_board_visits(p.cap_board_blocks(p.create_chunks(ADAM_INITIAL, adam, w)))
    assert [(c["activity"], c["duration"]) for c in chunks[:2]] == [
        ("clock in at reception from 7:00 am to 7:05 am", 5),
        ("review any new backend tasks that need to be added to the project board "
         "from 7:05 am to 7:20 am", 15)]


def test_a_dropped_visit_does_not_stretch_clock_out(w):
    adam = w["agents"]["adam"]
    p = Planner(adam)
    chunks = p.thin_board_visits(p.create_chunks([
        "update the project board from 4:00 pm to 4:10 pm",
        "work on the API from 4:10 pm to 4:30 pm",
        "update the project board from 4:30 pm to 4:40 pm",
        "clock out at reception from 4:40 pm to 4:45 pm",
        "update the project board from 4:45 pm to 4:50 pm",
    ], adam, w))
    clock_out = next(c for c in chunks if "clock out" in c["activity"])
    assert clock_out["duration"] == 5


def test_thinning_keeps_only_board_visits_with_a_purpose(w):
    board_pages_wait_on_api(w)
    eve = w["agents"]["eve"]
    p = Planner(eve)
    chunks = p.thin_board_visits(_plan(p, EVE_910, eve, w))
    board = [c["activity"].split(" from ")[0] for c in chunks if c.get("board_work")]
    assert board == [
        "update the project board in the conference room",              # 9:10, first of the day
        "update the project board in the conference room",              # 11:00, 100 min later
        "review and approve a teammate's task on the project board",    # has a purpose
        "start T-3 'Design and implement about page layout'",
        "report progress on T-3 on the project board",
        "report progress on T-4 on the project board",
        "update the project board in the conference room",              # last of the day
    ]
    assert all((c.get("duration") or 0) > 0 for c in chunks)            # 5:00-5:00 is gone


def test_dropped_visit_time_goes_to_the_work_before_it_not_to_a_break(w):
    eve = w["agents"]["eve"]
    p = Planner(eve)
    chunks = p.thin_board_visits(_plan(p, [
        "update the project board in the conference room from 9:00 am to 9:10 am",
        "work on the navigation bar from 9:10 am to 10:00 am",
        "update the project board in the conference room from 10:00 am to 10:10 am",
        "take a short break from 10:10 am to 10:20 am",
        "update the project board in the conference room from 10:20 am to 10:30 am",
        "work on the footer from 10:30 am to 5:00 pm",
        "update the project board before clocking out from 4:50 pm to 5:00 pm",
    ], eve, w))
    acts = [c["activity"] for c in chunks]
    assert "work on the navigation bar from 9:10 am to 10:10 am" in acts
    assert "take a short break from 10:10 am to 10:20 am" in acts      # not stretched
    assert sum(c.get("board_work", False) for c in chunks) == 2          # first and last


def test_the_same_step_for_the_same_task_twice_in_a_row_is_dropped(w):
    board_pages_wait_on_api(w)
    eve = w["agents"]["eve"]
    p = Planner(eve)
    chunks = p.thin_board_visits(_plan(p, [
        "start T-3 'About page' from 1:00 pm to 1:10 pm",
        "start T-3 (move it to in progress) on the project board from 1:10 pm to 1:20 pm",
        "work on T-3 'About page' from 1:20 pm to 2:10 pm",
        "report progress on T-3 on the project board from 2:10 pm to 2:20 pm",
    ], eve, w))
    assert [c["activity"].split(" ")[0] for c in chunks] == ["start", "work", "report"]


def test_checkpoints_wrap_work_on_a_todo_task(fake_llm, isolated_data, w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    planner = Planner(w["agents"]["adam"])
    plan = [_work_block("work on T-1 'Set up the API' from 9:00 am to 11:00 am", "9:00 am", 120),
            _work_block("lunch in the break room from 12:00 pm to 1:00 pm", "12:00 pm", 60)]

    out = planner.add_task_checkpoints(plan, w["agents"]["adam"], pb.open_tasks_for(w, "adam"))

    assert [(c["start_time"], c["duration"], c["board_work"]) for c in out] == [
        ("9:00 am", 10, True), ("9:10 am", 100, False), ("10:50 am", 10, True),
        ("12:00 pm", 60, False)]
    assert out[0]["activity"] == ("start T-1 (move it to in progress) on the project board "
                                  "from 9:00 am to 9:10 am")
    assert out[1]["activity"] == "work on T-1 'Set up the API' from 9:10 am to 10:50 am"
    assert out[2]["activity"].startswith("report progress on T-1 on the project board")


def test_task_already_in_progress_only_gets_the_wrap_up(fake_llm, isolated_data, w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    planner = Planner(w["agents"]["adam"])
    plan = [_work_block("work on T-1 from 9:00 am to 10:00 am", "9:00 am", 60),
            _work_block("work on T-1 from 1:00 pm to 3:00 pm", "1:00 pm", 120)]

    out = planner.add_task_checkpoints(plan, w["agents"]["adam"], pb.open_tasks_for(w, "adam"))

    assert [(c["start_time"], c["board_work"]) for c in out] == [
        ("9:00 am", False), ("1:00 pm", False), ("2:50 pm", True)]   # wrap-up after the last block


def test_existing_board_visit_and_other_peoples_tasks_are_left_alone(fake_llm, isolated_data, w):
    pb.create_task(w, "adam", "API", assign_to_self=True)
    pb.create_task(w, "eve", "Home page", area="frontend", assign_to_self=True)
    planner = Planner(w["agents"]["adam"])
    plan = [_work_block("move T-1 to in progress on the project board from 8:50 am to 9:00 am",
                   "8:50 am", 10, board_work=True),
            _work_block("work on T-1 and help with T-2 from 9:00 am to 9:20 am", "9:00 am", 20)]

    out = planner.add_task_checkpoints(plan, w["agents"]["adam"], pb.open_tasks_for(w, "adam"))
    assert out == plan            # T-1 already visited; 20 min is too short to split; T-2 isn't Adam's


def test_checkpoint_starts_only_one_task(fake_llm, isolated_data, w):
    pb.create_task(w, "adam", "API", assign_to_self=True)
    pb.create_task(w, "adam", "Deploy", assign_to_self=True)
    plan = [{"activity": "work on T-1 and T-2 from 9:00 am to 11:00 am", "location": "office_1",
             "start_time": "9:00 am", "duration": 120, "target_agent": None, "board_work": False}]
    out = Planner(w["agents"]["adam"]).add_task_checkpoints(
        plan, w["agents"]["adam"], pb.open_tasks_for(w, "adam"))
    assert out[0]["activity"].startswith("start T-1 (move it to in progress)")
    assert out[-1]["activity"].startswith("report progress on T-1, T-2")


@pytest.mark.parametrize("activity,target", [
    ("work on T-1 in the office with Eve Smith from 8:10 am to 11:40 am", None),
    ("collaborate with Eve Smith on frontend-backend integration from 10:00 am to 11:00 am", "eve"),
    ("collaborate with Eve Smith on integration from 9:00 am to 12:00 pm", "eve"),
    ("review T-1 with Eve Smith from 7:35 am to 8:00 am", "eve"),    # short: kept
])
def test_long_solo_blocks_do_not_follow_a_teammate(w, activity, target):
    adam = w["agents"]["adam"]
    assert Planner(adam).create_chunks([activity], adam, w)[0]["target_agent"] == target


@pytest.mark.parametrize("step,target", [
    ("9:00 am: Discuss integration requirements", "eve"),
    ("9:40 am: Debug integration issues with Eve", "eve"),
    ("8:10 am: Review authentication requirements with Eve Smith", "eve"),
    ("9:10 am: Set up the development environment", None),
    ("9:15 am: Review backend requirements", None),
    ("8:30 am: Summarize key points from the discussion", None),     # a noun isn't working together
    ("8:20 am: Synchronize the data models", None),
])
def test_finer_steps_keep_the_teammate_only_when_working_with_them(w, step, target):
    adam = w["agents"]["adam"]
    parent = {"activity": "collaborate with Eve Smith on frontend-backend integration",
              "target_agent": "eve"}
    assert Planner(adam)._chunk_from_activity(step, adam, parent)["target_agent"] == target


def test_a_teammate_block_the_replan_kept_is_not_doubled(fake_llm, isolated_data, w):
    eve = w["agents"]["eve"]
    p = Planner(eve)
    kept = p.create_chunks(["sync with Adam Beck from 10:15 am to 11:00 am"], eve, w)
    new = p.create_chunks(["review the API with Adam Beck from 10:00 am to 10:30 am",
                           "work on the about page from 10:30 am to 1:00 pm"], eve, w)
    assert p.keep_blocks(new, kept) == new


@pytest.mark.parametrize("block,expected", [
    # The wrong id: the work described stays, the id goes.
    ("work on T-2 'Implement and secure user authentication endpoints' from 8:10 am to 9:30 am",
     "work on 'Implement and secure user authentication endpoints' from 8:10 am to 9:30 am"),
    # Eve's task as Adam's own work: Adam's own work instead, same time.
    ("work on T-2 'Design and implement the home page layout' from 8:10 am to 9:30 am",
     "keep working on T-1 'Design and implement the backend API for the company website' "
     "from 8:10 am to 9:30 am"),
    # Board steps for Eve's task: only she moves it.
    ("start T-2 (move it to in progress) on the project board from 8:00 am to 8:10 am", None),
    ("update the project board with progress on T-2 from 9:30 am to 9:45 am", None),
    # Own and teammate's: the teammate's id goes.
    ("report progress on T-1 and T-2 on the project board from 9:30 am to 9:40 am",
     "report progress on T-1 on the project board from 9:30 am to 9:40 am"),
    # Allowed: pairing with the owner, reviewing.
    ("pair with Eve Smith on T-2 from 10:00 am to 11:00 am",
     "pair with Eve Smith on T-2 from 10:00 am to 11:00 am"),
    ("help Eve with T-2 from 10:00 am to 11:00 am", "help Eve with T-2 from 10:00 am to 11:00 am"),
    ("review and approve T-2 on the project board from 3:00 pm to 3:15 pm",
     "review and approve T-2 on the project board from 3:00 pm to 3:15 pm"),
    # Naming Eve isn't enough: working on her task beside her is still doing it alone.
    ("work on T-2 in the office near Eve from 10:00 am to 11:00 am",
     "keep working on T-1 'Design and implement the backend API for the company website' "
     "from 10:00 am to 11:00 am"),
])
def test_plans_only_work_on_own_tasks(w, block, expected):
    board_api_and_pages(w)
    adam = w["agents"]["adam"]
    assert Planner(adam).own_tasks_only(block, adam, w) == expected


def test_the_owner_plans_her_own_task_as_usual(w):
    board_api_and_pages(w)
    eve = w["agents"]["eve"]
    block = "work on T-2 'Design and implement the home page layout' from 8:10 am to 9:30 am"
    assert Planner(eve).own_tasks_only(block, eve, w) == block


def test_a_replan_loses_blocks_on_a_teammates_task(w):
    """The 7:25 replan from the log, through create_chunks: no T-2 left in it."""
    board_api_and_pages(w)
    adam = w["agents"]["adam"]
    chunks = Planner(adam).create_chunks([
        "review and update any pending backend tasks from 7:25 am to 8:00 am",
        "start T-2 (move it to in progress) on the project board from 8:00 am to 8:10 am",
        "work on T-2 'Implement and secure user authentication endpoints' from 8:10 am to 9:30 am",
        "update the project board with progress on T-2 from 9:30 am to 9:45 am",
        "review documentation and finalize deployment configuration from 9:45 am to 12:00 pm",
    ], adam, w)
    assert not any("T-2" in c["activity"] for c in chunks)
    assert [c["start_time"] for c in chunks] == ["7:25 am", "8:10 am", "9:45 am"]


@pytest.mark.parametrize("activity,expected", [
    ("work on T-5 'Implement API endpoints for user authentication' from 10:15 am to 12:15 pm",
     "work on 'Implement API endpoints for user authentication' from 10:15 am to 12:15 pm"),
    ("review T-1, T-6 and T-7 with Eve", "review T-1 with Eve"),
    ("work on T-1 'Set up backend API' from 7:30 am to 8:50 am",
     "work on T-1 'Set up backend API' from 7:30 am to 8:50 am"),
])
def test_made_up_task_ids_are_removed_from_plans(w, activity, expected):
    board_pages_wait_on_api(w)
    assert Planner.drop_unknown_task_ids(activity, w) == expected
    chunk = Planner(w["agents"]["adam"]).create_chunks([activity], w["agents"]["adam"], w)[0]
    assert chunk["activity"] == expected


def test_a_wrong_title_in_a_plan_drops_the_id(w):
    """The id is what's wrong, not the title: putting T-1's title there would
    point the agent at different work than the plan described."""
    board_api_pages_and_db(w)
    eve = w["agents"]["eve"]
    chunk = Planner(eve).create_chunks(
        ["review and approve T-1 'Backend data structure' with Adam Beck from 11:30 am to 11:45 am"],
        eve, w)[0]
    assert chunk["activity"] == ("review and approve 'Backend data structure' with Adam Beck "
                                 "from 11:30 am to 11:45 am")


@pytest.mark.parametrize("activity,room", [
    ("clock in and start work at 7:00 am", "reception"),
    ("take a lunch break", "break room"),
    ("attend engineering meetings from 8:00 am to 9:00 am", "conference room"),
    ("work on memory-system development", "office_1"),
])
def test_keyword_location(planner, adam, activity, room):
    assert planner.resolve_location_keywords(activity, adam) == room


def test_board_activity_goes_to_the_conference_room(fake_llm, isolated_data, w):
    planner = Planner(_adam(w))
    assert planner.resolve_location_keywords("check the project board", _adam(w)) == \
        "conference room"


@pytest.mark.parametrize("activity,where,expected", [
    ("Outline database models", "office_1", "office_1"),           # solo work, own office
    ("Plan the API structure", "conference_room", "office_1"),      # solo work, go back
    ("take a short break", "break_room", "break_room"),             # already in the right room
    ("take a short break", "office_1", None),                       # elsewhere: let the LLM decide
    ("discuss the API with Eve", "office_1", None),                 # involves a teammate
])
def test_obvious_locations(w, activity, where, expected):
    adam = w["agents"]["adam"]
    adam["current_location"] = where
    assert Planner(adam).obvious_location(activity, adam, w) == expected


@pytest.mark.parametrize("activity,expected", [
    ("set up the development environment", "office_1"),
    ("pair with Eve on the navigation bar", "office_2"),  # about Eve: her office is fine
])
def test_choose_location_keeps_solo_work_out_of_a_teammates_office(
        fake_llm, isolated_data, w, monkeypatch, activity, expected):
    def gen(prompt, options=None, format=None):
        if isinstance(format, dict) and "location" in format["properties"]:
            return json.dumps({"reason": "Eve is there", "location": "office_2"})
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam = w["agents"]["adam"]
    assert Planner(adam).choose_location(activity, adam, w) == expected


def test_the_plan_example_is_a_short_board_check(fake_llm, isolated_data, w, monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if "Continue the plan as a numbered list" in prompt:
            prompts.append(prompt)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    eve = w["agents"]["eve"]
    Planner(eve).generate_initial_plan(eve, w)
    assert "1. clock in at reception from 7:00 am to 7:05 am" in prompts[0]
    assert "check the project board in the conference room from 7:05 am to 7:20 am" in prompts[0]


def test_plans_know_who_the_team_is(fake_llm, isolated_data, w, monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if "Continue the plan as a numbered list" in prompt:
            prompts.append(" ".join(prompt.split()))
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    eve = w["agents"]["eve"]
    Planner(eve).generate_initial_plan(eve, w)
    assert "Eve Smith's only teammates are Adam Beck (backend); there is no one else" in prompts[0]
    assert "designers" not in prompts[0]
    assert "fixed meetings" not in prompts[0]      # meetings come from the agents, not the world


def test_initial_plan_uses_what_the_agent_remembers_about_the_project(
    fake_llm, isolated_data, w, monkeypatch
):
    stream = MemoryStream("adam")
    stream.add_observations([
        "The team tracks its work on the project board in the conference room.",
        "Adam Beck is responsible for the website's backend.",
    ])
    prompts = []
    monkeypatch.setattr(llm, "generate", _capture(prompts, "Continue the plan"))
    Planner(w["agents"]["adam"], stream).generate_initial_plan(w["agents"]["adam"])

    assert "What Adam Beck remembers about the project:" in prompts[0]
    assert "- Adam Beck is responsible for the website's backend." in prompts[0]


def test_task_guidance_is_empty_without_tasks(fake_llm, isolated_data, world, monkeypatch):
    rt = _runtime(world, monkeypatch, [])
    assert rt.planner.task_guidance(rt.agent, []) == ""


def test_replan_guidance_reflects_the_board_even_without_own_tasks(fake_llm, isolated_data, w):
    pb.create_task(w, "eve", "Home page", area="frontend", assign_to_self=True)
    guidance = Planner(w["agents"]["adam"]).task_guidance(w["agents"]["adam"], w)
    assert "The project board, as Adam Beck last saw it:" in guidance
    assert "no open backend tasks" in guidance
    assert "add or take on tasks at the project board" in guidance


def test_planning_guidance_names_the_real_tasks(w):
    board_pages_wait_on_api(w)
    guidance = " ".join(Planner(w["agents"]["adam"]).task_guidance(w["agents"]["adam"], w).split())
    assert "The only tasks on the board are T-1, T-2, T-3, T-4;" in guidance
    assert "Don't plan work on a task that can't start yet because it waits on another" in guidance


def test_planning_guidance_with_an_empty_board_names_no_ids(w):
    guidance = " ".join(Planner(w["agents"]["eve"]).task_guidance(w["agents"]["eve"], w).split())
    assert "There are no tasks on the board yet, so don't name any task ids." in guidance
    assert "T-3" not in guidance                       # the old example made one up


def test_the_plan_example_is_a_task_the_agent_can_start(w):
    board_pages_wait_on_api(w)
    pb.create_task(w, "eve", "Navigation bar", area="frontend", assign_to_self=True)
    guidance = Planner(w["agents"]["eve"]).task_guidance(w["agents"]["eve"], w)
    assert "work on T-5 'Navigation bar'" in guidance  # not T-2, which waits on T-1


def test_the_planning_prompt_says_whose_tasks_are_whose(fake_llm, isolated_data, w):
    board_api_and_pages(w)
    adam = w["agents"]["adam"]
    guidance = " ".join(Planner(adam).task_guidance(adam, w).split())
    assert "Adam Beck's own tasks: T-1." in guidance
    assert "Eve Smith's tasks: T-2, T-3, T-4." in guidance


def test_the_plan_example_is_in_the_agents_own_area(fake_llm, isolated_data, w):
    adam = w["agents"]["adam"]
    pb.get_project(w)["tasks"].clear()
    guidance = Planner(adam).task_guidance(adam, w)
    assert "work on the website's backend" in guidance and "navigation bar" not in guidance


def test_the_react_prompt_has_the_facts_about_the_tasks_in_play(fake_llm, isolated_data, w, monkeypatch):
    board_pages_wait_on_api(w)
    prompts = []

    def gen(prompt, options=None, format=None):
        if "DECISION" in prompt:
            prompts.append(" ".join(prompt.split()))
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam = w["agents"]["adam"]
    adam["current_location"] = "conference_room"
    Planner(adam).continue_plan_or_react(
        adam, "status", {"activity": "review the board", "location": "conference room"},
        ["Task T-2 on the project board is todo and assigned to Eve Smith, waiting on T-1."], w)
    p = prompts[0]
    assert "T-1 'Set up backend API for company website' (Adam Beck's) is in progress" in p


def test_the_react_rules_say_what_waits_for_the_task_in_progress(fake_llm, isolated_data, w):
    board_api_pages_and_db(w)
    adam = w["agents"]["adam"]
    rules = " ".join(Planner(adam)._board_rules(adam, w).split())
    assert "T-1 is in progress, so Adam Beck's other tasks wait until it goes to review" in rules


def test_the_reaction_prompt_says_which_ids_exist(fake_llm, isolated_data, w, monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if "DECISION" in prompt:
            prompts.append(" ".join(prompt.split()))
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam = w["agents"]["adam"]
    Planner(adam).continue_plan_or_react(
        adam, "status", {"activity": "clock in", "location": "reception"},
        ["Eve Smith is currently arriving at the reception area."], w)
    assert "The project board has no tasks yet, so no work has a task id or number yet" in prompts[0]
    assert not TASK_ID.search(prompts[0])
