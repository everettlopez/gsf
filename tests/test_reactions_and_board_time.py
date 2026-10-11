"""
Fixes for what the second debug run.log (17:06, 30 steps) showed:

1. Board time: 20 board LLM calls, 8 changes. Board blocks were broken into
   5-minute steps (one board call each), plans had hour-long board blocks, and
   the day's first item ("clock in ... at 7:00 am") filled the whole first hour.
2. 7 of 9 reactions were "move to the project_board to update tasks", made at
   the board or right after it; Eve made the same one at 8:30 and 8:35.
3. "Move to the conference room to update the project board" was taken for a
   plain room change (7:05 Adam, 9:15 Eve).
4. Adam followed Eve: "work on T-1 in the office with Eve Smith" for 3.5 hours,
   and "the discussion" counted as collaborating.
5. A board reply ran out of tokens in its reason and the whole visit was lost.
6. A task titled "T-1: Set up the API ..." and a leftover stand-up at 8:10.
And, closer to the paper: a reaction now regenerates the rest of the day.
"""

import copy

import pytest

import llm
import main
import project_board as pb
from conftest import fake_generate
from plan import Planner


@pytest.fixture
def w(world):
    pb.clear_workload_changed()
    pb.clear_commitments()
    main._last_talk.clear()
    yield copy.deepcopy(world)
    pb.clear_workload_changed()
    pb.clear_commitments()
    main._last_talk.clear()


def _plan(p, lines, agent, w):
    return p.create_chunks(lines, agent, w)


# --- 1. board time -----------------------------------------------------------
EVE_INITIAL = [   # Eve's initial plan in the run
    "clock in and start work at 7:00 am",
    "attend the daily stand-up with the whole team from 8:00 am to 8:15 am",
    "check the project board in the conference room from 8:15 am to 9:00 am",
    "review and update frontend tasks on the project board from 9:00 am to 10:00 am",
    "collaborate with Adam Beck on technical challenges from 10:00 am to 11:00 am",
    "work on implementing a new page layout for the company website from 11:00 am to 12:00 pm",
]


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


def test_initial_plans_get_the_same_cleanup_as_replans(fake_llm, isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if "Continue the plan as a numbered list" in prompt:
            return "\n".join(f"{i + 2}. {line}" for i, line in enumerate(EVE_INITIAL[1:]))
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = main.make_runtime("eve", w)
    main.initial_planning(rt, w)
    board = [c for c in rt.broad_chunks if c.get("board_work")]
    assert board and all(c["duration"] <= 15 for c in board)


def test_a_board_block_is_one_step_not_many(fake_llm, isolated_data, w, monkeypatch):
    rt = main.make_runtime("eve", w)
    calls = []

    def gen(prompt, options=None, format=None):
        if "Break it into actions" in prompt or "Create EXACTLY" in prompt:
            calls.append(prompt)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    p = rt.planner
    # 80 minutes: long enough to have been split into hours, then 5-minute steps
    rt.broad_chunks = _plan(p, ["visit the project board to add frontend tasks from 7:10 am to 8:30 am"],
                            rt.agent, w)
    first = main.get_chunk_lazy(rt, "7:10 am", w)
    later = main.get_chunk_lazy(rt, "8:20 am", w)
    assert calls == []                                    # no decomposition LLM calls
    assert first is later and first["board_work"]         # same chunk: one board call


# --- 2. reactions that only repeat what is happening ----------------------------
def _at_board(w, rt):
    rt.agent["current_location"] = "conference_room"
    rt.agent["interacting_with"] = pb.BOARD_OBJECT


def test_no_board_reaction_while_at_the_board(fake_llm, isolated_data, w):
    rt = main.make_runtime("eve", w)
    _at_board(w, rt)
    rt.chunk = {"activity": "7:25 am: Add new frontend tasks", "board_work": True}
    assert main.pointless_reaction(rt, "Move to the project_board to update tasks",
                                   "7:35 am", w) == "already at the board"


def test_no_board_reaction_right_after_using_the_board(fake_llm, isolated_data, w):
    rt = main.make_runtime("adam", w)
    rt.chunk = {"activity": "8:30 am: Summarize key points", "board_work": False}
    rt.last_board_at = rt.planner._parse_time("8:10 am")
    assert main.pointless_reaction(rt, "Move to the project_board to update task T-4",
                                   "8:35 am", w) == "used the board at 8:10 am"
    assert main.pointless_reaction(rt, "Move to the project_board to update task T-4",
                                   "8:45 am", w) == ""    # 35 minutes later: fine


def test_the_same_reaction_twice_is_ignored(fake_llm, isolated_data, w):
    rt = main.make_runtime("eve", w)
    reaction = "Move to the project_board to update task T-2 and check on T-4's status"
    rt.chunk = {"activity": f"8:30 am: {reaction}", "is_reaction": True, "board_work": True}
    assert main.pointless_reaction(rt, reaction, "8:35 am", w) == "already doing this"


def test_a_real_reaction_still_goes_through(fake_llm, isolated_data, w):
    rt = main.make_runtime("adam", w)
    rt.chunk = {"activity": "9:10 am: Outline the core functionality of the API"}
    assert main.pointless_reaction(rt, "Ask Eve which fields the contact form sends",
                                   "9:15 am", w) == ""


def test_decide_phase_drops_a_pointless_reaction(fake_llm, isolated_data, w, monkeypatch):
    rt = main.make_runtime("eve", w)
    _at_board(w, rt)
    rt.prev_company = set()

    def gen(prompt, options=None, format=None):
        if "DECISION" in prompt:
            return "DECISION: 2\nREACTION: Move to the project_board to update tasks"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    monkeypatch.setattr(main, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "7:25 am: Add new frontend tasks based on current priorities",
        "location": "conference room", "start_time": "7:25 am", "duration": 15,
        "target_agent": None, "board_work": True})
    rt.new_obs = ["Task T-2 on the project board is todo and assigned to Eve Smith."]
    main.decide_phase(rt, 8, "7:35 am", w)
    assert (rt.decision, rt.reaction) == ("1", "")
    assert rt.last_react_step < 0                         # and no cooldown started


def test_board_use_is_remembered_only_when_using_the_board(fake_llm, isolated_data, w):
    rt = main.make_runtime("eve", w)
    rt.agent["interacting_with"] = "desk"
    main.note_board_use(rt, "7:05 am")
    assert rt.last_board_at is None
    rt.agent["interacting_with"] = pb.BOARD_OBJECT
    main.note_board_use(rt, "7:10 am")
    assert rt.planner._fmt(rt.last_board_at) == "7:10 am"


# --- 3. a move with another purpose is a real reaction --------------------------
@pytest.mark.parametrize("reaction,current,expected", [
    ("Walk to the conference room to update the project board",
     "Review the day's tasks and priorities", None),
    ("Move to the conference room to update the project board",
     "Begin designing the homepage layout", None),
    ("Move to office_1 to write the deployment docs", "Debug integration issues", None),
    ("Move to office_1 to set up the development environment as planned",
     "Set up the development environment", "office_1"),
    ("Move to office_1 to continue debugging integration issues",
     "Debug integration issues", "office_1"),
    ("Move to office_1", "Debug integration issues", "office_1"),
])
def test_relocation_only_when_the_purpose_is_the_current_work(w, reaction, current, expected):
    adam = w["agents"]["adam"]
    adam["current_location"] = "office_2"
    assert main.relocation_target(reaction, adam, w, current) == expected


def test_project_board_with_an_underscore_is_board_work():
    assert pb.mentions_board_explicitly("Move to the project_board to update tasks")


# --- paper 4.3.1: a reaction regenerates the plan ---------------------------------
def test_a_reaction_replans_the_rest_of_the_day(fake_llm, isolated_data, w, monkeypatch):
    replans = []

    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            replans.append(" ".join(prompt.split()))
            return "1. work on the API from 9:25 am to 5:00 pm"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = main.make_runtime("adam", w)
    main.initial_planning(rt, w)
    rt.reaction = "Ask Eve which fields the contact form sends"
    main.react(rt, 30, "9:15 am", w)

    assert len(replans) == 1
    assert "it is now 9:25 am" in replans[0]              # after the 10-minute reaction
    assert "decided to: Ask Eve which fields the contact form sends" in replans[0]
    assert rt.override["activity"] == "9:15 am: Ask Eve which fields the contact form sends"
    assert any("work on the API" in c["activity"] for c in rt.broad_chunks)


def test_the_old_detour_is_still_available(fake_llm, isolated_data, w, monkeypatch):
    monkeypatch.setattr(main, "REACT_REPLANS", False)
    calls = []

    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            calls.append(prompt)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = main.make_runtime("adam", w)
    main.initial_planning(rt, w)
    rt.reaction = "Ask Eve which fields the contact form sends"
    main.react(rt, 30, "9:15 am", w)
    assert calls == []


def test_cooldown_is_the_reactions_own_ten_minutes():
    assert main.REACT_COOLDOWN_STEPS * main.STEP_MINUTES == 10


# --- 4. following a teammate only when working with them -------------------------
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
    ("8:30 am: Summarize key points from the discussion", None),
    ("8:20 am: Synchronize the data models", None),
    ("8:10 am: Review authentication requirements with Eve Smith", "eve"),
    ("9:00 am: Discuss integration requirements", "eve"),
])
def test_nouns_are_not_collaboration(w, step, target):
    adam = w["agents"]["adam"]
    parent = {"activity": "collaborate with Eve Smith", "target_agent": "eve"}
    assert Planner(adam)._chunk_from_activity(step, adam, parent)["target_agent"] == target


# --- 5. a cut-off board reply keeps its actions -----------------------------------
def test_actions_come_before_the_reason():
    schema = pb.action_schema({"project": {"tasks": []}, "agents": {"eve": {}}})
    assert list(schema["properties"]) == ["actions", "reason"]


def test_a_reply_cut_off_in_its_reason_still_applies_its_actions(fake_llm, isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if isinstance(format, dict) and "actions" in format["properties"]:
            return ('{"actions": [{"action": "create_task", "title": "Homepage", "area": '
                    '"frontend", "assign_to_self": true}], "reason": "Eve Smith is currently at '
                    'the project board and has promised to')
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    eve = w["agents"]["eve"]
    done = pb.interact(eve, w, "check the project board", planner=Planner(eve))
    assert done == ["Eve Smith added task T-1 'Homepage' (frontend, medium priority) "
                    "to the project board and took it on."]


# --- 6. titles and leftovers --------------------------------------------------------
def test_task_titles_lose_a_copied_task_id(w):
    t = pb.create_task(w, "adam", "T-1: Set up the API endpoints for user authentication")
    assert t["title"] == "Set up the API endpoints for user authentication"


def test_a_replan_during_the_stand_up_drops_the_models_own_stand_up(w):
    eve = w["agents"]["eve"]
    p = Planner(eve)
    chunks = p.apply_schedule(_plan(p, [
        "visit the project board to update tasks from 8:05 am to 8:10 am",
        "attend the daily stand-up with the whole team from 8:10 am to 8:15 am",
        "work on T-2 'Homepage' from 8:15 am to 9:00 am",
    ], eve, w), w, "8:05 am")
    assert not any("stand-up" in c["activity"] for c in chunks)
