"""
Fixes for what the 130-step run showed: every task started at once and re-set
on every visit, task-less notes failing, an agent stuck in a gap of the broad
plan, the same vague conversation four times, and a meeting filed as a task.
"""

import copy
import json

import pytest

import llm
import main
import project_board as pb
from conftest import fake_generate
from conversation import run_conversation
from memory_stream import MemoryStream
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


# --- 1. one task at a time; repeating a status is not a mistake ---------------
def test_only_one_task_in_progress_at_a_time(w):
    for title in ("API", "Deploy pipeline"):
        pb.create_task(w, "adam", title, area="backend", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    with pytest.raises(pb.BoardError, match="already working on T-1; move it to review"):
        pb.update_status(w, "adam", "T-2", "in_progress")

    pb.update_status(w, "adam", "T-1", "blocked", note="waiting on hosting access")
    pb.update_status(w, "adam", "T-2", "in_progress")          # T-1 is parked


def test_repeating_a_status_is_skipped_quietly(fake_llm, isolated_data, w, monkeypatch):
    pb.create_task(w, "adam", "API", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")

    def resend(prompt, options=None, format=None):
        if isinstance(format, dict) and "actions" in format["properties"]:
            return json.dumps({"reason": "", "actions": [
                {"action": "update_status", "task_id": "T-1", "status": "in_progress"}]})
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", resend)
    stream = MemoryStream("adam")
    before = len(stream.memories)
    adam = w["agents"]["adam"]
    assert pb.interact(adam, w, "report progress on T-1", planner=Planner(adam, stream)) == []
    assert len(stream.memories) == before                      # nothing to remember


# --- 2. notes on the board itself ---------------------------------------------
def test_note_without_a_task_goes_on_the_board(w):
    done, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "add_note", "note": "Hosting decision: we deploy to Fly.io"},
        {"action": "add_note", "note": "Hosting decision: we deploy to Fly.io"},   # repeat
    ])
    assert done == ["Adam Beck left a note on the project board: Hosting decision: we deploy to Fly.io."]
    assert refused == []
    assert pb.board_observations(w)[-1] == ('The latest note on the project board, from Adam Beck, '
                                            'says: "Hosting decision: we deploy to Fly.io"')
    assert "board note (Adam Beck" in pb.board_text_for_prompt(w)


# --- 5. tasks are deliverables -------------------------------------------------
@pytest.mark.parametrize("title", [
    "Review Eve Smith's tasks for frontend/backend integration",
    "Sync with Eve on the API", "Meeting about deployment"])
def test_meetings_and_reviews_are_not_tasks(w, title):
    with pytest.raises(pb.BoardError, match="things to build or ship"):
        pb.create_task(w, "adam", title)


def test_review_in_the_middle_of_a_title_is_fine(w):
    assert pb.create_task(w, "eve", "Code review checklist page")["id"] == "T-1"


# --- checkpoints: one task started, neutral wording ---------------------------
def test_checkpoint_starts_only_one_task(fake_llm, isolated_data, w):
    pb.create_task(w, "adam", "API", assign_to_self=True)
    pb.create_task(w, "adam", "Deploy", assign_to_self=True)
    plan = [{"activity": "work on T-1 and T-2 from 9:00 am to 11:00 am", "location": "office_1",
             "start_time": "9:00 am", "duration": 120, "target_agent": None, "board_work": False}]
    out = Planner(w["agents"]["adam"]).add_task_checkpoints(
        plan, w["agents"]["adam"], pb.open_tasks_for(w, "adam"))
    assert out[0]["activity"].startswith("start T-1 (move it to in progress)")
    assert out[-1]["activity"].startswith("report progress on T-1, T-2")


# --- 3. gaps in the broad plan --------------------------------------------------
def test_gap_in_the_broad_plan_is_filled_with_the_task_in_progress(
    fake_llm, isolated_data, w
):
    rt = main.make_runtime("adam", w)
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    rt.broad_chunks = [
        {"activity": "update the project board from 11:15 am to 11:20 am",
         "location": "conference room", "start_time": "11:15 am", "duration": 5,
         "target_agent": None, "board_work": True},
        {"activity": "lunch in the break room from 12:30 pm to 1:30 pm",
         "location": "break room", "start_time": "12:30 pm", "duration": 60,
         "target_agent": None, "board_work": False},
    ]
    chunk = main.get_chunk_lazy(rt, "11:40 am", w)
    assert rt.planner.covers(chunk, "11:40 am")
    assert "board" not in chunk["activity"]                     # not stuck on the 11:15 step

    gap = main.gap_block(rt, "11:40 am", w)
    assert gap["activity"] == ("keep working on T-1 'Set up the API' in the office "
                               "from 11:20 am to 12:30 pm")
    assert (gap["start_time"], gap["duration"]) == ("11:20 am", 70)


def test_no_gap_filling_after_the_workday(fake_llm, isolated_data, w):
    rt = main.make_runtime("adam", w)
    rt.broad_chunks = [{"activity": "clock out at 4:00 pm", "location": "reception",
                        "start_time": "4:00 pm", "duration": 60, "target_agent": None}]
    assert main.gap_block(rt, "5:10 pm", w) is None


def test_replan_prompt_asks_for_no_gaps_and_no_reaction_wording(fake_llm, isolated_data, w,
                                                                monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            prompts.append(prompt)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = main.make_runtime("eve", w)
    main.initial_planning(rt, w)
    main.replan_after_conversation(rt, "build T-4 first", "9:00 am", w)
    assert "What just happened: Eve Smith agreed in a conversation to: build T-4 first" in prompts[0]
    assert "decided to react" not in prompts[0]
    assert "(no gaps)" in prompts[0]


# --- 4. conversations: specific, remembered, not repeated ----------------------
def test_conversation_sees_both_agents_tasks(fake_llm, isolated_data, w, monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if "is talking with" in prompt:
            prompts.append(prompt)
            return "END" if len(prompts) > 1 else "Can you review T-4?"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam, eve = main.make_runtime("adam", w), main.make_runtime("eve", w)
    pb.create_task(w, "adam", "Set up the API", area="backend", assign_to_self=True)
    pb.create_task(w, "eve", "Home page", area="frontend", assign_to_self=True)

    run_conversation(eve, adam, "10:00 am", "ask Adam to review T-2", w)

    assert "What Eve Smith knows from the project board:" in prompts[0]
    assert "Adam Beck's open tasks: T-1 'Set up the API' (todo)." in prompts[0]
    assert "name the tasks on the board by their ids" in " ".join(prompts[0].split())
    assert "The only tasks on the board are T-1, T-2; there is no other task id." in prompts[0]


def test_talking_again_too_soon_is_skipped_unless_a_task_is_named(fake_llm, isolated_data, w):
    adam, eve = main.make_runtime("adam", w), main.make_runtime("eve", w)
    main.remember_conversation(adam, eve, "10:25 am", "review integration issues")

    assert main.talked_recently(adam, eve, "11:00 am")
    assert not main.talked_recently(adam, eve, "11:30 am")

    for rt in (adam, eve):
        rt.agent["current_location"] = "conference_room"
    adam.decision, adam.reaction = "2", "walk Eve through the integration issues"
    eve.decision = "1"
    main.resolve_conversations([adam, eve], 20, "10:45 am", w)
    assert (adam.decision, adam.reaction) == ("1", "")          # kept working instead

    memory = "Adam Beck and Eve Smith talked at 10:25 am about: review integration issues."
    assert memory in [m.description for m in adam.stream.memories]
    assert memory in [m.description for m in eve.stream.memories]


def test_a_repeated_promise_does_not_replan_again(fake_llm, isolated_data, w, monkeypatch):
    calls = []

    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            calls.append(prompt)
            return "1. work on the API from 9:05 am to 5:00 pm"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = main.make_runtime("adam", w)
    main.initial_planning(rt, w)

    assert main.replan_after_conversation(
        rt, "I will review the integration issues and task list with Eve", "9:00 am", w)
    assert not main.replan_after_conversation(
        rt, "I will review the integration issues and the task list with Eve Smith",
        "10:30 am", w)
    assert main.replan_after_conversation(rt, "I will deploy T-2 to staging", "11:00 am", w)
    assert len(calls) == 2
