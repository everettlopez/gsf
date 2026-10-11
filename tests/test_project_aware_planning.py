"""
Plans follow the state of the project, and creating, starting, finishing and
handing off work all go through the board with a note a teammate can use.
"""

import copy
import json

import pytest

import clock
import llm
import main
import project_board as pb
from conftest import fake_generate
from memory_stream import MemoryStream
from plan import Planner


@pytest.fixture
def w(world):
    pb.clear_workload_changed()
    pb.clear_commitments()
    yield copy.deepcopy(world)
    pb.clear_workload_changed()
    pb.clear_commitments()


# --- what the board means for one agent ---------------------------------------
def test_empty_board_status(w):
    lines = pb.status_for_agent(w, "adam")
    assert lines[0].startswith("The project board has no tasks yet")
    assert ("There are no open backend tasks on the board, and backend is Adam Beck's "
            "area: Adam Beck should add the backend work the goal needs.") in lines
    assert lines[-1] == "Adam Beck has no open tasks."


def test_frontend_only_board_tells_adam_backend_is_missing(w):
    # The run from 7:00 am: Eve adds and takes four frontend tasks, Adam none.
    for title in ("Home page", "About page", "Contact form", "Accessibility pass"):
        pb.create_task(w, "eve", title, area="frontend", assign_to_self=True)
    adam = pb.status_for_agent(w, "adam")
    assert "Open tasks per area: backend 0, frontend 4." in adam
    assert any("no open backend tasks" in line for line in adam)
    assert not any("no open frontend tasks" in line for line in pb.status_for_agent(w, "eve"))


def test_status_shows_reviews_dependencies_and_unassigned_work(w):
    pb.create_task(w, "adam", "Set up the API", area="backend", assign_to_self=True)
    pb.create_task(w, "eve", "Contact form", area="frontend", assign_to_self=True,
                   depends_on=["T-1"])
    pb.create_task(w, "eve", "Email service", area="backend")
    pb.create_task(w, "eve", "Home page", area="frontend", assign_to_self=True)
    pb.update_status(w, "eve", "T-4", "in_progress")
    pb.update_status(w, "eve", "T-4", "review", note="check mobile layout")

    adam = pb.status_for_agent(w, "adam")
    assert "Unassigned backend tasks: T-3 'Email service'." in adam
    assert "Eve Smith's T-4 'Home page' is waiting for a teammate's review." in adam
    assert "Eve Smith's T-2 'Contact form' is waiting on Adam Beck's T-1." in adam
    assert "Adam Beck's open tasks: T-1 'Set up the API' (todo)." in adam

    pb.approve_task(w, "adam", "T-4")
    assert not any("waiting for a teammate's review" in line
                   for line in pb.status_for_agent(w, "adam"))


# --- notes that go with handoffs ----------------------------------------------
def test_assigning_to_a_teammate_needs_a_note(w):
    pb.create_task(w, "eve", "Contact form backend", area="backend")
    done, refused = pb.apply_actions(w, w["agents"]["eve"], [
        {"action": "assign_task", "task_id": "T-1", "assignee": "adam"}])
    assert done == []
    assert refused == ["assign_task: needs a note to tell the teammate what the task "
                       "needs and why it is theirs"]
    assert pb.get_task(w, "T-1")["assignee"] is None


def test_handoff_note_is_on_the_task_and_in_the_notice(w):
    pb.create_task(w, "eve", "Contact form backend", area="backend")
    pb.apply_actions(w, w["agents"]["eve"], [
        {"action": "assign_task", "task_id": "T-1", "assignee": "Adam Beck",
         "note": "The form needs a POST /contact endpoint; you own the API"}])

    task = pb.get_task(w, "T-1")
    assert pb.latest_note(task)["note"] == "The form needs a POST /contact endpoint; you own the API"
    assert pb.workload_notices("adam") == [
        "Eve Smith assigned task T-1 'Contact form backend' to Adam Beck on the project "
        "board, noting: \"The form needs a POST /contact endpoint; you own the API\"."]
    assert "The latest note on it, from Eve Smith" in pb.board_observations(w)[1]


def test_taking_a_task_yourself_needs_no_note(w):
    pb.create_task(w, "eve", "API", area="backend")
    done, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "assign_task", "task_id": "T-1", "assignee": "adam"}])
    assert refused == [] and pb.get_task(w, "T-1")["assignee"] == "adam"


@pytest.mark.parametrize("status,why", [
    ("review", "say what the reviewer should check"),
    ("blocked", "say what is blocking it"),
])
def test_review_and_blocked_need_a_note(w, status, why):
    clock.set_time("9:00 am")
    pb.create_task(w, "adam", "API", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    clock.set_time("11:30 am")                  # long enough in progress to ask for review
    adam = w["agents"]["adam"]
    _, refused = pb.apply_actions(w, adam, [
        {"action": "update_status", "task_id": "T-1", "status": status}])
    assert refused == [f"update_status: needs a note to {why}"]
    done, _ = pb.apply_actions(w, adam, [
        {"action": "update_status", "task_id": "T-1", "status": status, "note": "see PR"}])
    assert len(done) == 1


def test_board_prompt_shows_area_status_and_note_guidance(w):
    adam = w["agents"]["adam"]
    prompt = pb.build_prompt(adam, w, "check the project board", "summary", "")
    assert "Adam Beck's area: backend" in prompt
    assert "What stands out on the board for Adam Beck:" in prompt
    assert "no open backend tasks" in prompt
    assert "Never a note that only says what Adam Beck is doing right now" in " ".join(prompt.split())


# --- planning from the project's state (paper 4.3) -----------------------------
def _capture(prompts, key):
    def gen(prompt, options=None, format=None):
        if key in prompt:
            prompts.append(prompt)
        return fake_generate(prompt, options, format)
    return gen


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


def test_replan_guidance_reflects_the_board_even_without_own_tasks(fake_llm, isolated_data, w):
    pb.create_task(w, "eve", "Home page", area="frontend", assign_to_self=True)
    guidance = Planner(w["agents"]["adam"]).task_guidance(w["agents"]["adam"], w)
    assert "The project board, as Adam Beck last saw it:" in guidance
    assert "no open backend tasks" in guidance
    assert "add or take on tasks at the project board" in guidance


# --- starting and finishing work go through the board --------------------------
def _chunk(activity, start, minutes, **extra):
    return {"activity": activity, "location": "office_1", "start_time": start,
            "duration": minutes, "target_agent": None, "board_work": False, **extra}


def test_checkpoints_wrap_work_on_a_todo_task(fake_llm, isolated_data, w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    planner = Planner(w["agents"]["adam"])
    plan = [_chunk("work on T-1 'Set up the API' from 9:00 am to 11:00 am", "9:00 am", 120),
            _chunk("lunch in the break room from 12:00 pm to 1:00 pm", "12:00 pm", 60)]

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
    plan = [_chunk("work on T-1 from 9:00 am to 10:00 am", "9:00 am", 60),
            _chunk("work on T-1 from 1:00 pm to 3:00 pm", "1:00 pm", 120)]

    out = planner.add_task_checkpoints(plan, w["agents"]["adam"], pb.open_tasks_for(w, "adam"))

    assert [(c["start_time"], c["board_work"]) for c in out] == [
        ("9:00 am", False), ("1:00 pm", False), ("2:50 pm", True)]   # wrap-up after the last block


def test_existing_board_visit_and_other_peoples_tasks_are_left_alone(fake_llm, isolated_data, w):
    pb.create_task(w, "adam", "API", assign_to_self=True)
    pb.create_task(w, "eve", "Home page", area="frontend", assign_to_self=True)
    planner = Planner(w["agents"]["adam"])
    plan = [_chunk("move T-1 to in progress on the project board from 8:50 am to 9:00 am",
                   "8:50 am", 10, board_work=True),
            _chunk("work on T-1 and help with T-2 from 9:00 am to 9:20 am", "9:00 am", 20)]

    out = planner.add_task_checkpoints(plan, w["agents"]["adam"], pb.open_tasks_for(w, "adam"))
    assert out == plan            # T-1 already visited; 20 min is too short to split; T-2 isn't Adam's


# --- a board visit stays at the board when it is broken down -------------------
def test_board_work_is_inherited_by_finer_chunks(fake_llm, isolated_data, w):
    planner = Planner(w["agents"]["adam"])
    parent = {"activity": "update the project board from 7:10 am to 8:00 am",
              "target_agent": None, "board_work": True}
    child = planner._chunk_from_activity("7:20 am: check all frontend tasks for updates",
                                         w["agents"]["adam"], parent)
    assert child["board_work"] is True


def test_inherited_board_work_goes_to_the_board(fake_llm, isolated_data, w, monkeypatch):
    rt = main.make_runtime("adam", w)
    monkeypatch.setattr(rt.planner, "choose_location",
                        lambda *a, **k: pytest.fail("board work should not ask the LLM"))
    rt.chunk = {"activity": "7:20 am: checking all frontend tasks for updates",
                "start_time": "7:20 am", "duration": 10, "target_agent": None,
                "board_work": True}
    main.apply_chunk(rt, w)
    assert (rt.agent["current_location"], rt.agent["interacting_with"]) == \
        ("conference_room", "project_board")


def test_short_chunks_are_not_sent_for_minute_decomposition(fake_llm, isolated_data, w,
                                                            monkeypatch):
    rt = main.make_runtime("adam", w)
    step = {"activity": "move T-1 to in progress on the project board from 9:00 am to 9:10 am",
            "start_time": "9:00 am", "duration": 10, "target_agent": None, "board_work": True}
    monkeypatch.setattr(llm, "generate", lambda *a, **k: pytest.fail("no LLM call expected"))
    assert main.expand_minutes(rt, step) == [step]


def test_replan_after_the_board_adds_checkpoints(fake_llm, isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            return "1. work on T-1 'Set up the API' in the office from 9:05 am to 11:00 am"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = main.make_runtime("adam", w)
    main.initial_planning(rt, w)
    pb.create_task(w, "adam", "Set up the API", area="backend", assign_to_self=True)
    main.replan_after_board(rt, "9:00 am", w)

    new = [c for c in rt.broad_chunks if c["start_time"] in ("9:05 am", "9:15 am", "10:50 am")]
    assert [(c["start_time"], bool(c.get("board_work"))) for c in new] == [
        ("9:05 am", True), ("9:15 am", False), ("10:50 am", True)]
