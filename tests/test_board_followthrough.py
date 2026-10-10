"""
Agents follow through on the board: they can assign and edit tasks, keep promises
made in conversation, learn from refused actions, and board work always happens
at the board. Each test mirrors a case seen in a real run.
"""

import copy
import json

import pytest

import llm
import main
import project_board as pb
from conftest import fake_generate
from memory_stream import MemoryStream
from plan import Planner
from world import generate_interactions


@pytest.fixture
def w(world):
    pb.clear_workload_changed()
    pb.clear_commitments()
    yield copy.deepcopy(world)
    pb.clear_workload_changed()
    pb.clear_commitments()


# --- assigning (run 1: "I'll update the task assignments for T-4") -----------
def test_unassigned_task_can_be_assigned_to_a_teammate_by_name(w):
    pb.create_task(w, "eve", "Implement contact form backend", area="backend")
    pb.assign_task(w, "eve", "T-1", "Adam Beck")

    assert pb.get_task(w, "T-1")["assignee"] == "adam"
    assert pb.workload_notices("adam") == [
        "Eve Smith assigned task T-1 'Implement contact form backend' to Adam Beck "
        "on the project board."]


def test_you_can_hand_over_your_own_task_but_not_someone_elses(w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    with pytest.raises(pb.BoardError, match="only they can hand it over"):
        pb.assign_task(w, "eve", "T-1", "eve")
    pb.assign_task(w, "adam", "T-1", "eve")
    assert pb.get_task(w, "T-1")["assignee"] == "eve"
    assert pb.workload_changed("adam")                  # gave work away: replan too


def test_assign_refusals(w):
    pb.create_task(w, "adam", "Set up the API")
    with pytest.raises(pb.BoardError, match="no teammate called"):
        pb.assign_task(w, "adam", "T-1", "Bob")
    pb.assign_task(w, "adam", "T-1", "eve")
    with pytest.raises(pb.BoardError, match="already assigned to Eve Smith"):
        pb.assign_task(w, "eve", "T-1", "eve")


# --- editing (run 2: "make sure the dependencies are clearly noted") ---------
def test_edit_dependencies_and_priority(w):
    pb.create_task(w, "eve", "Build the home page")
    pb.create_task(w, "eve", "Build the contact form")
    task, changed = pb.edit_task(w, "eve", "T-2", depends_on=["t-1", "T-2", "T-9"],
                                 priority="high")
    assert task["depends_on"] == ["T-1"]                # self and unknown ids dropped
    assert task["priority"] == "high"
    assert changed == ["dependencies", "priority"]
    with pytest.raises(pb.BoardError, match="nothing to change"):
        pb.edit_task(w, "eve", "T-2", priority="high")


def test_only_the_assignee_edits_an_assigned_task(w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    with pytest.raises(pb.BoardError, match="belongs to Adam Beck"):
        pb.edit_task(w, "eve", "T-1", priority="low")


def test_assign_and_edit_actions(w):
    pb.create_task(w, "eve", "Contact form backend", area="backend")
    done, refused = pb.apply_actions(w, w["agents"]["eve"], [
        {"action": "assign_task", "task_id": "T-1", "assignee": "adam",
         "note": "Needs a POST endpoint that emails the form"},
        {"action": "edit_task", "task_id": "T-1", "priority": "high"},   # now Adam's
    ])
    assert done == ["Eve Smith assigned task T-1 'Contact form backend' to Adam Beck "
                    "on the project board, noting: \"Needs a POST endpoint that emails the form\"."]
    assert refused == ["edit_task: T-1 belongs to Adam Beck"]


# --- promises made in conversation -------------------------------------------
def test_board_commitments_are_recognised():
    assert pb.is_board_commitment("I will update the task assignments for T-4")
    assert pb.is_board_commitment("make sure the project board reflects the current status")
    assert not pb.is_board_commitment("bring snacks to the meeting")


def test_promises_reach_the_board_prompt_and_settle_when_acted_on(
    fake_llm, isolated_data, w, monkeypatch
):
    prompts = []

    def capture(prompt, options=None, format=None):
        if isinstance(format, dict) and "actions" in format["properties"]:
            prompts.append(prompt)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", capture)
    eve = w["agents"]["eve"]
    pb.add_commitment("eve", "I will assign T-4 to Adam")

    pb.interact(eve, w, "update the project board", planner=Planner(eve, MemoryStream("eve")))

    assert "Eve Smith promised teammates to do this on the board:" in prompts[0]
    assert "- I will assign T-4 to Adam" in prompts[0]
    assert pb.commitments_for("eve") == []              # she acted, so it's settled


def test_promises_are_dropped_after_visits_without_acting(w):
    pb.add_commitment("eve", "assign T-4")
    pb._after_visit("eve", acted=False)
    assert pb.commitments_for("eve") == ["assign T-4"]
    pb._after_visit("eve", acted=False)
    assert pb.commitments_for("eve") == []


def test_refused_actions_become_memories(fake_llm, isolated_data, w, monkeypatch):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    pb.update_status(w, "adam", "T-1", "review")

    def premature_done(prompt, options=None, format=None):
        if isinstance(format, dict) and "actions" in format["properties"]:
            return json.dumps({"reason": "", "actions": [
                {"action": "update_status", "task_id": "T-1", "status": "done"}]})
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", premature_done)
    stream = MemoryStream("adam")
    adam = w["agents"]["adam"]
    pb.interact(adam, w, "update T-1 on the project board", planner=Planner(adam, stream))

    assert stream.memories[-1].description == (
        "Adam Beck tried to move a task on the project board but couldn't: "
        "T-1 needs a teammate to approve it first.")


# --- board work happens at the board -----------------------------------------
def test_board_activity_goes_to_the_board_without_asking_the_llm(
    fake_llm, isolated_data, w, monkeypatch
):
    rt = main.make_runtime("eve", w)
    monkeypatch.setattr(rt.planner, "choose_location",
                        lambda *a, **k: pytest.fail("board work should not ask the LLM"))
    rt.chunk = {"activity": "11:40 am: reviewing the project board for updates",
                "start_time": "11:40 am", "duration": 10, "target_agent": None}
    main.apply_chunk(rt, w)
    assert rt.agent["current_location"] == "conference_room"
    assert rt.agent["interacting_with"] == "project_board"


def test_board_activity_with_a_teammate_still_uses_the_board(
    fake_llm, isolated_data, w
):
    adam = w["agents"]["adam"]
    adam["current_location"] = "conference_room"
    chunk = {"activity": "review the project board with Eve", "target_agent": "eve"}
    adam, _ = generate_interactions(adam, w, [], Planner(adam, MemoryStream("adam")), chunk)
    assert adam["interacting_with"] == "project_board"


def test_other_work_in_the_conference_room_never_uses_the_board(
    fake_llm, isolated_data, w, monkeypatch
):
    def wants_board(prompt, options=None, format=None):
        if format == "json":
            assert "project_board" not in prompt          # not even offered
            return json.dumps({"interacting_with": "project_board", "changes": [
                {"object": "project_board", "new_state": "wiped"}]})
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", wants_board)
    eve = w["agents"]["eve"]
    eve["current_location"] = "conference_room"
    chunk = {"activity": "designing the homepage layout", "target_agent": None}
    eve, _ = generate_interactions(eve, w, [], Planner(eve, MemoryStream("eve")), chunk)
    assert eve["interacting_with"] is None
    assert pb.get_project(w)["tasks"] == []


@pytest.mark.parametrize("activity,expected", [
    ("updating task statuses based on recent progress", True),
    ("assign T-4 to Adam", True),
    ("update T-1", True),
    ("work on T-1 'Set up the API'", False),
    ("add tests for T-1", False),
])
def test_board_only_activities_route_to_the_board(activity, expected):
    assert pb.mentions_board_explicitly(activity) is expected


# --- after a conversation ------------------------------------------------------
def _revise_returns_board_visit(prompts):
    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            prompts.append(prompt)
            return ("1. update the project board from 11:40 am to 11:50 am\n"
                    "2. work on the home page from 11:50 am to 12:00 pm")
        return fake_generate(prompt, options, format)
    return gen


def test_board_promise_at_the_board_is_kept_right_away(fake_llm, isolated_data, w, monkeypatch):
    prompts = []
    monkeypatch.setattr(llm, "generate", _revise_returns_board_visit(prompts))
    rt = main.make_runtime("eve", w)
    main.initial_planning(rt, w)
    rt.agent["current_location"] = "conference_room"
    promise = "I will assign T-4 to Adam and note the dependencies on the project board"

    main.replan_after_conversation(rt, promise, "11:35 am", w)
    main.queue_board_visit(rt, promise, "11:35 am", w)

    assert pb.commitments_for("eve") == [promise]
    assert "can only keep this promise at the project board" in prompts[0]
    assert rt.queued["location"] == "conference_room" and rt.queued["pinned"]
    assert rt.queued["start_time"] == "11:40 am"


def test_no_immediate_visit_when_the_talk_was_elsewhere(fake_llm, isolated_data, w, monkeypatch):
    monkeypatch.setattr(llm, "generate", _revise_returns_board_visit([]))
    rt = main.make_runtime("eve", w)
    rt.agent["current_location"] = "break_room"
    main.queue_board_visit(rt, "assign T-4 to Adam", "12:30 pm", w)
    assert rt.queued is None


def test_assignee_is_notified_and_replans(fake_llm, isolated_data, w, monkeypatch):
    prompts = []
    monkeypatch.setattr(llm, "generate", _revise_returns_board_visit(prompts))
    rt = main.make_runtime("adam", w)
    main.initial_planning(rt, w)
    pb.create_task(w, "eve", "Contact form backend", area="backend")
    pb.assign_task(w, "eve", "T-1", "adam")

    main.replan_after_board(rt, "11:35 am", w)

    notice = "Eve Smith assigned task T-1 'Contact form backend' to Adam Beck on the project board."
    assert notice in [m.description for m in rt.stream.memories]
    assert notice in prompts[0]
    assert "T-1 'Contact form backend' (todo)" in prompts[0]
