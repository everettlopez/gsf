"""
Fixes for what testSim_Plan5 (the first 65 steps of a 130-step run) showed:

1. Stale "start T-1" board steps: T-1 was started at 7:10, the plan still said
   "start T-1 (move it to in progress)" at 8:00, and the model moved it to review.
2. Review 50 minutes after starting, then straight back to in progress.
3. Adam promised T-2 while T-1 was in progress and hit the WIP limit twice.
4. Three times in the same room with no conversation: a planned collaboration
   never became a talk, and "move to office_1" reactions burned the cooldown.
5. choose_location put Adam at Eve's desk for solo work.
6. Two replans in one step after a conversation.
7. "updateding the project board".
"""

import copy
import json

import pytest

import clock
import llm
import main
import project_board as pb
from conftest import fake_generate
from plan import Planner
from world import action_to_present


@pytest.fixture
def w(world):
    pb.clear_workload_changed()
    pb.clear_commitments()
    main._last_talk.clear()
    yield copy.deepcopy(world)
    pb.clear_workload_changed()
    pb.clear_commitments()
    main._last_talk.clear()


def _adam_with_backend_tasks(w):
    for title in ("Backend API", "Database schema"):
        pb.create_task(w, "adam", title, area="backend", assign_to_self=True)


# --- 1. board steps are brought up to date at the board ------------------------
def test_start_step_for_a_task_already_in_progress_becomes_a_progress_report(w):
    _adam_with_backend_tasks(w)
    pb.update_status(w, "adam", "T-1", "in_progress")
    step = "start T-1 (move it to in progress) on the project board from 8:00 am to 8:10 am"
    assert pb.reconcile_activity(w, "adam", step) == (
        "report progress on T-1 (already in progress) on the project board "
        "from 8:00 am to 8:10 am")


def test_start_step_while_another_task_is_in_progress_asks_about_that_one_first(w):
    _adam_with_backend_tasks(w)
    pb.update_status(w, "adam", "T-1", "in_progress")
    out = pb.reconcile_activity(w, "adam", "start T-2 (move it to in progress) on the project board")
    assert "check whether T-1 is finished" in out
    assert "keep T-1 in progress and leave T-2 for later" in out


def test_start_step_that_is_still_valid_is_left_alone(w):
    _adam_with_backend_tasks(w)
    step = "start T-1 (move it to in progress) on the project board"
    assert pb.reconcile_activity(w, "adam", step) == step
    assert pb.reconcile_activity(w, "adam", "update the project board") == "update the project board"


def test_the_board_prompt_gets_the_reconciled_step(fake_llm, isolated_data, w, monkeypatch):
    _adam_with_backend_tasks(w)
    pb.update_status(w, "adam", "T-1", "in_progress")
    prompts = []

    def gen(prompt, options=None, format=None):
        if isinstance(format, dict) and "actions" in format["properties"]:
            prompts.append(prompt)
            return json.dumps({"reason": "", "actions": []})
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam = w["agents"]["adam"]
    pb.interact(adam, w, "start T-1 (move it to in progress) on the project board",
                planner=Planner(adam))
    assert "about to: report progress on T-1 (already in progress)" in prompts[0]
    assert "in progress since 7:00 am" in prompts[0]


# --- 2. review is for finished work, and waits for the teammate -----------------
def test_review_right_after_starting_is_refused(w):
    clock.set_time("7:10 am")
    _adam_with_backend_tasks(w)
    pb.update_status(w, "adam", "T-1", "in_progress")
    clock.set_time("8:00 am")
    adam = w["agents"]["adam"]
    done, refused = pb.apply_actions(w, adam, [
        {"action": "update_status", "task_id": "T-1", "status": "review", "note": "check it"}])
    assert done == []
    assert refused == ["update_status: T-1 only went in progress at 7:10 am; review is "
                       "for finished work, so keep working on it"]

    clock.set_time("9:15 am")
    done, refused = pb.apply_actions(w, adam, [
        {"action": "update_status", "task_id": "T-1", "status": "review", "note": "check it"}])
    assert refused == [] and pb.get_task(w, "T-1")["status"] == "review"


def _in_review(w):
    clock.set_time("7:00 am")
    _adam_with_backend_tasks(w)
    pb.update_status(w, "adam", "T-1", "in_progress")
    pb.update_status(w, "adam", "T-1", "review", note="check the endpoints")


def test_assignee_cannot_pull_a_task_back_out_of_review_on_their_own(w):
    _in_review(w)
    _, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "update_status", "task_id": "T-1", "status": "in_progress"}])
    assert refused == ["update_status: T-1 is waiting for a teammate's review; leave it "
                       "in review unless a teammate asks for changes"]
    assert pb.get_task(w, "T-1")["status"] == "review"


def test_a_teammates_note_lets_the_assignee_reopen_it(w):
    _in_review(w)
    pb.apply_actions(w, w["agents"]["eve"], [
        {"action": "add_note", "task_id": "T-1", "note": "POST /contact returns 500"}])
    done, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "update_status", "task_id": "T-1", "status": "in_progress"}])
    assert refused == [] and pb.get_task(w, "T-1")["status"] == "in_progress"


def test_own_note_does_not_count_as_requested_changes(w):
    _in_review(w)
    pb.add_note(w, "adam", "T-1", "actually I want to tweak it")
    assert not pb.changes_requested(pb.get_task(w, "T-1"))


def test_moving_a_review_task_to_blocked_is_still_allowed(w):
    _in_review(w)
    _, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "update_status", "task_id": "T-1", "status": "blocked",
         "note": "hosting credentials expired"}])
    assert refused == []


# --- 4. teammates in the same room actually talk -------------------------------
def _runtimes(w, room="office_2"):
    adam, eve = main.make_runtime("adam", w), main.make_runtime("eve", w)
    for rt in (adam, eve):
        rt.agent["current_location"] = room
    return adam, eve


def _chunk(activity, **extra):
    return {"activity": activity, "location": "office_2", "start_time": "11:00 am",
            "duration": 60, "target_agent": None, **extra}


def test_planned_collaboration_with_a_teammate_here_starts_a_conversation(
        fake_llm, isolated_data, w, monkeypatch):
    adam, _ = _runtimes(w)
    adam.last_react_step = 50                       # mid-cooldown, as at 11:25 in the log
    monkeypatch.setattr(main, "get_chunk_lazy", lambda rt, t, world=None: _chunk(
        "11:00 am: discuss integration requirements with Eve", target_agent="eve"))

    main.decide_phase(adam, 52, "11:05 am", w)
    assert adam.decision == "2"
    assert adam.reaction == "talk with Eve Smith about: discuss integration requirements with Eve"


def test_planned_collaboration_waits_for_the_conversation_cooldown(
        fake_llm, isolated_data, w, monkeypatch):
    adam, eve = _runtimes(w)
    main.remember_conversation(adam, eve, "10:45 am", "integration")
    monkeypatch.setattr(main, "get_chunk_lazy", lambda rt, t, world=None: _chunk(
        "11:00 am: discuss integration requirements with Eve", target_agent="eve"))

    main.decide_phase(adam, 52, "11:05 am", w)
    assert adam.decision == "1"


def test_a_teammate_walking_in_is_considered_even_during_the_react_cooldown(
        fake_llm, isolated_data, w, monkeypatch):
    monkeypatch.setattr(main, "REACT_COOLDOWN_STEPS", 6)   # the window this test is about
    adam, _ = _runtimes(w, room="conference_room")
    adam.last_react_step = 52                       # reacted at 11:15
    adam.prev_company = set()                       # Eve wasn't here last step
    asked = []

    def gen(prompt, options=None, format=None):
        if "DECISION" in prompt:
            asked.append(prompt)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    monkeypatch.setattr(main, "get_chunk_lazy",
                        lambda rt, t, world=None: _chunk("11:25 am: test API endpoints"))
    adam.new_obs = ["Eve Smith is currently reporting progress on T-4."]

    main.decide_phase(adam, 54, "11:25 am", w)
    assert len(asked) == 1                          # the cooldown didn't swallow it

    asked.clear()
    main.decide_phase(adam, 55, "11:30 am", w)      # she's still here: cooldown applies
    assert asked == []


def test_a_move_only_reaction_relocates_without_a_cooldown(fake_llm, isolated_data, w, monkeypatch):
    adam, _ = _runtimes(w, room="office_2")
    w["agents"]["eve"]["current_location"] = "conference_room"
    adam.prev_company = set()

    def gen(prompt, options=None, format=None):
        if "DECISION" in prompt:
            return ("DECISION: 2\nREACTION: Move to office_1 to set up the development "
                    "environment as planned")
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    monkeypatch.setattr(main, "get_chunk_lazy",
                        lambda rt, t, world=None: rt.override or _chunk(
                            "11:10 am: set up development environment"))
    adam.new_obs = ["The office_2 is currently quiet."]

    main.decide_phase(adam, 52, "11:15 am", w)
    assert (adam.decision, adam.reaction) == ("1", "")
    assert adam.last_react_step < 0                 # no cooldown started
    assert adam.chunk["location"] == "office_1" and adam.chunk["pinned"]
    assert adam.chunk["activity"] == "11:10 am: set up development environment"


@pytest.mark.parametrize("reaction,expected", [
    ("Move to office_1 to continue debugging integration issues", "office_1"),
    ("head back to my office to keep coding", "office_1"),
    ("go to the conference room to update T-1", None),    # board work: a real reaction
    ("Move to office_2 to talk with Eve", None),          # about a teammate: a real reaction
    ("Move to office_2 to check something", None),        # already there
    ("Let's quickly check in on T-4 progress", None),     # not a move
    ("ask Eve about the API", None),
])
def test_relocation_target(w, reaction, expected):
    adam = w["agents"]["adam"]
    adam["current_location"] = "office_2"
    assert main.relocation_target(reaction, adam, w) == expected


# --- 5. no solo work at a teammate's desk --------------------------------------
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


# --- 6. one replan per conversation --------------------------------------------
def test_a_conversation_replan_absorbs_the_pending_board_replan(
        fake_llm, isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            return "1. work on T-1 'Backend API' in the office from 7:10 am to 12:00 pm"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = main.make_runtime("adam", w)
    main.initial_planning(rt, w)
    _adam_with_backend_tasks(w)                       # claiming marks the workload changed
    pb.assign_task(w, "eve", pb.create_task(w, "eve", "Contact endpoint", area="backend")["id"],
                   "adam", note="the form needs POST /contact")
    assert pb.workload_changed("adam")

    assert main.replan_after_conversation(rt, "I will start on T-1 today", "7:05 am", w)
    assert not pb.workload_changed("adam")            # replan_after_board won't fire again
    memories = [m.description for m in rt.stream.memories]
    assert any("assigned task T-3 'Contact endpoint' to Adam Beck" in m for m in memories)


# --- 7. present participles ----------------------------------------------------
@pytest.mark.parametrize("action,expected", [
    ("updated the project board", "updating the project board"),
    ("update the project board", "updating the project board"),
    ("Let's quickly check in on T-4", "Let's quickly check in on T-4"),
    ("reporting progress on T-1", "reporting progress on T-1"),
    ("set up the environment", "setting up the environment"),
])
def test_action_to_present(action, expected):
    assert action_to_present(action) == expected


# --- testSim_Plan6 -------------------------------------------------------------
# 8. T-5 went from todo straight to review at 9:20 while T-2 sat "in progress".
def _eve_with_frontend_tasks(w):
    for title in ("Homepage UI", "About page UI", "Contact form UI", "Accessibility pass"):
        pb.create_task(w, "eve", title, area="frontend", assign_to_self=True)


def test_a_task_that_was_never_started_cannot_go_to_review(w):
    clock.set_time("7:10 am")
    _eve_with_frontend_tasks(w)
    pb.update_status(w, "eve", "T-1", "in_progress")
    clock.set_time("9:20 am")
    with pytest.raises(pb.BoardError, match="T-4 hasn't been started; move it to in progress"):
        pb.update_status(w, "eve", "T-4", "review", note="check contrast")
    assert pb.get_task(w, "T-4")["status"] == "todo"


def test_a_blocked_task_can_still_go_to_review(w):
    _eve_with_frontend_tasks(w)
    pb.update_status(w, "eve", "T-1", "in_progress")
    pb.update_status(w, "eve", "T-1", "blocked", note="waiting on the API")
    pb.update_status(w, "eve", "T-1", "review", note="unblocked and finished")
    assert pb.get_task(w, "T-1")["status"] == "review"


# 9. Eve promised T-5 (here T-4) but started the first task on the list, then
#    hit the WIP limit three times trying to start the rest.
def test_a_promise_about_a_task_says_to_start_that_task(w):
    _eve_with_frontend_tasks(w)
    eve = w["agents"]["eve"]
    promise = ["I'll check the accessibility for T-4 and make sure the UI components are aligned"]
    assert pb.promised_task_steps(w, eve, promise) == [
        "The promise is about T-4: move T-4 to in progress (not another task)."]

    prompt = " ".join(pb.build_prompt(eve, w, "update the project board", "summary", "",
                                      commitments=promise).split())
    assert "What that means on the board: - The promise is about T-4" in prompt


def test_a_promise_about_a_task_while_another_is_in_progress_explains_the_switch(w):
    _eve_with_frontend_tasks(w)
    pb.update_status(w, "eve", "T-1", "in_progress")
    steps = pb.promised_task_steps(w, w["agents"]["eve"], ["I'll do the accessibility pass, T-4"])
    assert steps == [
        "The promise is about T-4, so T-4 is the task to work on. T-1 is in progress: "
        "move T-1 to review if it is finished, or back to todo if Eve Smith hasn't really "
        "worked on it yet, then move T-4 to in progress."]
    # ...and the switch it describes is allowed on the board.
    done, refused = pb.apply_actions(w, w["agents"]["eve"], [
        {"action": "update_status", "task_id": "T-1", "status": "todo"},
        {"action": "update_status", "task_id": "T-4", "status": "in_progress"}])
    assert refused == [] and len(done) == 2


def test_promises_without_a_task_or_about_started_tasks_add_nothing(w):
    _eve_with_frontend_tasks(w)
    pb.update_status(w, "eve", "T-4", "in_progress")
    eve = w["agents"]["eve"]
    assert pb.promised_task_steps(w, eve, ["I'll review the designs with Adam"]) == []
    assert pb.promised_task_steps(w, eve, ["I'll keep going on T-4"]) == []
    assert pb.promised_task_steps(w, eve, ["I'll review Adam's T-9"]) == []   # no such task


def test_only_one_start_per_visit(w):
    _eve_with_frontend_tasks(w)
    eve = w["agents"]["eve"]
    done, refused = pb.apply_actions(w, eve, [
        {"action": "update_status", "task_id": t, "status": "in_progress"}
        for t in ("T-1", "T-2", "T-3", "T-4")])
    assert len(done) == 1 and refused == []          # was: 1 done, 3 WIP refusals
    assert [t["status"] for t in pb.get_project(w)["tasks"]] == [
        "in_progress", "todo", "todo", "todo"]


def test_a_wip_refusal_is_reported_once(w):
    _eve_with_frontend_tasks(w)
    pb.update_status(w, "eve", "T-1", "in_progress")
    _, refused = pb.apply_actions(w, w["agents"]["eve"], [
        {"action": "update_status", "task_id": t, "status": "in_progress"}
        for t in ("T-2", "T-3", "T-4")])
    assert refused == ["update_status: Eve Smith is already working on T-1; "
                       "move it to review (or blocked, if it is stuck) first"]


def test_finishing_one_task_then_starting_the_next_still_works(w):
    clock.set_time("7:00 am")
    _eve_with_frontend_tasks(w)
    pb.update_status(w, "eve", "T-1", "in_progress")
    clock.set_time("10:00 am")
    done, refused = pb.apply_actions(w, w["agents"]["eve"], [
        {"action": "update_status", "task_id": "T-1", "status": "review", "note": "check nav"},
        {"action": "update_status", "task_id": "T-2", "status": "in_progress"}])
    assert refused == [] and len(done) == 2
