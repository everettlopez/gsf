"""
Fixes for what the third debug run.log (18:11, 30 steps) and world.json showed:

1. Eve moved T-3 and T-4 from todo straight to "blocked" (meaning "later"),
   with no dependency behind it, and the next conversation was all about
   "unblocking" them.
2. At 7:05 both agents "reacted" with exactly their planned step, and each
   reaction cost a full replan.
3. "I'll keep working on T-2" counted as a promise to keep at the board, so
   both went back to the board right after talking (third visit in 20 min).
4. Right after they talked, Eve's plan still said "collaborate with Adam
   7:30-8:00", so she stood silently in his office for half an hour.
5. T-1 was "the whole backend API" (done_when stays optional).
6. Turns 3-6 of a conversation restated the same thing in new words.
"""

import copy
import json

import pytest

import llm
import agent as ag
import main
import project_board as pb
from conftest import fake_generate
from conversation import run_conversation


@pytest.fixture
def w(world):
    pb.clear_workload_changed()
    pb.clear_commitments()
    main.clear_talks()
    yield copy.deepcopy(world)
    pb.clear_workload_changed()
    pb.clear_commitments()
    main.clear_talks()


def _eves_tasks(w):
    for title in ("Design and implement homepage layout", "Design and implement about page layout",
                  "Design and implement contact form"):
        pb.create_task(w, "eve", title, area="frontend", assign_to_self=True)


# --- 1. blocked is for started work that got stuck --------------------------------
def test_a_task_that_was_never_started_cannot_be_blocked(w):
    _eves_tasks(w)
    pb.update_status(w, "eve", "T-1", "in_progress")
    with pytest.raises(pb.BoardError, match="T-2 hasn't been started, so it can't be blocked; "
                                            "if it has to wait for another task, add that task "
                                            "to its depends_on with edit_task"):
        pb.update_status(w, "eve", "T-2", "blocked", note="Waiting for T-1 to be completed")
    assert pb.get_task(w, "T-2")["status"] == "todo"


def test_started_or_reviewed_work_can_still_be_blocked(w):
    _eves_tasks(w)
    pb.update_status(w, "eve", "T-1", "in_progress")
    pb.update_status(w, "eve", "T-1", "blocked", note="the API isn't up yet")
    pb.update_status(w, "eve", "T-1", "in_progress")
    pb.update_status(w, "eve", "T-1", "review", note="check the hero section")
    pb.update_status(w, "eve", "T-1", "blocked", note="design assets missing")
    assert pb.get_task(w, "T-1")["status"] == "blocked"


def test_waiting_is_expressed_with_a_dependency(w):
    _eves_tasks(w)
    done, refused = pb.apply_actions(w, w["agents"]["eve"], [
        {"action": "edit_task", "task_id": "T-2", "depends_on": ["T-1"]}])
    assert refused == [] and pb.get_task(w, "T-2")["depends_on"] == ["T-1"]


def test_board_prompt_explains_blocked_and_waiting(w):
    prompt = " ".join(pb.build_prompt(w["agents"]["eve"], w, "check the board", "summary", "").split())
    assert "blocked only when work already started got stuck" in prompt
    assert "gets that task in its depends_on through edit_task instead" in prompt
    assert "Tasks not started yet just stay in todo." in prompt


# --- 2. a reaction that is the plan is not a reaction ------------------------------
@pytest.mark.parametrize("reaction,planned,same", [
    ("go to the conference room to check the project board",
     "check the project board in the conference room from 7:05 am to 7:20 am", True),
    ("Ask Eve which fields the contact form sends", "Outline the API structure", False),
    ("Go to the conference room to review the project board with Eve Smith",
     "Review pending backend tasks", False),
])
def test_is_the_plan(reaction, planned, same):
    assert ag._is_the_plan(reaction, planned) is same


def test_reacting_with_the_planned_step_is_ignored(fake_llm, isolated_data, w):
    rt = ag.Agent.load("adam", w)
    rt.chunk = {"activity": "check the project board in the conference room from 7:05 am to 7:20 am",
                "board_work": True}
    assert ag.pointless_reaction(rt, "go to the conference room to check the project board",
                                   "7:05 am", w) == "that is the plan already"


@pytest.mark.parametrize("reaction,ignored", [
    ("continue working on the homepage layout", True),
    ("keep going with the API endpoints", True),
    ("continue the discussion with Eve about the API", False),     # involves a teammate
])
def test_continue_reactions_are_the_plan(fake_llm, isolated_data, w, reaction, ignored):
    rt = ag.Agent.load("adam", w)
    rt.chunk = {"activity": "7:30 am: Sketch the homepage layout"}
    assert bool(ag.pointless_reaction(rt, reaction, "7:35 am", w)) is ignored


def test_no_replan_for_a_reaction_that_is_the_plan(fake_llm, isolated_data, w, monkeypatch):
    replans = []

    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            replans.append(prompt)
        if "DECISION" in prompt:
            return "DECISION: 2\nREACTION: go to the conference room to check the project board"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("adam", w)
    rt.prev_company = set()
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "check the project board in the conference room from 7:05 am to 7:20 am",
        "location": "conference room", "start_time": "7:05 am", "duration": 15,
        "target_agent": None, "board_work": True})
    rt.new_obs = ["Eve Smith is currently clocking in at reception."]
    rt.decide(2, "7:05 am", w)
    assert rt.decision == "1" and replans == []


# --- 3. promises that need the board ----------------------------------------------
@pytest.mark.parametrize("promise,board", [
    ("Eve Smith will continue working on T-2 and wait for T-1 to be moved to review so she "
     "can unblock T-3 and T-4", False),
    ("Adam Beck will continue working on T-1 and keep Eve Smith updated once it's ready for "
     "review so she can proceed with T-3 and T-4", False),
    ("I will check the accessibility for T-5", False),
    ("I'll move T-1 to review once it's complete", True),
    ("I will approve T-4 after we go over it", True),
    ("I'll start T-3", True),
    ("I'll update the status of T-2", True),
    ("assign T-3 to Eve", True),
    ("I will add the backend tasks for the goal", True),
    ("make sure the project board reflects the current status", True),
])
def test_board_promises_need_a_board_action(promise, board):
    assert pb.is_board_commitment(promise) is board


def test_keep_working_promise_does_not_send_anyone_to_the_board(fake_llm, isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            return "1. work on T-1 in the office from 7:25 am to 5:00 pm"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("adam", w)
    ag.initial_planning(rt, w)
    rt.agent["current_location"] = "conference_room"
    promise = "Adam Beck will continue working on T-1 and keep Eve Smith updated"
    assert ag.replan_after_conversation(rt, promise, "7:20 am", w, partner="Eve Smith")
    assert pb.commitments_for("adam") == []
    ag.queue_board_visit(rt, promise, "7:20 am", w)
    assert rt.queued is None


# --- 4. no silent shadowing right after a talk -----------------------------------
def test_after_talking_a_planned_collaboration_step_happens_in_your_own_office(
        fake_llm, isolated_data, w, monkeypatch):
    eve, adam = ag.Agent.load("eve", w), ag.Agent.load("adam", w)
    eve.agent["current_location"] = "conference_room"
    adam.agent["current_location"] = "office_1"
    main.remember_conversation(eve, adam, "7:20 am", "T-3 and T-4")
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "7:30 am: Start collaborating with Adam Beck on technical challenges",
        "location": "conference room", "start_time": "7:30 am", "duration": 10,
        "target_agent": "adam"})

    eve.decide(8, "7:30 am", w)
    assert eve.chunk["target_agent"] is None
    assert eve.chunk["location"] == "office_2" and eve.chunk["pinned"]
    ag.apply_chunk(eve, w)
    assert eve.agent["current_location"] == "office_2"        # not Adam's office


def test_before_talking_the_step_still_goes_to_the_teammate(fake_llm, isolated_data, w, monkeypatch):
    eve = ag.Agent.load("eve", w)
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "7:30 am: Start collaborating with Adam Beck",
        "location": "conference room", "start_time": "7:30 am", "duration": 10,
        "target_agent": "adam"})
    eve.decide(8, "7:30 am", w)
    assert eve.chunk["target_agent"] == "adam"


def test_the_replan_after_a_talk_knows_they_just_talked(fake_llm, isolated_data, w, monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            prompts.append(" ".join(prompt.split()))
            return "1. work on T-2 in the office from 7:25 am to 5:00 pm"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("eve", w)
    ag.initial_planning(rt, w)
    ag.replan_after_conversation(rt, "I will keep working on T-2", "7:20 am", w, partner="Adam Beck")
    assert ("Eve Smith and Adam Beck just talked, so leave out any block in the next hour that "
            "is only about talking or collaborating with Adam Beck") in prompts[0]


# --- 5. tasks say when they are done, and stay small -------------------------------
def test_a_new_task_does_not_need_done_when(w):
    """done_when is optional: requiring it stopped every task from being created
    (18:48 run: "couldn't create_task: needs done_when" at every board visit)."""
    done, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "create_task", "title": "Auth endpoints", "area": "backend",
         "assign_to_self": True}])
    assert refused == []
    assert done == ["Adam Beck added task T-1 'Auth endpoints' (backend, medium priority) "
                    "to the project board and took it on."]

    pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "create_task", "title": "Contact form endpoint", "area": "backend",
         "done_when": "POST /contact stores a message and returns 201"}])
    assert pb.get_task(w, "T-2")["done_when"].startswith("POST /contact")   # still kept if given


def test_board_prompt_asks_for_small_tasks(w):
    prompt = " ".join(pb.build_prompt(w["agents"]["adam"], w, "check the board", "summary", "").split())
    assert "Keep each task to a few hours of work" in prompt
    assert "done_when" not in prompt.split("Rules:")[1].split("Keep each task")[1].split("Do only")[0]
    assert 'instead of one "build the whole backend" task' in prompt


# --- 6. conversations end when they start repeating --------------------------------
RUN_LINES = [   # the 7:20 conversation in the run, in order
    "Adam, I need to unblock T-3 and T-4 so I can continue working on the about page and "
    "contact form. Let me know if you need help with T-1 or if there's anything else I can assist with.",
    "I'll move T-1 to review once it's complete so you can proceed with T-3 and T-4. Let me know "
    "if you need help with anything else.",
    "I'll start working on T-2 since it's in progress, and I'll wait for T-1 to be moved to "
    "review so I can unblock T-3 and T-4.",
    "I'll focus on completing T-1 so we can move it to review and unblock T-3 and T-4. Let me "
    "know if you need any support with T-2.",
    "I'll continue working on T-2 and wait for T-1 to be moved to review so I can unblock T-3 "
    "and T-4. Let me know if you need help with anything else.",
    "I'll continue working on T-1 and keep you updated once it's ready for review so you can "
    "proceed with T-3 and T-4. Let me know if you need anything else.",
]


def test_a_conversation_that_starts_repeating_ends(fake_llm, isolated_data, w, monkeypatch):
    lines = iter(RUN_LINES)

    def gen(prompt, options=None, format=None):
        if "is talking with" in prompt:
            return next(lines)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    pb.create_task(w, "adam", "Backend API", area="backend", assign_to_self=True)   # T-1
    _eves_tasks(w)                                                                   # T-2..T-4
    eve, adam = ag.Agent.load("eve", w), ag.Agent.load("adam", w)
    history = run_conversation(eve, adam, "7:20 am", "talk with Adam about T-3 and T-4", w)
    assert [who for who, _ in history] == ["Eve Smith", "Adam Beck", "Eve Smith"]   # was 6 turns


def test_a_conversation_with_new_content_runs_on(fake_llm, isolated_data, w, monkeypatch):
    lines = iter(["How is T-1 going?", "The auth endpoints are done; contact is next.",
                  "Great, which fields does the contact form send?", "END"])

    def gen(prompt, options=None, format=None):
        if "is talking with" in prompt:
            return next(lines)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    pb.create_task(w, "adam", "Backend API", area="backend", assign_to_self=True)   # T-1
    eve, adam = ag.Agent.load("eve", w), ag.Agent.load("adam", w)
    history = run_conversation(eve, adam, "9:00 am", "ask Adam about T-1", w)
    assert len(history) == 3


# --- 18:48 run --------------------------------------------------------------------------
def test_agents_create_tasks_at_the_board_without_done_when(fake_llm, isolated_data, w, monkeypatch):
    """The 7:05 board visit from the run, where every create_task was refused."""
    def gen(prompt, options=None, format=None):
        if isinstance(format, dict) and "actions" in format["properties"]:
            return json.dumps({"actions": [
                {"action": "create_task", "title": "Auth endpoints", "area": "backend",
                 "priority": "high", "assign_to_self": True},
                {"action": "create_task", "title": "Contact form endpoint", "area": "backend",
                 "priority": "high", "assign_to_self": True}],
                "reason": "Adam's area has no tasks yet."})
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam = w["agents"]["adam"]
    done = pb.interact(adam, w, "review any new backend tasks added to the project board",
                       planner=ag.Planner(adam))
    assert len(done) == 2 and [t["id"] for t in pb.get_project(w)["tasks"]] == ["T-1", "T-2"]


def test_going_to_the_board_while_the_plan_is_a_board_visit_is_the_plan(fake_llm, isolated_data, w):
    rt = ag.Agent.load("adam", w)
    rt.agent["current_location"] = "reception"
    rt.chunk = {"activity": "review any new backend tasks added to the project board from "
                            "7:05 am to 7:20 am", "board_work": True}
    assert ag.pointless_reaction(rt, "go to the conference room to review the project board",
                                   "7:05 am", w) == "that is the plan already"
