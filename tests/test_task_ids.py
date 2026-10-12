"""
Agents named tasks that didn't exist. In the 18:37 run, with an empty board,
Adam opened the 7:05 conversation with "T-1 is ready for review, and T-2 needs
the API endpoint". The ids came from the prompts themselves: the conversation
prompt said "name the tasks (T-1, T-2, ...)" and used "I'll move T-1 to
review" as an example, and the outcome prompt used "I've approved T-1".

Now no prompt carries an example task id, prompts say which ids exist (or that
none do), a conversation line naming a task that isn't on the board is retried
once and then has the id removed, made-up ids are removed from commitments,
and notes can't mention them.

Also from that run: the first board check was dropped and its time given to
the clock-in ("clock in 7:00-7:20"), which was then split into steps like
"greet the receptionist in office_1".
"""

import copy
import json
import re

import pytest

import llm
import main
import project_board as pb
from conftest import fake_generate
from conversation import run_conversation
from plan import Planner

TASK_ID = re.compile(r"\bT-\d+\b")


@pytest.fixture
def w(world):
    pb.clear_workload_changed()
    pb.clear_commitments()
    main._last_talk.clear()
    yield copy.deepcopy(world)
    pb.clear_workload_changed()
    pb.clear_commitments()
    main._last_talk.clear()


# --- no prompt invents an id ----------------------------------------------------------
def test_no_prompt_mentions_a_task_id_while_the_board_is_empty(isolated_data, monkeypatch):
    """Run the loop on an empty board (board visits change nothing) with one
    reaction that starts a conversation, and look at every prompt sent."""
    prompts = []
    state = {"reacted": False}

    def gen(prompt, options=None, format=None):
        prompts.append(prompt)
        if isinstance(format, dict) and "actions" in format["properties"]:
            return json.dumps({"actions": [], "reason": "nothing to change"})
        if "DECISION" in prompt and "Adam Beck's status" in prompt and not state["reacted"]:
            state["reacted"] = True
            return "DECISION: 2\nREACTION: Greet Eve Smith and start reviewing her tasks together"
        if "decided to react" in prompt:
            return "Morning Eve, what are you starting on today?"
        if "Here is the dialogue history" in prompt:
            return "END"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    monkeypatch.setattr(llm, "embed", __import__("conftest").fake_embed)
    monkeypatch.setattr(main, "NUM_STEPS", 6)
    monkeypatch.setattr(main, "REFLECTION_TIMES", [])
    main.timings.clear()
    main.main()

    assert any("is talking with" in p for p in prompts)          # a conversation happened
    offenders = [TASK_ID.findall(p) for p in prompts if TASK_ID.search(p)]
    assert offenders == []


def test_the_empty_board_line_names_no_id(w):
    line = pb.task_id_line(w)
    assert "no tasks yet" in line and not TASK_ID.search(line)


def test_with_tasks_the_prompts_list_the_real_ids(w):
    pb.create_task(w, "adam", "Auth endpoints", area="backend", assign_to_self=True)
    pb.create_task(w, "eve", "Homepage", area="frontend", assign_to_self=True)
    assert pb.task_id_line(w) == "The only tasks on the board are T-1, T-2; there is no other task id."


def test_the_conversation_example_uses_a_real_task(fake_llm, isolated_data, w, monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if "is talking with" in prompt:
            prompts.append(" ".join(prompt.split()))
            return "END" if len(prompts) > 1 else "How is the API going?"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    pb.create_task(w, "adam", "Auth endpoints", area="backend", assign_to_self=True)
    pb.create_task(w, "eve", "Homepage", area="frontend", assign_to_self=True)
    adam, eve = main.make_runtime("adam", w), main.make_runtime("eve", w)
    run_conversation(eve, adam, "9:00 am", "ask Adam about the API", w)
    assert "\"I'll move T-2 to review once it works\"" in prompts[0]      # Eve's own task


# --- a made-up id in a conversation line ----------------------------------------------
def _talk(w, monkeypatch, replies):
    replies = iter(replies)
    calls = []

    def gen(prompt, options=None, format=None):
        if "is talking with" in prompt:
            calls.append(prompt)
            return next(replies)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam, eve = main.make_runtime("adam", w), main.make_runtime("eve", w)
    history = run_conversation(adam, eve, "7:05 am", "Greet Eve and start reviewing tasks", w)
    return history, calls, adam


def test_a_line_naming_a_missing_task_is_retried(fake_llm, isolated_data, w, monkeypatch):
    history, calls, _ = _talk(w, monkeypatch, [
        "Good morning, Eve. T-1 is ready for review, and T-2 needs the API endpoint.",
        "Good morning, Eve. Shall we put the website work on the board?",
        "END"])
    assert history[0] == ("Adam Beck", "Good morning, Eve. Shall we put the website work on the board?")
    assert "Careful: T-1, T-2 are not on the project board." in calls[1]


def test_if_the_retry_still_invents_one_the_id_is_removed(fake_llm, isolated_data, w, monkeypatch):
    history, _, adam = _talk(w, monkeypatch, [
        "T-1 is ready for review.",
        "Let's look at T-1 together.",
        "END"])
    assert history[0] == ("Adam Beck", "Let's look at together.")
    assert not any(TASK_ID.search(m.description) for m in adam.stream.memories)


def test_lines_about_real_tasks_are_not_touched(fake_llm, isolated_data, w, monkeypatch):
    pb.create_task(w, "adam", "Auth endpoints", area="backend", assign_to_self=True)
    history, calls, _ = _talk(w, monkeypatch, ["T-1 is in progress; auth is half done.", "END"])
    assert history[0][1] == "T-1 is in progress; auth is half done." and len(calls) == 2


# --- commitments, notes, reactions ------------------------------------------------------
def test_commitments_lose_made_up_ids(fake_llm, isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if isinstance(format, dict) and "meeting_agreed" in format["properties"]:
            assert not TASK_ID.search(prompt.split("Conversation between")[0] +
                                      prompt.split("Locations:")[1])
            return json.dumps({"meeting_agreed": False, "location": "office_1", "activity": "",
                               "minutes": 0, "commitments": [
                                   {"person": "Eve Smith", "commitment": "I will review T-1 with Adam"}]})
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam, eve = w["agents"]["adam"], w["agents"]["eve"]
    out = Planner(adam).extract_outcomes("Adam: T-1 is ready\nEve: ok", adam, eve, w)
    assert out["commitments"] == {"Eve Smith": "I will review with Adam"}


def test_a_note_about_a_missing_task_is_refused(w):
    pb.create_task(w, "eve", "Homepage", area="frontend", assign_to_self=True)
    done, refused = pb.apply_actions(w, w["agents"]["eve"], [
        {"action": "add_note", "task_id": "T-1", "note": "Waiting on T-7 from Adam"}])
    assert done == []
    assert refused == ["add_note: the note mentions T-7, but there is no T-7 on the board"]


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


# --- the first board check stays right after clock-in -----------------------------------
ADAM_INITIAL = [   # from the 18:37 run
    "clock in at reception from 7:00 am to 7:05 am",
    "check the project board in the conference room from 7:05 am to 7:20 am",
    "review any new backend tasks that need to be added to the project board from 7:20 am to 7:35 am",
    "collaborate with Eve Smith on technical challenges related to the website from 7:40 am to 8:30 am",
]


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
