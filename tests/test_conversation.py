"""
conversation.py and the conversation step in main.py: what each speaker knows
(the board, whose tasks are whose, only real task ids), when a talk ends (a
repeat, a claim to have done what was just promised), who talks when, and
what a talk leaves behind (commitments without made-up or past-tense claims).
"""

import json
import re

import pytest

import agent as ag
import llm
import main
import project_board as pb
from conftest import board_api_and_pages, board_pages_wait_on_api, fake_generate
from conversation import _board_context, claims_promised_work, run_conversation
from plan import Planner, future_part


def _talk(w, monkeypatch, replies):
    replies = iter(replies)
    calls = []

    def gen(prompt, options=None, format=None):
        if "is talking with" in prompt:
            calls.append(prompt)
            return next(replies)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam, eve = ag.Agent.load("adam", w), ag.Agent.load("eve", w)
    history = run_conversation(adam, eve, "7:05 am", "Greet Eve and start reviewing tasks", w)
    return history, calls, adam


TASK_ID = re.compile(r"\bT-\d+\b")


def _eves_tasks(w):
    for title in ("Design and implement homepage layout", "Design and implement about page layout",
                  "Design and implement contact form"):
        pb.create_task(w, "eve", title, area="frontend", assign_to_self=True)


RUN_LINES = [   # a talk where each line restates the one before in new words
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


def test_conversation_sees_both_agents_tasks(fake_llm, isolated_data, w, monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if "is talking with" in prompt:
            prompts.append(prompt)
            return "END" if len(prompts) > 1 else "Can you review T-4?"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam, eve = ag.Agent.load("adam", w), ag.Agent.load("eve", w)
    pb.create_task(w, "adam", "Set up the API", area="backend", assign_to_self=True)
    pb.create_task(w, "eve", "Home page", area="frontend", assign_to_self=True)

    run_conversation(eve, adam, "10:00 am", "ask Adam to review T-2", w)

    assert "What Eve Smith knows from the project board:" in prompts[0]
    assert "Adam Beck's open tasks: T-1 'Set up the API' (todo)." in prompts[0]
    assert "The only tasks on the board are T-1, T-2; there is no other task id." in prompts[0]


def test_conversations_see_the_facts_and_that_talking_moves_nothing(
        fake_llm, isolated_data, w, monkeypatch):
    board_pages_wait_on_api(w)
    prompts = []

    def gen(prompt, options=None, format=None):
        if "is talking with" in prompt:
            prompts.append(" ".join(prompt.split()))
            return "END" if len(prompts) > 1 else "How is T-1 going?"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam, eve = ag.Agent.load("adam", w), ag.Agent.load("eve", w)
    run_conversation(adam, eve, "7:05 am", "Review and approve T-1 to unblock Eve's frontend tasks", w)

    first = prompts[0]
    assert "Where those tasks really stand on the board right now:" in first
    assert "T-1 'Set up backend API for company website' (Adam Beck's) is in progress since " \
           "7:10 am: not in review, not approved, not done." in first
    assert "It can't start until T-1 is done." in first


def test_speakers_know_whose_tasks_are_whose(fake_llm, isolated_data, w):
    board_api_and_pages(w)
    adam, eve = ag.Agent.load("adam", w), ag.Agent.load("eve", w)
    text = " ".join(_board_context(adam, eve, w, "T-2").split())
    assert "Eve Smith's tasks: T-2, T-3, T-4." in text
    assert "offering to work on one together, which Eve Smith can accept or turn down" in text


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
    adam, eve = ag.Agent.load("adam", w), ag.Agent.load("eve", w)
    run_conversation(eve, adam, "9:00 am", "ask Adam about the API", w)
    assert "\"I'll move T-2 to review once it works\"" in prompts[0]      # Eve's own task


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
    assert [who for who, _ in history] == ["Eve Smith", "Adam Beck", "Eve Smith"]   # not 6 turns


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


def test_claiming_promised_work_mid_talk_is_caught():
    promise = ("Let's finalize the plan first so we can structure the backend tasks properly "
               "before adding frontend work to the board. I'll start by outlining the API "
               "endpoints and database structure.")
    claim = ("I've outlined the API endpoints and database structure. Let's review them and "
             "then you can add the frontend tasks to the board.")
    assert claims_promised_work(claim, [promise])
    assert not claims_promised_work(claim, [])                           # nothing promised here
    assert not claims_promised_work("I've started sketching the layout.", ["I'll set up the database."])


def test_the_made_up_line_ends_the_talk_unremembered(fake_llm, isolated_data, w, monkeypatch):
    # Adam promises to outline the API, then claims two lines later to have done it.
    lines = iter([
        "Adam, let me check the time clock with you—should I add the frontend tasks to the "
        "board now, or do you want to finalize the plan first?",
        "SAY: Let's finalize the plan first so we can structure the backend tasks properly before "
        "adding frontend work to the board. I'll start by outlining the API endpoints and database structure.",
        "SAY: I'll wait until you've outlined the API endpoints and database structure so we can "
        "align on the frontend tasks to add to the board.",
        "SAY: I've outlined the API endpoints and database structure. Let's review them and then "
        "you can add the frontend tasks to the board."])

    def gen(prompt, options=None, format=None):
        if "talking with" in prompt:
            return next(lines)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    eve, adam = ag.Agent.load("eve", w), ag.Agent.load("adam", w)
    for rt in (eve, adam):
        rt.status = "at reception"
    history = run_conversation(eve, adam, "7:05 am", "ask Adam about the tasks", w)
    assert len(history) == 3
    assert not any("I've outlined" in m.description for m in adam.stream.memories)


def test_talking_again_too_soon_is_skipped_unless_a_task_is_named(fake_llm, isolated_data, w):
    adam, eve = ag.Agent.load("adam", w), ag.Agent.load("eve", w)
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


@pytest.mark.parametrize("said,kept", [
    ("I've approved T-1 and will provide input or review as needed",
     "I will provide input or review as needed"),
    ("I've approved T-1", ""),
    ("I approved T-1", ""),
    ("I've marked T-1 done, so I'll start T-2", "I will start T-2"),
    ("I will start with T-2 'Design and implement homepage layout' next",
     "I will start with T-2 'Design and implement homepage layout' next"),
    ("I'll check the accessibility for T-5", "I'll check the accessibility for T-5"),
])
def test_claims_about_the_past_are_not_commitments(said, kept):
    assert future_part(said) == kept


def test_extract_outcomes_drops_claimed_actions(fake_llm, isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if isinstance(format, dict) and "meeting_agreed" in format["properties"]:
            return json.dumps({"meeting_agreed": False, "location": "office_1", "activity": "",
                               "minutes": 0, "commitments": [
                                   {"person": "Adam Beck", "commitment": "I've approved T-1"},
                                   {"person": "Eve Smith", "commitment": "I will start on T-2"}]})
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    board_pages_wait_on_api(w)                                   # T-2 exists, so it's kept
    adam, eve = w["agents"]["adam"], w["agents"]["eve"]
    out = Planner(adam).extract_outcomes("Adam: ...\nEve: ...", adam, eve, w)
    assert out["commitments"] == {"Eve Smith": "I will start on T-2"}


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


@pytest.mark.parametrize("commitment,kept", [
    ("Adam Beck will update the board once there is progress on T-2, T-3, and T-4", ""),
    ("I will ensure T-2 is stable and meets the review criteria so we can move it to review", ""),
    ("I will pair with Eve on T-2 at 10 am", "I will pair with Eve on T-2 at 10 am"),
    ("I will help Eve Smith with T-2 after lunch", "I will help Eve Smith with T-2 after lunch"),
    ("I will review T-2 with Eve once it is in review", "I will review T-2 with Eve once it is in review"),
    ("I will move T-1 to review and check on T-2", "I will move T-1 to review"),
    ("I will finish T-1", "I will finish T-1"),
    # Cutting just the id out left "I will ensure is properly set up" and
    # "keep T-6 in todo until is in review".
    ("I will make sure T-2 is properly set up so we can move it to in progress, "
     "and I'll check in on T-1 later today", "I'll check in on T-1 later today"),
    ("I will start T-1 as soon as T-2 moves to review", "I will start T-1 as soon as T-2 moves to review"),
    ("I will keep T-1 in todo until T-2 is in review", "I will keep T-1 in todo until T-2 is in review"),
])
def test_promises_about_a_teammates_task(w, commitment, kept):
    board_api_and_pages(w)
    adam = w["agents"]["adam"]
    assert Planner(adam).own_commitment(commitment, adam, w) == kept


def test_outcomes_drop_promises_about_a_teammates_tasks(
        fake_llm, isolated_data, w, monkeypatch):
    board_api_and_pages(w)
    prompts = []

    def gen(prompt, options=None, format=None):
        if isinstance(format, dict) and "meeting_agreed" in format["properties"]:
            prompts.append(" ".join(prompt.split()))
            return ('{"meeting_agreed": false, "location": "office_1", "activity": "", "minutes": 0,'
                    ' "commitments": ['
                    '{"person": "Adam Beck", "commitment": "Adam Beck will update the board once '
                    'there is progress on T-2, T-3, and T-4"},'
                    '{"person": "Eve Smith", "commitment": "Eve Smith will start with T-2 and T-3 '
                    'and work on the contact form later"}]}')
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam, eve = w["agents"]["adam"], w["agents"]["eve"]
    out = Planner(adam).extract_outcomes("ADAM: ...\nEVE: ...", adam, eve, w)
    assert out["commitments"] == {
        "Eve Smith": "Eve Smith will start with T-2 and T-3 and work on the contact form later"}
