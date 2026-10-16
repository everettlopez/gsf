"""
Meetings two agents agree on in conversation go into both their days.

In the run behind this, each agent wrote its own "collaborate with ..." blocks:
Adam planned 11:15-12:00 with Eve (her lunch was 11:00-12:00), Eve planned
3:30-4:30 with Adam (not in his plan), the 8:00 stand-up was dropped, and they
were alone in their offices from 7:35 to 9:30. In the paper, agents agree on a
time and place in conversation (Isabella's party) and both plan around it.
"""

import copy
import json

import pytest

import llm
import agent as ag
import main
import project_board as pb
from conftest import fake_generate
from conversation import _plans_with
from plan import Planner


@pytest.fixture
def w(world):
    pb.clear_workload_changed()
    pb.clear_commitments()
    main.clear_talks()
    yield copy.deepcopy(world)
    pb.clear_workload_changed()
    pb.clear_commitments()
    main.clear_talks()


ADAM_DAY = [
    "clock in at reception from 7:00 am to 7:05 am",
    "check the project board in the conference room from 7:05 am to 7:20 am",
    "work on backend API development for the website from 7:20 am to 11:00 am",
    "collaborate with Eve Smith on frontend-backend integration challenges from 11:15 am to 12:00 pm",
    "take a lunch break in the break room from 12:00 pm to 1:00 pm",
]
EVE_DAY = [
    "clock in at reception from 7:00 am to 7:05 am",
    "check the project board in the conference room from 7:05 am to 7:20 am",
    "sketch additional page layouts for the website from 7:20 am to 11:00 am",
    "take a lunch break in the break room from 11:00 am to 12:00 pm",
    "collaborate with Adam Beck on technical challenges from 3:30 pm to 4:30 pm",
]
STANDUP = {"location": "conference_room", "activity": "stand-up on the API and the homepage",
           "minutes": 15, "start": "8:00 am"}


def _pair(w):
    adam, eve = ag.Agent.load("adam", w), ag.Agent.load("eve", w)
    adam.broad_chunks = adam.planner.create_chunks(ADAM_DAY, adam.agent, w)
    eve.broad_chunks = eve.planner.create_chunks(EVE_DAY, eve.agent, w)
    return adam, eve


def _at(rt, time):
    return next(c for c in rt.broad_chunks if rt.planner.covers(c, time))


# --- extracting the agreement ---------------------------------------------------
@pytest.mark.parametrize("said,start", [
    ("10:30 am", "10:30 am"),
    ("10 am", "10:00 am"),
    ("at 2:15pm", "2:15 pm"),
    ("now", None),                 # right afterwards: the existing path
    ("", None),
    ("7:10 am", None),             # already here
    ("6:00 pm", None),             # after the workday
])
def test_when_an_agreed_meeting_starts(w, said, start):
    assert Planner(w["agents"]["adam"])._meeting_start(said, "7:10 am") == start


def test_extraction_asks_for_later_meetings_and_keeps_the_time(fake_llm, isolated_data, w, monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if isinstance(format, dict) and "meeting_agreed" in format["properties"]:
            prompts.append(" ".join(prompt.split()))
            assert "meeting_time" in format["properties"]
            return json.dumps({"meeting_agreed": True, "location": "conference_room",
                               "activity": "stand-up", "meeting_time": "8:00 am", "minutes": 15,
                               "commitments": []})
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam, eve = w["agents"]["adam"], w["agents"]["eve"]
    out = Planner(adam).extract_outcomes("...", adam, eve, w, "7:10 am")
    assert out["meeting"] == {"location": "conference_room", "activity": "stand-up",
                              "minutes": 15, "start": "8:00 am"}
    assert "either right afterwards or at a set time later today" in prompts[0]


@pytest.mark.parametrize("activity,said,start", [
    ("check-in at 8:00 am", "now", "8:00 am"),     # the run: agreed at 7:45, put at 7:50
    ("check-in at 7:00 am", "now", None),          # already past: right afterwards
    ("check-in", "now", None),
    ("check-in", "9:30 am", "9:30 am"),
])
def test_a_time_named_in_the_activity_wins_over_now(fake_llm, isolated_data, w, monkeypatch,
                                                    activity, said, start):
    monkeypatch.setattr(llm, "generate", lambda p, options=None, format=None: json.dumps(
        {"meeting_agreed": True, "location": "conference_room", "activity": activity,
         "meeting_time": said, "minutes": 30, "commitments": []}))
    adam, eve = w["agents"]["adam"], w["agents"]["eve"]
    out = Planner(adam).extract_outcomes("...", adam, eve, w, "7:45 am")
    assert out["meeting"]["start"] == start


# --- both plans ----------------------------------------------------------------------
def test_the_meeting_goes_into_both_days(fake_llm, isolated_data, w):
    adam, eve = _pair(w)
    main.schedule_meeting(adam, eve, STANDUP, "7:10 am")
    for rt, other in ((adam, "eve"), (eve, "adam")):
        block = _at(rt, "8:05 am")
        assert block["agreed"] and block["pinned"]
        assert block["target_agent"] == other and block["location"] == "conference_room"
        assert (block["start_time"], block["duration"]) == ("8:00 am", 15)
        # The work around it is cut, not dropped.
        assert rt.planner._parse_time(_at(rt, "7:55 am")["start_time"]) < rt.planner._parse_time("8:00 am")
        assert _at(rt, "8:15 am")["start_time"] == "8:15 am"
        assert any("agreed to meet in conference_room at 8:00 am" in m.description
                   for m in rt.stream.memories)


def test_a_replan_keeps_the_meeting_even_right_after_their_talk(fake_llm, isolated_data, w, monkeypatch):
    """One-sided teammate blocks within the hour after a talk are dropped
    (they just talked), but a meeting they agreed on in that talk stays."""
    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            return ("1. work on the website from 7:15 am to 12:00 pm\n"
                    "2. take a lunch break in the break room from 12:00 pm to 1:00 pm\n"
                    "3. work on the website from 1:00 pm to 5:00 pm")
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam, eve = _pair(w)
    main.remember_conversation(adam, eve, "7:10 am", "the plan")
    main.schedule_meeting(adam, eve, STANDUP, "7:10 am")
    assert ag.replan_rest_of_day(adam, "7:15 am", "talked", w, "test")
    meetings = [c for c in adam.broad_chunks if c.get("agreed")]
    assert [(c["start_time"], c["duration"]) for c in meetings] == [("8:00 am", 15)]


def test_the_meeting_isnt_doubled_by_a_replan_that_kept_it(fake_llm, isolated_data, w):
    adam, eve = _pair(w)
    main.schedule_meeting(adam, eve, STANDUP, "7:10 am")
    agreed = [c for c in adam.broad_chunks if c.get("agreed")]
    p = adam.planner
    own_version = p.create_chunks(["stand-up with Eve Smith in the conference room from 8:00 am to 8:15 am",
                                   "work on the API from 8:15 am to 11:00 am"], adam.agent, w)
    out = p.keep_blocks(own_version, agreed)
    assert [c.get("agreed", False) for c in out if c.get("target_agent") == "eve"] == [True]


def test_a_meeting_is_never_split_by_task_checkpoints(w):
    pb.get_project(w)["tasks"].clear()
    pb.create_task(w, "adam", "API", area="backend", assign_to_self=True)
    adam = w["agents"]["adam"]
    p = Planner(adam)
    meeting = {"activity": "meet Eve Smith in conference_room about: T-1 from 10:00 am to 10:45 am",
               "location": "conference_room", "start_time": "10:00 am", "duration": 45,
               "target_agent": "eve", "pinned": True, "agreed": True}
    assert p.add_task_checkpoints([meeting], adam, pb.open_tasks_for(w, "adam")) == [meeting]


def test_a_meeting_is_one_step_not_broken_down(fake_llm, isolated_data, w):
    adam, eve = _pair(w)
    long = dict(STANDUP, minutes=60, start="10:00 am")
    main.schedule_meeting(adam, eve, long, "7:10 am")
    block = _at(adam, "10:00 am")
    assert ag.expand_hours(adam, block, w) == [block]
    assert ag.expand_minutes(adam, block, w) == [block]


# --- meeting up ----------------------------------------------------------------------
def test_they_talk_at_the_meeting_even_inside_the_cooldown_and_only_once(fake_llm, isolated_data, w):
    adam, eve = _pair(w)
    main.remember_conversation(adam, eve, "7:30 am", "the plan")        # 30 min before
    main.schedule_meeting(adam, eve, STANDUP, "7:30 am")
    adam.chunk = _at(adam, "8:00 am")
    for rt in (adam, eve):
        rt.agent["current_location"] = "conference_room"
    assert ag.planned_partner(adam, w, "8:00 am")["id"] == "eve"
    main.remember_conversation(adam, eve, "8:00 am", "the stand-up")      # held
    assert ag.planned_partner(adam, w, "8:05 am") is None


def test_after_the_meeting_they_go_back_to_their_own_work(fake_llm, isolated_data, w):
    adam, eve = _pair(w)
    main.schedule_meeting(adam, eve, dict(STANDUP, minutes=30), "7:10 am")
    adam.chunk = _at(adam, "8:00 am")
    ag.drop_spent_target(adam, "8:00 am", w)                            # not held yet: go
    assert adam.chunk["agreed"] and adam.chunk["location"] == "conference_room"
    main.remember_conversation(adam, eve, "8:00 am", "the stand-up")
    adam.chunk = _at(adam, "8:05 am")
    ag.drop_spent_target(adam, "8:10 am", w)
    assert adam.chunk["location"] == "office_1" and adam.chunk["target_agent"] is None
    assert "focus on Adam Beck's own work" in adam.chunk["activity"]


# --- in conversation -----------------------------------------------------------------
def test_speakers_see_what_they_planned_with_each_other(fake_llm, isolated_data, w):
    adam, eve = _pair(w)
    text = " ".join(_plans_with(adam, eve, "7:10 am").split())
    assert ("collaborate with Eve Smith on frontend-backend integration challenges from 11:15 am "
            "to 12:00 pm") in text
    assert "it only happens if Eve Smith agrees" in text
    assert _plans_with(adam, eve, "12:00 pm") == ""                     # nothing left


def test_a_talk_that_agrees_a_later_meeting_puts_it_in_both_days(isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if isinstance(format, dict) and "meeting_agreed" in format["properties"]:
            return json.dumps({"meeting_agreed": True, "location": "conference_room",
                               "activity": "stand-up", "meeting_time": "8:00 am", "minutes": 15,
                               "commitments": []})
        if "talking with" in prompt:
            return "SAY: Stand-up at 8 in the conference room?" if "dialogue history" not in prompt else "END"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    monkeypatch.setattr(llm, "embed", __import__("conftest").fake_embed)
    adam, eve = _pair(w)
    for rt in (adam, eve):
        rt.agent["current_location"] = "conference_room"
        rt.status = "at the board"
    adam.decision, adam.reaction = "2", "ask Eve Smith about a stand-up"
    eve.decision = "1"
    main.resolve_conversations([adam, eve], 3, "7:10 am", w)
    for rt in (adam, eve):
        assert _at(rt, "8:05 am").get("agreed")
        assert rt.queued is None                    # not "right afterwards"


def test_the_speakers_prompt_includes_those_plans(fake_llm, isolated_data, w, monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if "talking with" in prompt:
            prompts.append(" ".join(prompt.split()))
            return "END" if "dialogue history" in prompt else "Morning Eve."
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    from conversation import run_conversation
    adam, eve = _pair(w)
    for rt in (adam, eve):
        rt.status = "at reception"
    run_conversation(adam, eve, "7:10 am", "greet Eve", w)
    assert "Adam Beck's plan for later today with Eve Smith:" in prompts[0]


def test_the_meeting_starts_a_talk_as_soon_as_both_are_there(fake_llm, isolated_data, w, monkeypatch):
    """Both arrived at 8:00 but talked at 8:10: the conversation step had its
    own cooldown check that didn't know about agreed meetings."""
    talks = []
    monkeypatch.setattr(main, "run_conversation", lambda a, b, t, reaction, world: talks.append(reaction) or [])
    adam, eve = _pair(w)
    main.remember_conversation(adam, eve, "7:10 am", "the plan")
    main.schedule_meeting(adam, eve, STANDUP, "7:10 am")
    for rt in (adam, eve):
        rt.agent["current_location"] = "conference_room"
        rt.prev_company = {"eve" if rt is adam else "adam"}
    for rt in (adam, eve):
        rt.decide(13, "8:05 am", w)
    main.resolve_conversations([adam, eve], 13, "8:05 am", w)
    assert talks == ["meet Eve Smith in conference_room about: stand-up on the API and the homepage"]


def test_the_talk_at_a_meeting_doesnt_schedule_it_again(isolated_data, w, monkeypatch):
    """At the meeting, extraction can read the meeting itself as "agreed to
    meet in the conference room now"; that would keep them standing there."""
    monkeypatch.setattr(llm, "generate", lambda p, options=None, format=None: (
        json.dumps({"meeting_agreed": True, "location": "conference_room", "activity": "stand-up",
                    "meeting_time": "now", "minutes": 30, "commitments": []})
        if isinstance(format, dict) and "meeting_agreed" in format["properties"]
        else ("SAY: Let's go over it." if "talking with" in p and "dialogue history" not in p
              else "END" if "talking with" in p else fake_generate(p, options, format))))
    monkeypatch.setattr(llm, "embed", __import__("conftest").fake_embed)
    adam, eve = _pair(w)
    main.schedule_meeting(adam, eve, STANDUP, "7:10 am")
    for rt in (adam, eve):
        rt.agent["current_location"] = "conference_room"
        rt.status = "meeting"
        rt.prev_company = {"eve" if rt is adam else "adam"}
    for rt in (adam, eve):
        rt.decide(14, "8:05 am", w)
    main.resolve_conversations([adam, eve], 14, "8:05 am", w)
    assert adam.queued is None and eve.queued is None


def test_a_meeting_agreed_for_right_now_goes_into_both_plans(isolated_data, w, monkeypatch):
    """At 11:00 Eve and Adam agreed to review T-6 in the conference room right
    away. The meeting was only a one-off chunk outside the plans, and Eve's
    replan (told to drop blocks with Adam for the next hour) sent her back to
    her office: "work on T-6 from 11:10 am in office_2"."""
    def gen(prompt, options=None, format=None):
        if isinstance(format, dict) and "meeting_agreed" in format["properties"]:
            return json.dumps({"meeting_agreed": True, "location": "conference_room",
                               "activity": "review T-6's progress", "meeting_time": "now",
                               "minutes": 15, "commitments": []})
        if "talking with" in prompt:
            return "SAY: Let's review it in the conference room." if "dialogue history" not in prompt else "END"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    monkeypatch.setattr(llm, "embed", __import__("conftest").fake_embed)
    adam, eve = _pair(w)
    for rt in (adam, eve):
        rt.agent["current_location"] = "office_2"
        rt.status = "in Eve's office"
    adam.decision, adam.reaction = "2", "check on T-6 with Eve Smith"
    eve.decision = "1"
    main.resolve_conversations([adam, eve], 3, "7:10 am", w)
    for rt in (adam, eve):
        block = _at(rt, "7:15 am")
        assert block.get("agreed") and block["location"] == "conference_room"
        assert block["start_time"] == "7:15 am"
        assert rt.queued is None

    # Still in Eve's office: they go to the conference room before holding it.
    for rt in (adam, eve):
        rt.override = None
        rt.chunk = _at(rt, "7:15 am")
    assert ag.planned_partner(adam, w, "7:15 am") is None
    for rt in (adam, eve):
        rt.agent["current_location"] = "conference_room"
    assert ag.planned_partner(adam, w, "7:20 am") is eve.agent
