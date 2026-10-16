"""
world.py and perceive.py: rooms, observations, the action an agent is shown
doing, and which object an activity uses (only board work uses the board).
"""

import copy
import json

import pytest

import llm
import project_board as pb
import world as world_module
from conftest import board_api_pages_and_db, fake_generate
from memory_stream import MemoryStream
from perceive import build_observations
from plan import Planner
from world import action_to_present, generate_interactions, mentions_board, refresh_room_states


def _adam(w):
    return w["agents"]["adam"]


def test_room_state_reflects_occupants(world):
    world = copy.deepcopy(world)
    world["agents"]["adam"]["current_location"] = "office_1"
    world["agents"]["eve"]["current_location"] = "office_1"
    refresh_room_states(world)
    assert world["locations"]["office_1"]["state"] == "occupied by Adam Beck and Eve Smith"
    assert world["locations"]["reception"]["state"] == "empty"


def test_observations_include_other_agents_in_the_room(world):
    adam = world["agents"]["adam"]
    observations = build_observations(adam, world)

    assert "Adam Beck is currently at reception." in observations
    assert any(o.startswith("Eve Smith is currently") for o in observations)   # same room
    assert "The time_clock in the reception is on." in observations


def test_observations_ignore_agents_in_other_rooms(world):
    world = copy.deepcopy(world)
    world["agents"]["eve"]["current_location"] = "break_room"
    observations = build_observations(world["agents"]["adam"], world)
    assert not any(o.startswith("Eve Smith") for o in observations)


@pytest.mark.parametrize("plan,shown", [
    ("work on the navigation menu", "working on the navigation menu"),
    ("set up the environment", "setting up the environment"),
    ("updated the project board", "updating the project board"),      # not "updateding"
    ("keep working on T-2 'Homepage'", "still working on T-2 'Homepage'"),
    ("continue working on the API", "still working on the API"),
    ("start building the navigation bar", "starting to build the navigation bar"),
    ("reporting progress on T-1", "reporting progress on T-1"),       # already -ing
    ("Let's quickly check in on T-4", "Let's quickly check in on T-4"),
    ("", ""),
])
def test_shown_actions_read_naturally(plan, shown):
    assert action_to_present(plan) == shown


def test_a_shown_action_keeps_text_after_a_colon(w):
    eve = w["agents"]["eve"]
    chunk = {"activity": "8:05 am: talk with Adam Beck about T-1: is the API ready",
             "location": "office_2", "start_time": "8:05 am", "duration": 10}
    world_module.update_agent_state(eve, chunk, w)
    assert eve["current_action"] == "talking with Adam Beck about T-1: is the API ready"


def test_the_shown_action_follows_the_reconciled_board_step(fake_llm, isolated_data, w):
    board_api_pages_and_db(w)
    eve = w["agents"]["eve"]
    chunk = {"activity": "start T-2 (move it to in progress) on the project board from 8:00 am to 8:10 am",
             "location": "conference room", "start_time": "8:00 am", "duration": 10,
             "target_agent": None, "board_work": True}
    world_module.update_agent_state(eve, chunk, w, location="conference_room")
    assert eve["current_action"].startswith("starting T-2")
    world_module.generate_interactions(eve, w, [], Planner(eve), chunk=chunk)
    assert eve["current_action"] == ("reporting progress on T-2 (already in progress) on the "
                                     "project board from 8:00 am to 8:10 am")


@pytest.mark.parametrize("activity,expected", [
    ("7:30 am: check the project board", True),
    ("add backend tasks", True),
    ("type on the keyboard", False),
    ("onboard the new hire", False),
])
def test_mentions_board(activity, expected):
    assert mentions_board(activity) is expected


def test_working_at_the_board_adds_tasks_and_remembers_it(fake_llm, isolated_data, w):
    agent = _adam(w)
    agent["current_location"] = "conference_room"
    stream = MemoryStream("adam")
    planner = Planner(agent, stream)
    chunk = {"activity": "7:30 am: check the project board and add backend tasks"}

    agent, w = generate_interactions(agent, w, [], planner, chunk=chunk)

    assert agent["interacting_with"] == "project_board"
    [task] = pb.get_project(w)["tasks"]
    assert (task["title"], task["assignee"], task["created_by"]) == \
        ("Set up the website API", "adam", "adam")
    assert stream.memories[-1].description.startswith(
        "Adam Beck added task T-1 'Set up the website API'")


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
