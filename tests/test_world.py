import copy

from perceive import build_observations
from world import action_to_present, refresh_room_states


def test_action_to_present_converts_leading_verb():
    assert action_to_present("work on the navigation menu") == "working on the navigation menu"
    assert action_to_present("") == ""


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
