import copy

import pytest

import project_board as pb
from memory_stream import MemoryStream
from perceive import build_observations
from plan import Planner
from world import generate_interactions, mentions_board

TASK_FIELDS = {"id", "title", "description", "area", "priority", "status", "assignee",
               "created_by", "depends_on", "done_when", "approved_by", "created_at",
               "updated_at", "history"}


@pytest.fixture
def w(world):
    return copy.deepcopy(world)


def adam(w):
    return w["agents"]["adam"]


def eve(w):
    return w["agents"]["eve"]


# --- world file ------------------------------------------------------------
def test_world_has_an_empty_project_and_a_board_in_the_conference_room(w):
    project = pb.get_project(w)
    assert project["goal"]
    assert project["tasks"] == [] and project["status"] == "not_started"
    assert pb.board_location(w) == "conference_room"


# --- creating tasks --------------------------------------------------------
def test_create_task_is_fully_structured_and_numbered(w):
    t1 = pb.create_task(w, "adam", "Set up the API", "Make the API.", "backend", "high",
                        done_when="Health check passes.")
    t2 = pb.create_task(w, "eve", "Build the home page", area="frontend", depends_on=["t-1"])

    assert set(t1) == TASK_FIELDS
    assert (t1["id"], t2["id"]) == ("T-1", "T-2")
    assert t1["status"] == "todo" and t1["assignee"] is None
    assert t1["history"][0]["event"] == "created"
    assert t2["depends_on"] == ["T-1"]          # ids are normalized
    assert pb.get_project(w)["status"] == "in_progress"


def test_duplicate_titles_are_refused(w):
    pb.create_task(w, "adam", "Set up the API")
    with pytest.raises(pb.BoardError, match="already on the board"):
        pb.create_task(w, "eve", "set up the API!")


def test_bad_fields_fall_back_to_safe_values(w):
    t = pb.create_task(w, "adam", "Thing", area="marketing", priority="urgent",
                       depends_on=["T-99"])
    assert (t["area"], t["priority"], t["depends_on"]) == ("general", "medium", [])


def test_assign_to_self_claims_the_new_task(w):
    t = pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    assert t["assignee"] == "adam"


# --- claiming and status ---------------------------------------------------
def test_a_claimed_task_cannot_be_taken_by_someone_else(w):
    pb.create_task(w, "adam", "Set up the API")
    pb.claim_task(w, "adam", "T-1")
    with pytest.raises(pb.BoardError, match="already claimed by Adam Beck"):
        pb.claim_task(w, "eve", "T-1")


def test_unknown_tasks_are_refused(w):
    with pytest.raises(pb.BoardError, match="no task T-7"):
        pb.claim_task(w, "adam", "T-7")


def test_starting_a_task_claims_it_and_respects_dependencies(w):
    pb.create_task(w, "adam", "Set up the API")
    pb.create_task(w, "eve", "Build the contact form", depends_on=["T-1"])

    with pytest.raises(pb.BoardError, match="waiting on T-1"):
        pb.update_status(w, "eve", "T-2", "in_progress")

    pb.update_status(w, "adam", "T-1", "in_progress")
    assert pb.get_task(w, "T-1")["assignee"] == "adam"
    finish(w, "T-1", note="deployed")
    pb.update_status(w, "eve", "T-2", "in_progress")
    assert pb.get_task(w, "T-1")["history"][-1]["note"] == "deployed"


def finish(w, task_id, owner="adam", reviewer="eve", note=""):
    """Walk an assigned task through review and approval to done."""
    if pb.get_task(w, task_id)["status"] != "in_progress":
        pb.update_status(w, owner, task_id, "in_progress")
    pb.update_status(w, owner, task_id, "review")
    pb.approve_task(w, reviewer, task_id)
    pb.update_status(w, owner, task_id, "done", note=note)


def test_only_the_assignee_moves_a_task(w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    for status in ("in_progress", "review", "done"):
        with pytest.raises(pb.BoardError, match="belongs to Adam Beck"):
            pb.update_status(w, "eve", "T-1", status)


def test_project_is_done_when_every_task_is(w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    finish(w, "T-1")
    assert pb.get_project(w)["status"] == "done"


# --- review and approval ---------------------------------------------------
def test_done_requires_review_and_a_teammates_approval(w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    with pytest.raises(pb.BoardError, match="go to review"):
        pb.update_status(w, "adam", "T-1", "done")

    pb.update_status(w, "adam", "T-1", "review")
    with pytest.raises(pb.BoardError, match="needs a teammate to approve"):
        pb.update_status(w, "adam", "T-1", "done")

    pb.approve_task(w, "eve", "T-1", note="looks good")
    pb.update_status(w, "adam", "T-1", "done")
    assert pb.get_task(w, "T-1")["status"] == "done"


def test_approval_rules(w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    with pytest.raises(pb.BoardError, match="not in review"):
        pb.approve_task(w, "eve", "T-1")

    pb.update_status(w, "adam", "T-1", "in_progress")
    pb.update_status(w, "adam", "T-1", "review")
    with pytest.raises(pb.BoardError, match="your own task"):
        pb.approve_task(w, "adam", "T-1")

    pb.approve_task(w, "eve", "T-1")
    with pytest.raises(pb.BoardError, match="already approved"):
        pb.approve_task(w, "eve", "T-1")


def test_reopening_a_task_clears_its_approvals(w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    pb.update_status(w, "adam", "T-1", "review")
    pb.approve_task(w, "eve", "T-1")
    pb.update_status(w, "adam", "T-1", "in_progress", note="found a bug")
    assert pb.get_task(w, "T-1")["approved_by"] == []


def test_unassigned_tasks_must_be_taken_before_moving(w):
    pb.create_task(w, "adam", "Set up the API")
    with pytest.raises(pb.BoardError, match="take it on first"):
        pb.update_status(w, "adam", "T-1", "review")


def test_approval_shows_on_the_board(w):
    pb.create_task(w, "adam", "Set up the API", area="backend", priority="high",
                   assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    pb.update_status(w, "adam", "T-1", "review")
    pb.approve_task(w, "eve", "T-1")
    assert pb.board_observations(w)[1] == (
        "Task T-1 on the project board, 'Set up the API' (backend, high priority), "
        "is review and assigned to Adam Beck, approved by Eve Smith.")


def test_approve_action_line(w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    pb.update_status(w, "adam", "T-1", "review")
    done, _ = pb.apply_actions(w, eve(w), [{"action": "approve_task", "task_id": "T-1"}])
    assert done == ["Eve Smith approved Adam Beck's task T-1 'Set up the API' on the project board."]


# --- workload changes (drive the replan in main.py) ------------------------
def test_taking_on_or_finishing_work_marks_a_workload_change(w):
    pb.clear_workload_changed()
    pb.create_task(w, "eve", "Write docs")
    assert not pb.workload_changed("eve")            # created, not taken on

    pb.claim_task(w, "eve", "T-1")
    assert pb.workload_changed("eve")
    pb.clear_workload_changed("eve")

    pb.update_status(w, "eve", "T-1", "in_progress")
    pb.update_status(w, "eve", "T-1", "review")
    assert not pb.workload_changed("eve")            # progress alone keeps the plan

    pb.approve_task(w, "adam", "T-1")
    pb.update_status(w, "eve", "T-1", "done")
    assert pb.workload_changed("eve")
    pb.clear_workload_changed()


def test_open_tasks_for_skips_done_and_other_peoples_tasks(w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    pb.create_task(w, "eve", "Build the home page", assign_to_self=True)
    pb.create_task(w, "adam", "Write docs", assign_to_self=True)
    finish(w, "T-3")
    assert [t["id"] for t in pb.open_tasks_for(w, "adam")] == ["T-1"]


# --- what agents see -------------------------------------------------------
def test_empty_board_observations(w):
    assert pb.board_observations(w) == [
        "The project board shows the team goal: Build and deploy the company website.",
        "The project board has no tasks on it yet.",
    ]


def test_board_lines_are_sorted_by_priority_and_show_blockers(w):
    pb.create_task(w, "adam", "Write docs", priority="low")
    pb.create_task(w, "adam", "Set up the API", area="backend", priority="high",
                   assign_to_self=True)
    pb.create_task(w, "eve", "Build the contact form", area="frontend", depends_on=["T-2"])

    lines = pb.board_observations(w)[1:]
    assert lines[0] == ("Task T-2 on the project board, 'Set up the API' (backend, high "
                        "priority), is todo and assigned to Adam Beck.")
    assert lines[1].endswith("is todo and unassigned, waiting on T-2.")
    assert "Write docs" in lines[2]


def test_board_object_state_tracks_the_tasks(w):
    pb.create_task(w, "adam", "Set up the API")
    assert w["locations"]["conference_room"]["objects"]["project_board"]["state"] == \
        "listing 1 tasks (1 todo)"


def test_only_agents_in_the_conference_room_see_the_board(w):
    adam(w)["current_location"] = "conference_room"
    seen = build_observations(adam(w), w)
    assert "The project board has no tasks on it yet." in seen
    assert not any("project_board" in o for o in seen)     # no raw object line

    assert not any("project board" in o for o in build_observations(eve(w), w))


# --- the LLM's proposals ---------------------------------------------------
def test_apply_actions_keeps_good_actions_and_skips_bad_ones(w):
    done, refused = pb.apply_actions(w, adam(w), [
        {"action": "create_task", "title": "Set up the API", "area": "backend",
         "priority": "high", "assign_to_self": True},
        {"action": "create_task", "title": "Set up the API"},        # duplicate
        {"action": "claim_task", "task_id": "T-9"},                  # unknown
        {"action": "update_status", "task_id": "T-1", "status": "in_progress"},
        "garbage",
    ])
    assert done == [
        "Adam Beck added task T-1 'Set up the API' (backend, high priority) to the "
        "project board and took it on.",
        "Adam Beck moved task T-1 'Set up the API' to in progress on the project board.",
    ]
    assert len(refused) == 2


def test_apply_actions_caps_one_visit(w):
    actions = [{"action": "create_task", "title": f"Task {i}"} for i in range(10)]
    done, _ = pb.apply_actions(w, adam(w), actions)
    assert len(done) == pb.MAX_ACTIONS_PER_VISIT


def test_schema_only_offers_existing_task_ids(w):
    assert pb.action_schema(w)["properties"]["actions"]["items"]["properties"]["task_id"] \
        == {"type": "string"}
    pb.create_task(w, "adam", "Set up the API")
    assert pb.action_schema(w)["properties"]["actions"]["items"]["properties"]["task_id"] \
        == {"type": "string", "enum": ["T-1"]}


# --- wired into the simulation ---------------------------------------------
@pytest.mark.parametrize("activity,expected", [
    ("7:30 am: check the project board", True),
    ("add backend tasks", True),
    ("type on the keyboard", False),
    ("onboard the new hire", False),
])
def test_mentions_board(activity, expected):
    assert mentions_board(activity) is expected


def test_board_activity_goes_to_the_conference_room(fake_llm, isolated_data, w):
    planner = Planner(adam(w))
    assert planner.resolve_location_keywords("check the project board", adam(w)) == \
        "conference room"


def test_working_at_the_board_adds_tasks_and_remembers_it(fake_llm, isolated_data, w):
    agent = adam(w)
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


def test_malformed_actions_are_refused_not_raised(w):
    done, refused = pb.apply_actions(w, adam(w), [
        {"action": "claim_task", "task_id": 7},
        {"action": "create_task", "title": ["not", "a", "string"]},
    ])
    assert done == [] and len(refused) == 2


def test_notes_are_visible_to_anyone_reading_the_board(w):
    pb.create_task(w, "eve", "Build the home page", area="frontend", assign_to_self=True)
    pb.add_note(w, "eve", "T-1", "Adam, which accessibility standard should we follow?")

    line = pb.board_observations(w)[1]
    assert line.endswith('The latest note on it, from Eve Smith, says: '
                         '"Adam, which accessibility standard should we follow?"')
    assert "latest note (Eve Smith)" in pb.board_text_for_prompt(w)
