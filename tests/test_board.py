"""
project_board.py: the rules of the board. Tasks are created, claimed, assigned
and edited; they move todo -> in_progress -> review -> done, one at a time per
agent, with a teammate's approval before done; notes must be true and useful
to a teammate; a visit's actions are applied, refused or merged.
"""

import pytest

import clock
import project_board as pb


TASK_FIELDS = {"id", "title", "description", "area", "priority", "status", "assignee",
               "created_by", "depends_on", "done_when", "approved_by", "created_at",
               "updated_at", "history"}


def _board_api_and_pages(w):
    """The board at 7:20 in the run."""
    pb.get_project(w)["tasks"].clear()
    pb.create_task(w, "adam", "Design and implement the backend API for the company website",
                   area="backend", priority="high", assign_to_self=True)            # T-1
    for title in ("Design and implement the home page layout",                     # T-2
                  "Design and implement the about page layout",                    # T-3
                  "Design and implement the contact form"):                        # T-4
        pb.create_task(w, "eve", title, area="frontend", priority="high", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    pb.update_status(w, "eve", "T-2", "in_progress")


def _board_api_pages_and_db(w):
    """The board at 8:00 in the run."""
    pb.get_project(w)["tasks"].clear()
    pb.create_task(w, "adam", "Design and implement the API for the company website",
                   area="backend", priority="high", assign_to_self=True)            # T-1
    for title in ("Design and implement the homepage layout",                      # T-2
                  "Design and implement the about page layout",                    # T-3
                  "Design and implement the contact form"):                        # T-4
        pb.create_task(w, "eve", title, area="frontend", priority="high", assign_to_self=True)
    pb.create_task(w, "adam", "Set up database for contact form", area="backend",
                   priority="high", assign_to_self=True)                           # T-5
    pb.update_status(w, "adam", "T-1", "in_progress")
    pb.update_status(w, "eve", "T-2", "in_progress")


def finish(w, task_id, owner="adam", reviewer="eve", note=""):
    """Walk an assigned task through review and approval to done."""
    if pb.get_task(w, task_id)["status"] != "in_progress":
        pb.update_status(w, owner, task_id, "in_progress")
    pb.update_status(w, owner, task_id, "review")
    pb.approve_task(w, reviewer, task_id)
    pb.update_status(w, owner, task_id, "done", note=note)


# 8. T-5 went from todo straight to review at 9:20 while T-2 sat "in progress".
def _eve_with_frontend_tasks(w):
    for title in ("Homepage UI", "About page UI", "Contact form UI", "Accessibility pass"):
        pb.create_task(w, "eve", title, area="frontend", assign_to_self=True)


def _eves_tasks(w):
    for title in ("Design and implement homepage layout", "Design and implement about page layout",
                  "Design and implement contact form"):
        pb.create_task(w, "eve", title, area="frontend", assign_to_self=True)


def _board_pages_wait_on_api(w):
    """The board as it stood from 7:10 on: Adam's T-1 in progress, Eve's three
    pages all waiting on it."""
    clock.set_time("7:00 am")
    pb.create_task(w, "adam", "Set up backend API for company website", area="backend",
                   assign_to_self=True)
    for title in ("Design and implement homepage layout", "Design and implement about page layout",
                  "Implement contact form with validation"):
        pb.create_task(w, "eve", title, area="frontend", depends_on=["T-1"], assign_to_self=True)
    clock.set_time("7:10 am")
    pb.update_status(w, "adam", "T-1", "in_progress")


def _eve(w):
    return w["agents"]["eve"]


def _adam_with_backend_tasks(w):
    for title in ("Backend API", "Database schema"):
        pb.create_task(w, "adam", title, area="backend", assign_to_self=True)


def _in_review(w):
    clock.set_time("7:00 am")
    _adam_with_backend_tasks(w)
    pb.update_status(w, "adam", "T-1", "in_progress")
    pb.update_status(w, "adam", "T-1", "review", note="check the endpoints")


def _adam(w):
    return w["agents"]["adam"]


def test_world_has_an_empty_project_and_a_board_in_the_conference_room(w):
    project = pb.get_project(w)
    assert project["goal"]
    assert project["tasks"] == [] and project["status"] == "not_started"
    assert pb.board_location(w) == "conference_room"


def test_create_task_is_fully_structured_and_numbered(w):
    t1 = pb.create_task(w, "adam", "Set up the API", "Make the API.", "backend", "high",
                        done_when="Health check passes.")
    t2 = pb.create_task(w, "eve", "Build the home page", area="frontend",
                        depends_on=["t-1", "T-1", "T-1"])

    assert set(t1) == TASK_FIELDS
    assert (t1["id"], t2["id"]) == ("T-1", "T-2")
    assert t1["status"] == "todo" and t1["assignee"] is None
    assert t1["history"][0]["event"] == "created"
    assert t2["depends_on"] == ["T-1"]          # ids are normalized and listed once
    assert pb.get_project(w)["status"] == "in_progress"



def test_duplicate_titles_are_refused(w):
    pb.create_task(w, "adam", "Set up the API")
    with pytest.raises(pb.BoardError, match="already on the board"):
        pb.create_task(w, "eve", "set up the API!")


def test_bad_fields_fall_back_to_safe_values(w):
    t = pb.create_task(w, "adam", "Thing", area="marketing", priority="urgent",
                       depends_on=["T-99"])
    assert (t["area"], t["priority"], t["depends_on"]) == ("general", "medium", [])


def test_task_titles_lose_a_copied_task_id(w):
    t = pb.create_task(w, "adam", "T-1: Set up the API endpoints for user authentication")
    assert t["title"] == "Set up the API endpoints for user authentication"


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


@pytest.mark.parametrize("title", [
    "Review Eve Smith's tasks for frontend/backend integration",
    "Sync with Eve on the API", "Meeting about deployment"])
def test_meetings_and_reviews_are_not_tasks(w, title):
    with pytest.raises(pb.BoardError, match="things to build or ship"):
        pb.create_task(w, "adam", title)


def test_review_in_the_middle_of_a_title_is_fine(w):
    assert pb.create_task(w, "eve", "Code review checklist page")["id"] == "T-1"


def test_assign_to_self_claims_the_new_task(w):
    t = pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    assert t["assignee"] == "adam"


def test_a_claimed_task_cannot_be_taken_by_someone_else(w):
    pb.create_task(w, "adam", "Set up the API")
    pb.claim_task(w, "adam", "T-1")
    with pytest.raises(pb.BoardError, match="already claimed by Adam Beck"):
        pb.claim_task(w, "eve", "T-1")


def test_unknown_tasks_are_refused(w):
    with pytest.raises(pb.BoardError, match="no task T-7"):
        pb.claim_task(w, "adam", "T-7")


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


def test_a_task_in_another_area_is_left_for_its_owner(w):
    _board_api_and_pages(w)
    adam = w["agents"]["adam"]
    line = pb.apply_action(w, adam, {"action": "create_task", "title": "Footer", "area": "frontend",
                                     "assign_to_self": True})
    assert "left it unassigned for Eve Smith, whose area that is" in line
    with pytest.raises(pb.BoardError, match="frontend work, which is Eve Smith's area"):
        pb.apply_action(w, adam, {"action": "claim_task", "task_id": "T-5"})
    pb.get_task(w, "T-4")["status"] = "done"                  # Eve had 3 open tasks
    pb.apply_action(w, w["agents"]["eve"], {"action": "claim_task", "task_id": "T-5"})
    assert pb.get_task(w, "T-5")["assignee"] == "eve"


def test_work_is_handed_only_to_its_areas_owner(w):
    _board_api_and_pages(w)
    pb.create_task(w, "adam", "Deploy the API", area="backend")                   # T-5
    adam = w["agents"]["adam"]
    with pytest.raises(pb.BoardError, match="backend work isn't Eve Smith's area"):
        pb.apply_action(w, adam, {"action": "assign_task", "task_id": "T-5", "assignee": "Eve Smith",
                                  "note": "Can you deploy it?"})
    pb.apply_action(w, adam, {"action": "assign_task", "task_id": "T-5", "assignee": "Adam Beck"})
    assert pb.get_task(w, "T-5")["assignee"] == "adam"


def test_general_work_is_anyones(w):
    _board_api_and_pages(w)
    line = pb.apply_action(w, w["agents"]["adam"], {"action": "create_task", "title": "Write the README",
                                                    "area": "general", "assign_to_self": True})
    assert "took it on" in line


def test_a_fourth_task_is_left_unassigned(w):
    _board_api_pages_and_db(w)
    eve = w["agents"]["eve"]                                  # T-2, T-3, T-4
    line = pb.apply_action(w, eve, {"action": "create_task", "title": "Footer", "area": "frontend",
                                    "assign_to_self": True})
    assert "left it unassigned, since Eve Smith already has 3 open tasks" in line
    assert pb.get_task(w, "T-6")["assignee"] is None
    with pytest.raises(pb.BoardError, match="already has 3 open tasks"):
        pb.apply_action(w, eve, {"action": "claim_task", "task_id": "T-6"})


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


def test_only_the_assignee_moves_a_task(w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    for status in ("in_progress", "review", "done"):
        with pytest.raises(pb.BoardError, match="belongs to Adam Beck"):
            pb.update_status(w, "eve", "T-1", status)


def test_unassigned_tasks_must_be_taken_before_moving(w):
    pb.create_task(w, "adam", "Set up the API")
    with pytest.raises(pb.BoardError, match="take it on first"):
        pb.update_status(w, "adam", "T-1", "review")


def test_project_is_done_when_every_task_is(w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    finish(w, "T-1")
    assert pb.get_project(w)["status"] == "done"


def test_only_one_task_in_progress_at_a_time(w):
    for title in ("API", "Deploy pipeline"):
        pb.create_task(w, "adam", title, area="backend", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    with pytest.raises(pb.BoardError, match="already working on T-1; move it to review"):
        pb.update_status(w, "adam", "T-2", "in_progress")

    pb.update_status(w, "adam", "T-1", "blocked", note="waiting on hosting access")
    pb.update_status(w, "adam", "T-2", "in_progress")          # T-1 is parked


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


def test_dropping_an_unneeded_dependency_unblocks_the_task(w):
    _board_pages_wait_on_api(w)
    eve = w["agents"]["eve"]
    done, refused = pb.apply_actions(w, eve, [
        {"action": "edit_task", "task_id": "T-2", "depends_on": []},
        {"action": "update_status", "task_id": "T-2", "status": "in_progress"}])
    assert refused == [] and pb.get_task(w, "T-2")["status"] == "in_progress"
    assert all("T-2" not in line for line in pb.waiting_lines(w, "eve"))


@pytest.mark.parametrize("reaction,why", [
    ("Start working on task T-5 'Set up database for contact form'", "T-1 is still in progress"),
    ("switch to T-5", "T-1 is still in progress"),
    ("keep working on T-1", ""),                    # the task in progress: fine
    ("Check on T-2 progress", ""),                  # not starting anything
])
def test_a_reaction_cannot_start_a_second_task(w, reaction, why):
    _board_api_pages_and_db(w)
    assert pb.start_conflict(w, "adam", reaction).startswith(why) if why \
        else pb.start_conflict(w, "adam", reaction) == ""


def test_a_reaction_cannot_start_a_task_that_waits(w):
    pb.get_project(w)["tasks"].clear()
    pb.create_task(w, "adam", "API", area="backend", assign_to_self=True)
    pb.create_task(w, "eve", "Contact form", area="frontend", depends_on=["T-1"], assign_to_self=True)
    assert pb.start_conflict(w, "eve", "start working on T-2") == "T-2 waits on T-1"


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


def test_approve_action_line(w):
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    pb.update_status(w, "adam", "T-1", "review")
    done, _ = pb.apply_actions(w, _eve(w), [{"action": "approve_task", "task_id": "T-1"}])
    assert done == ["Eve Smith approved Adam Beck's task T-1 'Set up the API' on the project board."]


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


def test_note_without_a_task_goes_on_the_board(w):
    done, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "add_note", "note": "Hosting decision: we deploy to Fly.io"},
        {"action": "add_note", "note": "Hosting decision: we deploy to Fly.io"},   # repeat
    ])
    assert done == ["Adam Beck left a note on the project board: Hosting decision: we deploy to Fly.io."]
    assert refused == []
    assert pb.board_observations(w)[-1] == ('The latest note on the project board, from Adam Beck, '
                                            'says: "Hosting decision: we deploy to Fly.io"')
    assert "board note (Adam Beck" in pb.board_text_for_prompt(w)


@pytest.mark.parametrize("note,ok", [
    ("Eve is currently addressing high-priority bugs or issues. Please check if this affects T-2 progress.", False),
    ("I'm working on the hero section", False),
    ("Eve Smith is still designing the layout", False),
    ("Eve is waiting on the API: Adam, can you share the endpoint list?", True),   # asks Adam
    ("I'm building against mock data until T-1 is ready, is the JSON shape settled?", True),
    ("The homepage needs the API's /pages endpoint before it can show real content", True),
])
def test_notes_that_only_narrate_are_refused(w, note, ok):
    _board_api_pages_and_db(w)
    action = {"action": "add_note", "task_id": "T-2", "note": note}
    if ok:
        pb.apply_action(w, w["agents"]["eve"], action)
    else:
        with pytest.raises(pb.BoardError, match="for a teammate to act on"):
            pb.apply_action(w, w["agents"]["eve"], action)


@pytest.mark.parametrize("note", [
    "T-2 can now proceed since T-1 is complete. Please confirm when you're ready to start.",
    "T-1 'Set up backend API for company website' has been completed",
    "T-1 is approved",
    "T-1 is in review now",
])
def test_notes_that_misstate_the_board_are_refused(w, note):
    _board_pages_wait_on_api(w)
    done, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "add_note", "task_id": "T-2", "note": note}])
    assert done == []
    assert refused[0].startswith("add_note: the note says T-1 is ")
    assert refused[0].endswith("but the board shows T-1 is in progress; notes have to match the board")


@pytest.mark.parametrize("note", [
    "Waiting on T-1 before starting the homepage",
    "Please review and approve T-1 once completed",
    "T-2 needs the /contact endpoint from T-1",
])
def test_ordinary_notes_still_go_through(w, note):
    _board_pages_wait_on_api(w)
    done, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "add_note", "task_id": "T-2", "note": note}])
    assert refused == [] and len(done) == 1


def test_a_true_claim_is_fine(w):
    _board_pages_wait_on_api(w)
    clock.set_time("10:00 am")
    pb.update_status(w, "adam", "T-1", "review", note="check the endpoints")
    pb.approve_task(w, "eve", "T-1")
    pb.update_status(w, "adam", "T-1", "done")
    done, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "add_note", "task_id": "T-2", "note": "T-1 is done, so T-2 can start"}])
    assert refused == []


def test_a_note_about_a_missing_task_is_refused(w):
    pb.create_task(w, "eve", "Homepage", area="frontend", assign_to_self=True)
    done, refused = pb.apply_actions(w, w["agents"]["eve"], [
        {"action": "add_note", "task_id": "T-1", "note": "Waiting on T-7 from Adam"}])
    assert done == []
    assert refused == ["add_note: the note mentions T-7, but there is no T-7 on the board"]


def test_apply_actions_keeps_good_actions_and_skips_bad_ones(w):
    done, refused = pb.apply_actions(w, _adam(w), [
        {"action": "create_task", "title": "Set up the API", "area": "backend",
         "priority": "high", "assign_to_self": True, "done_when": "GET /health answers"},
        {"action": "create_task", "title": "Set up the API", "done_when": "x"},   # duplicate
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
    actions = [{"action": "create_task", "title": f"Task {i}", "done_when": "it works"}
               for i in range(10)]
    done, _ = pb.apply_actions(w, _adam(w), actions)
    assert len(done) == pb.MAX_ACTIONS_PER_VISIT


def test_malformed_actions_are_refused_not_raised(w):
    done, refused = pb.apply_actions(w, _adam(w), [
        {"action": "claim_task", "task_id": 7},
        {"action": "create_task", "title": ["not", "a", "string"]},
    ])
    assert done == [] and len(refused) == 2


def test_three_approvals_of_tasks_not_in_review_are_one_refusal(w):
    _board_pages_wait_on_api(w)
    done, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "approve_task", "task_id": t} for t in ("T-2", "T-3", "T-4")])
    assert refused == ["approve_task: T-2 is not in review (same for T-3, T-4)"]


def test_different_mistakes_stay_separate(w):
    _board_pages_wait_on_api(w)
    _, refused = pb.apply_actions(w, w["agents"]["eve"], [
        {"action": "approve_task", "task_id": "T-2"},
        {"action": "update_status", "task_id": "T-3", "status": "in_progress"}])
    assert refused == ["approve_task: T-2 is not in review",
                       "update_status: T-3 is waiting on T-1"]


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
    ("I will update the task assignments for T-4", True),
    ("bring snacks to the meeting", False),
])
def test_board_promises_need_a_board_action(promise, board):
    assert pb.is_board_commitment(promise) is board



def test_promises_are_dropped_after_visits_without_acting(w):
    pb.add_commitment("eve", "assign T-4")
    pb._after_visit("eve", acted=False)
    assert pb.commitments_for("eve") == ["assign T-4"]
    pb._after_visit("eve", acted=False)
    assert pb.commitments_for("eve") == []
