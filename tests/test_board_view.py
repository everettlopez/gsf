"""
project_board.py: what an agent sees and is told at the board. Observations
and status lines, the facts about each task, which activities count as board
work, stale board steps brought up to date, promises turned into board steps,
and a whole board visit (prompt -> LLM -> actions -> memories).
"""

import json
import re

import pytest

import agent as ag
import clock
import llm
import project_board as pb
from conftest import fake_generate
from memory_retriever import MemoryRetriever
from memory_stream import MemoryStream
from perceive import build_observations
from plan import Planner


def _eve(w):
    return w["agents"]["eve"]


def _adam(w):
    return w["agents"]["adam"]


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


TASK_ID = re.compile(r"\bT-\d+\b")


def _adam_with_backend_tasks(w):
    for title in ("Backend API", "Database schema"):
        pb.create_task(w, "adam", title, area="backend", assign_to_self=True)


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


# 8. T-5 went from todo straight to review at 9:20 while T-2 sat "in progress".
def _eve_with_frontend_tasks(w):
    for title in ("Homepage UI", "About page UI", "Contact form UI", "Accessibility pass"):
        pb.create_task(w, "eve", title, area="frontend", assign_to_self=True)


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


def test_approval_shows_on_the_board(w):
    pb.create_task(w, "adam", "Set up the API", area="backend", priority="high",
                   assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    pb.update_status(w, "adam", "T-1", "review")
    pb.approve_task(w, "eve", "T-1")
    assert pb.board_observations(w)[1] == (
        "Task T-1 on the project board, 'Set up the API' (backend, high priority), "
        "is review and assigned to Adam Beck, approved by Eve Smith.")


def test_notes_are_visible_to_anyone_reading_the_board(w):
    pb.create_task(w, "eve", "Build the home page", area="frontend", assign_to_self=True)
    pb.add_note(w, "eve", "T-1", "Adam, which accessibility standard should we follow?")

    line = pb.board_observations(w)[1]
    assert line.endswith('The latest note on it, from Eve Smith, says: '
                         '"Adam, which accessibility standard should we follow?"')
    assert "latest note (Eve Smith)" in pb.board_text_for_prompt(w)


def test_board_object_state_tracks_the_tasks(w):
    pb.create_task(w, "adam", "Set up the API")
    assert w["locations"]["conference_room"]["objects"]["project_board"]["state"] == \
        "listing 1 tasks (1 todo)"


def test_only_agents_in_the_conference_room_see_the_board(w):
    _adam(w)["current_location"] = "conference_room"
    seen = build_observations(_adam(w), w)
    assert "The project board has no tasks on it yet." in seen
    assert not any("project_board" in o for o in seen)     # no raw object line

    assert not any("project board" in o for o in build_observations(_eve(w), w))


def test_empty_board_status(w):
    lines = pb.status_for_agent(w, "adam")
    assert lines[0].startswith("The project board has no tasks yet")
    assert ("There are no open backend tasks on the board, and backend is Adam Beck's "
            "area: Adam Beck should add the backend work the goal needs.") in lines
    assert lines[-1] == "Adam Beck has no open tasks."


def test_frontend_only_board_tells_adam_backend_is_missing(w):
    # The run from 7:00 am: Eve adds and takes four frontend tasks, Adam none.
    for title in ("Home page", "About page", "Contact form", "Accessibility pass"):
        pb.create_task(w, "eve", title, area="frontend", assign_to_self=True)
    adam = pb.status_for_agent(w, "adam")
    assert "Open tasks per area: backend 0, frontend 4." in adam
    assert any("no open backend tasks" in line for line in adam)
    assert not any("no open frontend tasks" in line for line in pb.status_for_agent(w, "eve"))


def test_status_shows_reviews_dependencies_and_unassigned_work(w):
    pb.create_task(w, "adam", "Set up the API", area="backend", assign_to_self=True)
    pb.create_task(w, "eve", "Contact form", area="frontend", assign_to_self=True,
                   depends_on=["T-1"])
    pb.create_task(w, "eve", "Email service", area="backend")
    pb.create_task(w, "eve", "Home page", area="frontend", assign_to_self=True)
    pb.update_status(w, "eve", "T-4", "in_progress")
    pb.update_status(w, "eve", "T-4", "review", note="check mobile layout")

    adam = pb.status_for_agent(w, "adam")
    assert "Unassigned backend tasks: T-3 'Email service'." in adam
    assert "Eve Smith's T-4 'Home page' is waiting for a teammate's review." in adam
    assert "Eve Smith's T-2 'Contact form' is waiting on Adam Beck's T-1." in adam
    assert "Adam Beck's open tasks: T-1 'Set up the API' (todo)." in adam

    pb.approve_task(w, "adam", "T-4")
    assert not any("waiting for a teammate's review" in line
                   for line in pb.status_for_agent(w, "adam"))


def test_the_board_shows_who_has_how_much(w):
    _board_api_pages_and_db(w)
    assert "Open tasks per person: Adam Beck 2, Eve Smith 3." in pb.status_for_agent(w, "adam")


def test_eve_is_told_plainly_that_nothing_can_start_and_how_to_unblock(w):
    _board_pages_wait_on_api(w)
    lines = pb.status_for_agent(w, "eve")
    assert "Eve Smith's T-2 waits on T-1, which is not done yet, so it can't start." in lines
    assert lines[-1] == (
        "None of Eve Smith's tasks can start until T-1 is done. If one of them doesn't really "
        "need T-1 (a page can be built against mock data), Eve Smith can drop that dependency "
        "with edit_task and start it.")


def test_task_facts_spell_out_what_in_progress_means(w):
    _board_pages_wait_on_api(w)
    assert pb.task_facts(w, ["T-2"]) == [
        "T-2 'Design and implement homepage layout' (Eve Smith's) is todo: not started. "
        "It can't start until T-1 is done.",
        "T-1 'Set up backend API for company website' (Adam Beck's) is in progress since "
        "7:10 am: not in review, not approved, not done."]


def test_the_task_id_line_lists_only_real_ids(w):
    line = pb.task_id_line(w)
    assert "no tasks yet" in line and not TASK_ID.search(line)

    pb.create_task(w, "adam", "Auth endpoints", area="backend", assign_to_self=True)
    pb.create_task(w, "eve", "Homepage", area="frontend", assign_to_self=True)
    assert pb.task_id_line(w) == "The only tasks on the board are T-1, T-2; there is no other task id."


def test_a_short_title_is_left_alone(w):
    _board_api_pages_and_db(w)
    assert pb.fix_task_refs(w, "work on T-2 'homepage layout'") == "work on T-2 'homepage layout'"


@pytest.mark.parametrize("activity,board", [
    ("updating task statuses based on recent progress", True),
    ("assign T-4 to Adam", True),
    ("update T-1", True),
    ("mark T-1 as completed and move it to the done column from 9:20 am to 9:30 am", True),
    ("start T-3 'Design and implement about page layout' from 1:00 pm to 1:10 pm", True),
    ("move it to the review column", True),
    ("Move to the project_board to update tasks", True),     # the object's name, as observed
    ("work on T-1 'Set up the API'", False),
    ("add tests for T-1", False),
    ("start working on T-3", False),
    ("discuss T-1 with Eve", False),
    ("move to office_1", False),
])
def test_what_counts_as_board_work(activity, board):
    assert pb.mentions_board_explicitly(activity) is board


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


def test_a_start_step_for_a_waiting_task_says_why_it_cant_start(w):
    _board_pages_wait_on_api(w)
    out = pb.reconcile_activity(
        w, "eve", "start T-2 (move it to in progress) on the project board from 7:20 am to 7:30 am")
    assert out.startswith("check on T-2: it waits on T-1, which isn't done, so it can't start yet")
    assert out.endswith("on the project board from 7:20 am to 7:30 am")


def test_a_teammates_start_step_is_turned_into_a_check(w):
    _board_api_and_pages(w)
    step = "start T-3 (move it to in progress) on the project board from 8:00 am to 8:10 am"
    assert pb.reconcile_activity(w, "adam", step).startswith(
        "check on T-3 (Eve Smith's task: only they move it on the board)")


def test_approving_a_task_not_in_review_becomes_a_check(w):
    _board_api_pages_and_db(w)
    step = "review and approve T-1 'Backend data structure' with Adam Beck on the project board"
    assert pb.reconcile_activity(w, "eve", step) == (
        "check on T-1 (it is in progress, not in review, so there is nothing to approve yet) "
        "with Adam Beck on the project board")
    pb.get_task(w, "T-1")["status"] = "review"
    assert pb.reconcile_activity(w, "eve", step) == step              # now there is


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


def test_promise_guidance_respects_dependencies(w):
    _board_pages_wait_on_api(w)
    steps = pb.promised_task_steps(w, w["agents"]["eve"], ["I will start with T-2 next"])
    assert steps == [
        "T-2 waits on T-1, which is not done, so T-2 can't start yet. If T-2 doesn't really "
        "need T-1, drop that dependency with edit_task and then start T-2; otherwise leave "
        "T-2 in todo."]


def test_a_promise_naming_several_tasks_starts_only_one(w):
    for title in ("Homepage", "About page", "Contact form"):
        pb.create_task(w, "eve", title, area="frontend", assign_to_self=True)
    steps = pb.promised_task_steps(w, w["agents"]["eve"], ["I will start on T-1, T-2, and T-3 now"])
    assert steps == [
        "The promise is about T-1: move T-1 to in progress (not another task).",
        "T-2 comes after that: one task at a time, so leave it in todo for now.",
        "T-3 comes after that: one task at a time, so leave it in todo for now."]


def test_board_prompt_carries_this_board_and_this_agent(w):
    adam = w["agents"]["adam"]
    prompt = pb.build_prompt(adam, w, "check the project board", "summary", "")
    assert "Adam Beck's area: backend" in prompt and "Teammates: Eve Smith" in prompt
    assert "about to: check the project board" in prompt
    assert "no open backend tasks" in prompt                   # what stands out for Adam

    pb.create_task(w, "eve", "Home page", area="frontend", assign_to_self=True)
    pb.update_status(w, "eve", "T-1", "in_progress")
    prompt = pb.build_prompt(adam, w, "check the project board", "summary", "")
    assert pb.board_text_for_prompt(w) in prompt
    for line in pb.status_for_agent(w, "adam"):
        assert line in prompt


def test_progress_context_uses_work_memories_not_board_lines(fake_llm, isolated_data, world):
    adam = world["agents"]["adam"]
    stream = MemoryStream("adam")
    stream.add_observations([
        "Adam Beck is currently writing the API endpoint code.",
        "Adam Beck is currently testing the API functionality.",
        "Task T-1 on the project board, 'Set up the API' (backend, high priority), is todo.",
    ])
    retriever = MemoryRetriever("adam", stream)
    pb.create_task(world, "adam", "Set up the API", assign_to_self=True)

    context = pb._progress_context(adam, world, retriever)
    assert "writing the API endpoint code" in context
    assert "testing the API functionality" in context
    assert "project board" not in context

    prompt = pb.build_prompt(adam, world, "update my tasks", "summary", "", context)
    assert "What Adam Beck has been doing lately:" in prompt
    assert "approve_task" in prompt


def test_progress_context_is_empty_without_open_tasks(fake_llm, isolated_data, world):
    retriever = MemoryRetriever("adam", MemoryStream("adam"))
    assert pb._progress_context(world["agents"]["adam"], world, retriever) == ""


def test_schema_only_offers_existing_task_ids(w):
    assert pb.action_schema(w)["properties"]["actions"]["items"]["properties"]["task_id"] \
        == {"type": "string"}
    pb.create_task(w, "adam", "Set up the API")
    assert pb.action_schema(w)["properties"]["actions"]["items"]["properties"]["task_id"] \
        == {"type": "string", "enum": ["T-1"]}


def test_actions_come_before_the_reason():
    schema = pb.action_schema({"project": {"tasks": []}, "agents": {"eve": {}}})
    assert list(schema["properties"]) == ["actions", "reason"]


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


def test_a_reply_cut_off_in_its_reason_still_applies_its_actions(fake_llm, isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if isinstance(format, dict) and "actions" in format["properties"]:
            return ('{"actions": [{"action": "create_task", "title": "Homepage", "area": '
                    '"frontend", "assign_to_self": true, "done_when": "the page renders"}], "reason": "Eve Smith is currently at '
                    'the project board and has promised to')
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    eve = w["agents"]["eve"]
    done = pb.interact(eve, w, "check the project board", planner=Planner(eve))
    assert done == ["Eve Smith added task T-1 'Homepage' (frontend, medium priority) "
                    "to the project board and took it on."]


def test_repeating_a_status_is_skipped_quietly(fake_llm, isolated_data, w, monkeypatch):
    pb.create_task(w, "adam", "API", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")

    def resend(prompt, options=None, format=None):
        if isinstance(format, dict) and "actions" in format["properties"]:
            return json.dumps({"reason": "", "actions": [
                {"action": "update_status", "task_id": "T-1", "status": "in_progress"}]})
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", resend)
    stream = MemoryStream("adam")
    before = len(stream.memories)
    adam = w["agents"]["adam"]
    assert pb.interact(adam, w, "report progress on T-1", planner=Planner(adam, stream)) == []
    assert len(stream.memories) == before                      # nothing to remember


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
