"""
Fixes for what run.log (30 steps, 7:00-9:25, debug) showed. Each test replays
one moment of that run:

1. Adam claimed T-1 was approved (7:05) and done (9:05) when it was in progress,
   acted on the claim at the board, and left notes saying "T-1 is complete".
2. Every frontend task waited on T-1, so Eve could start nothing all morning,
   while the promise and stale-step guidance told her to start T-2 anyway.
3. A replan named tasks T-5, T-6, T-7 that weren't on the board, and plans
   included "a designer" and "a backend engineer" who don't exist.
4. "mark T-1 as completed and move it to the done column" happened in an office.
5. Nine plain board visits in one replan, a duplicate start, a 0-minute block.
6. Three refusals for the same mistake in one visit.
7. Every choose_location call (~14s) decided "stay in your own office".
"""

import copy
import importlib.util
import json

import pytest

import clock
import llm
import agent as ag
import main
import project_board as pb
import world as world_module
from conftest import ROOT, fake_generate
from conversation import run_conversation
from plan import Planner, future_part


@pytest.fixture
def w(world):
    pb.clear_workload_changed()
    pb.clear_commitments()
    main.clear_talks()
    yield copy.deepcopy(world)
    pb.clear_workload_changed()
    pb.clear_commitments()
    main.clear_talks()


def _run_log_board(w):
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


# --- 1. talk, notes and plans follow the board ----------------------------------
def test_task_facts_spell_out_what_in_progress_means(w):
    _run_log_board(w)
    assert pb.task_facts(w, ["T-2"]) == [
        "T-2 'Design and implement homepage layout' (Eve Smith's) is todo: not started. "
        "It can't start until T-1 is done.",
        "T-1 'Set up backend API for company website' (Adam Beck's) is in progress since "
        "7:10 am: not in review, not approved, not done."]


def test_conversations_see_the_facts_and_that_talking_moves_nothing(
        fake_llm, isolated_data, w, monkeypatch):
    _run_log_board(w)
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
    assert "Talking doesn't change the board." in first
    assert "If a memory (even something Adam Beck said earlier) disagrees with the board, " \
           "the board is right." in first


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
            assert "neither is something they say they already did" in " ".join(prompt.split())
            return json.dumps({"meeting_agreed": False, "location": "office_1", "activity": "",
                               "minutes": 0, "commitments": [
                                   {"person": "Adam Beck", "commitment": "I've approved T-1"},
                                   {"person": "Eve Smith", "commitment": "I will start on T-2"}]})
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    _run_log_board(w)                                   # T-2 exists, so it's kept
    adam, eve = w["agents"]["adam"], w["agents"]["eve"]
    out = Planner(adam).extract_outcomes("Adam: ...\nEve: ...", adam, eve, w)
    assert out["commitments"] == {"Eve Smith": "I will start on T-2"}


@pytest.mark.parametrize("note", [
    "T-2 can now proceed since T-1 is complete. Please confirm when you're ready to start.",
    "T-1 'Set up backend API for company website' has been completed",
    "T-1 is approved",
    "T-1 is in review now",
])
def test_notes_that_misstate_the_board_are_refused(w, note):
    _run_log_board(w)
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
    _run_log_board(w)
    done, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "add_note", "task_id": "T-2", "note": note}])
    assert refused == [] and len(done) == 1


def test_a_true_claim_is_fine(w):
    _run_log_board(w)
    clock.set_time("10:00 am")
    pb.update_status(w, "adam", "T-1", "review", note="check the endpoints")
    pb.approve_task(w, "eve", "T-1")
    pb.update_status(w, "adam", "T-1", "done")
    done, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "add_note", "task_id": "T-2", "note": "T-1 is done, so T-2 can start"}])
    assert refused == []


def test_reaction_prompt_knows_the_board_rules(fake_llm, isolated_data, w, monkeypatch):
    _run_log_board(w)
    prompts = []

    def gen(prompt, options=None, format=None):
        if "DECISION" in prompt:
            prompts.append(" ".join(prompt.split()))
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    adam = w["agents"]["adam"]
    adam["current_location"] = "conference_room"
    Planner(adam).continue_plan_or_react(
        adam, "status", {"activity": "review the board", "location": "conference room"},
        ["Task T-2 on the project board is todo and assigned to Eve Smith, waiting on T-1."], w)
    p = prompts[0]
    assert "a teammate (never the assignee) approves it while it is in review" in p
    assert "T-1 'Set up backend API for company website' (Adam Beck's) is in progress" in p
    assert "written as an action, not a line of dialogue" in p


def test_board_prompt_says_the_board_beats_memory(w):
    _run_log_board(w)
    prompt = " ".join(pb.build_prompt(w["agents"]["adam"], w, "report progress on T-1",
                                      "summary", "").split())
    assert "the board is right" in prompt


# --- 2. dependencies -------------------------------------------------------------
def test_eve_is_told_plainly_that_nothing_can_start_and_how_to_unblock(w):
    _run_log_board(w)
    lines = pb.status_for_agent(w, "eve")
    assert "Eve Smith's T-2 waits on T-1, which is not done yet, so it can't start." in lines
    assert lines[-1] == (
        "None of Eve Smith's tasks can start until T-1 is done. If one of them doesn't really "
        "need T-1 (a page can be built against mock data), Eve Smith can drop that dependency "
        "with edit_task and start it.")


def test_dropping_an_unneeded_dependency_unblocks_the_task(w):
    _run_log_board(w)
    eve = w["agents"]["eve"]
    done, refused = pb.apply_actions(w, eve, [
        {"action": "edit_task", "task_id": "T-2", "depends_on": []},
        {"action": "update_status", "task_id": "T-2", "status": "in_progress"}])
    assert refused == [] and pb.get_task(w, "T-2")["status"] == "in_progress"
    assert all("T-2" not in line for line in pb.waiting_lines(w, "eve"))


def test_promise_guidance_respects_dependencies(w):
    _run_log_board(w)
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


def test_a_start_step_for_a_waiting_task_says_why_it_cant_start(w):
    _run_log_board(w)
    out = pb.reconcile_activity(
        w, "eve", "start T-2 (move it to in progress) on the project board from 7:20 am to 7:30 am")
    assert out.startswith("check on T-2: it waits on T-1, which isn't done, so it can't start yet")
    assert out.endswith("on the project board from 7:20 am to 7:30 am")


def test_dependencies_are_listed_once(w):
    pb.create_task(w, "adam", "API", area="backend")
    t = pb.create_task(w, "eve", "Contact form", area="frontend", depends_on=["T-1", "t-1", "T-1"])
    assert t["depends_on"] == ["T-1"]


def test_board_prompt_explains_when_to_add_a_dependency(w):
    prompt = " ".join(pb.build_prompt(w["agents"]["eve"], w, "check the board", "summary", "").split())
    assert "depends_on lists only tasks whose result this one truly can't start without" in prompt
    assert "a page layout can be built against mock data before the API exists" in prompt


# --- 3. only real tasks and real people in plans ---------------------------------
def test_planning_guidance_names_the_real_tasks(w):
    _run_log_board(w)
    guidance = " ".join(Planner(w["agents"]["adam"]).task_guidance(w["agents"]["adam"], w).split())
    assert "The only tasks on the board are T-1, T-2, T-3, T-4; never name any other task id" in guidance
    assert "Don't plan work on a task that can't start yet because it waits on another" in guidance


def test_planning_guidance_with_an_empty_board_names_no_ids(w):
    guidance = " ".join(Planner(w["agents"]["eve"]).task_guidance(w["agents"]["eve"], w).split())
    assert "There are no tasks on the board yet, so don't name any task ids." in guidance
    assert "T-3" not in guidance                       # the old example made one up


def test_eves_example_work_block_is_a_task_she_can_start(w):
    _run_log_board(w)
    pb.create_task(w, "eve", "Navigation bar", area="frontend", assign_to_self=True)
    guidance = Planner(w["agents"]["eve"]).task_guidance(w["agents"]["eve"], w)
    assert "work on T-5 'Navigation bar'" in guidance  # not T-2, which waits on T-1


@pytest.mark.parametrize("activity,expected", [
    ("work on T-5 'Implement API endpoints for user authentication' from 10:15 am to 12:15 pm",
     "work on 'Implement API endpoints for user authentication' from 10:15 am to 12:15 pm"),
    ("review T-1, T-6 and T-7 with Eve", "review T-1 with Eve"),
    ("work on T-1 'Set up backend API' from 7:30 am to 8:50 am",
     "work on T-1 'Set up backend API' from 7:30 am to 8:50 am"),
])
def test_made_up_task_ids_are_removed_from_plans(w, activity, expected):
    _run_log_board(w)
    assert Planner.drop_unknown_task_ids(activity, w) == expected
    chunk = Planner(w["agents"]["adam"]).create_chunks([activity], w["agents"]["adam"], w)[0]
    assert chunk["activity"] == expected


def test_plans_know_who_the_team_is(fake_llm, isolated_data, w, monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if "Continue the plan as a numbered list" in prompt:
            prompts.append(" ".join(prompt.split()))
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    eve = w["agents"]["eve"]
    Planner(eve).generate_initial_plan(eve, w)
    assert "Eve Smith's only teammates are Adam Beck (backend); there is no one else" in prompts[0]
    assert "designers" not in prompts[0]
    assert "fixed meetings" not in prompts[0]      # meetings come from the agents, not the world


# --- 4. board actions are recognised as board work ------------------------------
@pytest.mark.parametrize("activity,board", [
    ("mark T-1 as completed and move it to the done column from 9:20 am to 9:30 am", True),
    ("start T-3 'Design and implement about page layout' from 1:00 pm to 1:10 pm", True),
    ("move it to the review column", True),
    ("work on T-1 'Set up backend API' from 7:30 am to 8:50 am", False),
    ("start working on T-3", False),
    ("discuss T-1 with Eve", False),
    ("move to office_1", False),
])
def test_board_actions_go_to_the_board(activity, board):
    assert pb.mentions_board_explicitly(activity) is board


# --- 5. fewer board trips --------------------------------------------------------
def _plan(planner, lines, agent, w):
    return planner.create_chunks(lines, agent, w)


EVE_910 = [   # Eve's 9:10 am replan in run.log, before checkpoints
    "update the project board in the conference room from 9:10 am to 9:20 am",
    "work on T-2 'Design and implement homepage layout' from 9:20 am to 9:30 am",
    "update the project board in the conference room from 9:30 am to 9:40 am",
    "work on accessibility improvements for the homepage from 9:40 am to 11:00 am",
    "update the project board in the conference room from 11:00 am to 11:10 am",
    "work on accessibility improvements for the homepage from 11:10 am to 12:00 pm",
    "take a short break from 12:00 pm to 12:15 pm",
    "review and approve a teammate's task on the project board from 12:15 pm to 1:00 pm",
    "start T-3 'Design and implement about page layout' from 1:00 pm to 1:10 pm",
    "update the project board in the conference room from 1:10 pm to 1:20 pm",
    "work on T-3 'Design and implement about page layout' from 1:20 pm to 2:10 pm",
    "report progress on T-3 on the project board from 2:10 pm to 2:20 pm",
    "update the project board in the conference room from 2:20 pm to 2:30 pm",
    "work on T-4 'Implement contact form with validation' from 2:30 pm to 4:00 pm",
    "update the project board in the conference room from 4:00 pm to 4:10 pm",
    "report progress on T-4 on the project board from 4:10 pm to 4:20 pm",
    "update the project board in the conference room from 4:20 pm to 4:30 pm",
    "review documentation and test the frontend build from 4:30 pm to 4:50 pm",
    "update the project board in the conference room from 4:50 pm to 5:00 pm",
    "update the project board in the conference room from 5:00 pm to 5:00 pm",
]


def test_eves_replan_keeps_the_useful_board_visits_only(w):
    _run_log_board(w)
    eve = w["agents"]["eve"]
    p = Planner(eve)
    chunks = p.thin_board_visits(_plan(p, EVE_910, eve, w))
    board = [c["activity"].split(" from ")[0] for c in chunks if c.get("board_work")]
    assert board == [
        "update the project board in the conference room",              # 9:10, first of the day
        "update the project board in the conference room",              # 11:00, 100 min later
        "review and approve a teammate's task on the project board",    # has a purpose
        "start T-3 'Design and implement about page layout'",
        "report progress on T-3 on the project board",
        "report progress on T-4 on the project board",
        "update the project board in the conference room",              # last of the day
    ]
    assert all((c.get("duration") or 0) > 0 for c in chunks)            # 5:00-5:00 is gone


def test_dropped_visit_time_goes_to_the_work_before_it_not_to_a_break(w):
    eve = w["agents"]["eve"]
    p = Planner(eve)
    chunks = p.thin_board_visits(_plan(p, [
        "update the project board in the conference room from 9:00 am to 9:10 am",
        "work on the navigation bar from 9:10 am to 10:00 am",
        "update the project board in the conference room from 10:00 am to 10:10 am",
        "take a short break from 10:10 am to 10:20 am",
        "update the project board in the conference room from 10:20 am to 10:30 am",
        "work on the footer from 10:30 am to 5:00 pm",
        "update the project board before clocking out from 4:50 pm to 5:00 pm",
    ], eve, w))
    acts = [c["activity"] for c in chunks]
    assert "work on the navigation bar from 9:10 am to 10:10 am" in acts
    assert "take a short break from 10:10 am to 10:20 am" in acts      # not stretched
    assert sum(c.get("board_work", False) for c in chunks) == 2          # first and last


def test_the_same_step_for_the_same_task_twice_in_a_row_is_dropped(w):
    _run_log_board(w)
    eve = w["agents"]["eve"]
    p = Planner(eve)
    chunks = p.thin_board_visits(_plan(p, [
        "start T-3 'About page' from 1:00 pm to 1:10 pm",
        "start T-3 (move it to in progress) on the project board from 1:10 pm to 1:20 pm",
        "work on T-3 'About page' from 1:20 pm to 2:10 pm",
        "report progress on T-3 on the project board from 2:10 pm to 2:20 pm",
    ], eve, w))
    assert [c["activity"].split(" ")[0] for c in chunks] == ["start", "work", "report"]


# --- 6. one refusal per mistake --------------------------------------------------
def test_three_approvals_of_tasks_not_in_review_are_one_refusal(w):
    _run_log_board(w)
    done, refused = pb.apply_actions(w, w["agents"]["adam"], [
        {"action": "approve_task", "task_id": t} for t in ("T-2", "T-3", "T-4")])
    assert refused == ["approve_task: T-2 is not in review (same for T-3, T-4)"]


def test_different_mistakes_stay_separate(w):
    _run_log_board(w)
    _, refused = pb.apply_actions(w, w["agents"]["eve"], [
        {"action": "approve_task", "task_id": "T-2"},
        {"action": "update_status", "task_id": "T-3", "status": "in_progress"}])
    assert refused == ["approve_task: T-2 is not in review",
                       "update_status: T-3 is waiting on T-1"]


# --- 7. no LLM call to decide "stay in your office" ------------------------------
@pytest.mark.parametrize("activity,where,expected", [
    ("Outline database models", "office_1", "office_1"),           # solo work, own office
    ("Plan the API structure", "conference_room", "office_1"),      # solo work, go back
    ("take a short break", "break_room", "break_room"),             # already in the right room
    ("take a short break", "office_1", None),                       # elsewhere: let the LLM decide
    ("discuss the API with Eve", "office_1", None),                 # involves a teammate
])
def test_obvious_locations(w, activity, where, expected):
    adam = w["agents"]["adam"]
    adam["current_location"] = where
    assert Planner(adam).obvious_location(activity, adam, w) == expected


def test_apply_chunk_skips_the_location_llm_for_solo_work(fake_llm, isolated_data, w, monkeypatch):
    rt = ag.Agent.load("adam", w)
    rt.agent["current_location"] = "office_1"
    monkeypatch.setattr(rt.planner, "choose_location",
                        lambda *a, **k: pytest.fail("choose_location should not be called"))
    rt.chunk = {"activity": "7:45 am: Outline database models", "location": "office_1",
                "start_time": "7:45 am", "duration": 15, "target_agent": None}
    ag.apply_chunk(rt, w)
    assert rt.agent["current_location"] == "office_1"


# --- carried over from Plan7: no trailing a teammate through solo steps ----------
@pytest.mark.parametrize("step,target", [
    ("9:00 am: Discuss integration requirements", "eve"),
    ("9:40 am: Debug integration issues with Eve", "eve"),
    ("9:10 am: Set up the development environment", None),
    ("9:15 am: Review backend requirements", None),
])
def test_finer_steps_keep_the_teammate_only_when_working_with_them(w, step, target):
    adam = w["agents"]["adam"]
    parent = {"activity": "collaborate with Eve Smith on frontend-backend integration",
              "target_agent": "eve"}
    assert Planner(adam)._chunk_from_activity(step, adam, parent)["target_agent"] == target


# --- reset ---------------------------------------------------------------------
def test_reset_clears_board_notes_and_puts_everyone_back_at_reception(isolated_data, monkeypatch):
    spec = importlib.util.spec_from_file_location("seed_memories",
                                                  ROOT / "scripts" / "seed_memories.py")
    seed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed)

    data = world_module.load_world()
    data["agents"]["adam"]["current_location"] = "office_1"
    data["project"]["notes"] = [{"time": "9:10 am", "agent": "adam", "note": "T-1 is complete"}]
    world_module.save_world(data)

    seed.clear_board()
    after = world_module.load_world()
    assert after["project"]["notes"] == [] and after["project"]["tasks"] == []
    assert {a["current_location"] for a in after["agents"].values()} == {"reception"}
