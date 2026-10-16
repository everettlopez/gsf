"""
Each agent works on their own tasks (the run where Adam worked on Eve's):

At 7:15 Adam offered to "split up" Eve's frontend tasks T-2, T-3 and T-4 and
"committed" to update the board on them; at 7:20 he committed to "ensure T-2
is stable and meets the review criteria". His replan then wrote "start T-2" and
"work on T-2 'Implement and secure user authentication endpoints'", the title
fix of the round before turned that into Eve's home page, and he spent the
morning sketching her wireframe.

The rule now, everywhere (seeds, board, plans, conversations, promises): a
task belongs to its assignee. A teammate can review it, ask about it, or offer
to work on it together with the owner, who decides. Helping means pairing.

Also here: reflection fires on the paper's importance threshold.
"""

import copy

import pytest

import llm
import agent as ag
import main
import project_board as pb
from conftest import fake_generate
from conversation import _board_context
import persona
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


def _run_board(w):
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


# --- plans ---------------------------------------------------------------------
@pytest.mark.parametrize("block,expected", [
    # The wrong id: the work described stays, the id goes.
    ("work on T-2 'Implement and secure user authentication endpoints' from 8:10 am to 9:30 am",
     "work on 'Implement and secure user authentication endpoints' from 8:10 am to 9:30 am"),
    # Eve's task as Adam's own work: Adam's own work instead, same time.
    ("work on T-2 'Design and implement the home page layout' from 8:10 am to 9:30 am",
     "keep working on T-1 'Design and implement the backend API for the company website' "
     "from 8:10 am to 9:30 am"),
    # Board steps for Eve's task: only she moves it.
    ("start T-2 (move it to in progress) on the project board from 8:00 am to 8:10 am", None),
    ("update the project board with progress on T-2 from 9:30 am to 9:45 am", None),
    # Own and teammate's: the teammate's id goes.
    ("report progress on T-1 and T-2 on the project board from 9:30 am to 9:40 am",
     "report progress on T-1 on the project board from 9:30 am to 9:40 am"),
    # Allowed: pairing with the owner, reviewing.
    ("pair with Eve Smith on T-2 from 10:00 am to 11:00 am",
     "pair with Eve Smith on T-2 from 10:00 am to 11:00 am"),
    ("help Eve with T-2 from 10:00 am to 11:00 am", "help Eve with T-2 from 10:00 am to 11:00 am"),
    ("review and approve T-2 on the project board from 3:00 pm to 3:15 pm",
     "review and approve T-2 on the project board from 3:00 pm to 3:15 pm"),
    # Naming Eve isn't enough: working on her task beside her is still doing it alone.
    ("work on T-2 in the office near Eve from 10:00 am to 11:00 am",
     "keep working on T-1 'Design and implement the backend API for the company website' "
     "from 10:00 am to 11:00 am"),
])
def test_plans_only_work_on_own_tasks(w, block, expected):
    _run_board(w)
    adam = w["agents"]["adam"]
    assert Planner(adam).own_tasks_only(block, adam, w) == expected


def test_the_owner_plans_her_own_task_as_usual(w):
    _run_board(w)
    eve = w["agents"]["eve"]
    block = "work on T-2 'Design and implement the home page layout' from 8:10 am to 9:30 am"
    assert Planner(eve).own_tasks_only(block, eve, w) == block


def test_adams_replan_from_the_run(w):
    """The 7:25 replan from the log, through create_chunks: no T-2 left in it."""
    _run_board(w)
    adam = w["agents"]["adam"]
    chunks = Planner(adam).create_chunks([
        "review and update any pending backend tasks from 7:25 am to 8:00 am",
        "start T-2 (move it to in progress) on the project board from 8:00 am to 8:10 am",
        "work on T-2 'Implement and secure user authentication endpoints' from 8:10 am to 9:30 am",
        "update the project board with progress on T-2 from 9:30 am to 9:45 am",
        "review documentation and finalize deployment configuration from 9:45 am to 12:00 pm",
    ], adam, w)
    assert not any("T-2" in c["activity"] for c in chunks)
    assert [c["start_time"] for c in chunks] == ["7:25 am", "8:10 am", "9:45 am"]


def test_finer_steps_are_held_to_the_same_rule(fake_llm, isolated_data, w, monkeypatch):
    _run_board(w)

    def gen(prompt, options=None, format=None):
        if "Break it into actions" in prompt:
            return ("1. 8:10 am: Sketch the T-2 home page wireframe\n"
                    "2. 8:25 am: Define the API routes\n"
                    "3. 8:40 am: Move T-2 to review on the project board")
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("adam", w)
    hour = {"activity": "8:10 am: work on the backend", "location": "office_1",
            "start_time": "8:10 am", "duration": 60, "target_agent": None}
    steps = ag.expand_minutes(rt, hour, w)
    assert [s["activity"] for s in steps] == [
        "8:10 am: keep working on T-1 'Design and implement the backend API for the company website'",
        "8:25 am: Define the API routes"]
    assert all(s["start_time"] for s in steps)


def test_the_planning_prompt_says_whose_tasks_are_whose(fake_llm, isolated_data, w):
    _run_board(w)
    adam = w["agents"]["adam"]
    guidance = " ".join(Planner(adam).task_guidance(adam, w).split())
    assert "Adam Beck's own tasks: T-1." in guidance
    assert "Eve Smith's tasks: T-2, T-3, T-4." in guidance
    assert "Work blocks are for Adam Beck's own tasks." in guidance


def test_the_plan_example_is_in_the_agents_own_area(fake_llm, isolated_data, w):
    adam = w["agents"]["adam"]
    pb.get_project(w)["tasks"].clear()
    guidance = Planner(adam).task_guidance(adam, w)
    assert "work on the website's backend" in guidance and "navigation bar" not in guidance


def test_a_teammates_start_step_is_turned_into_a_check(w):
    _run_board(w)
    step = "start T-3 (move it to in progress) on the project board from 8:00 am to 8:10 am"
    assert pb.reconcile_activity(w, "adam", step).startswith(
        "check on T-3 (Eve Smith's task: only they move it on the board)")


# --- promises --------------------------------------------------------------------
@pytest.mark.parametrize("commitment,kept", [
    ("Adam Beck will update the board once there is progress on T-2, T-3, and T-4", ""),
    ("I will ensure T-2 is stable and meets the review criteria so we can move it to review", ""),
    ("I will pair with Eve on T-2 at 10 am", "I will pair with Eve on T-2 at 10 am"),
    ("I will help Eve Smith with T-2 after lunch", "I will help Eve Smith with T-2 after lunch"),
    ("I will review T-2 with Eve once it is in review", "I will review T-2 with Eve once it is in review"),
    ("I will move T-1 to review and check on T-2", "I will move T-1 to review"),
    ("I will finish T-1", "I will finish T-1"),
    # Cutting just the id out left "I will ensure is properly set up" and
    # "keep T-6 in todo until is in review" in the run log.
    ("I will make sure T-2 is properly set up so we can move it to in progress, "
     "and I'll check in on T-1 later today", "I'll check in on T-1 later today"),
    ("I will start T-1 as soon as T-2 moves to review", "I will start T-1 as soon as T-2 moves to review"),
    ("I will keep T-1 in todo until T-2 is in review", "I will keep T-1 in todo until T-2 is in review"),
])
def test_promises_about_a_teammates_task(w, commitment, kept):
    _run_board(w)
    adam = w["agents"]["adam"]
    assert Planner(adam).own_commitment(commitment, adam, w) == kept


def test_the_conversation_from_the_run_leaves_adam_no_promise_about_eves_tasks(
        fake_llm, isolated_data, w, monkeypatch):
    _run_board(w)
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
    assert "moving it on the board or doing it alone is the owner's job" in prompts[0]


# --- conversations ---------------------------------------------------------------
def test_speakers_know_whose_tasks_are_whose(fake_llm, isolated_data, w):
    _run_board(w)
    adam, eve = ag.Agent.load("adam", w), ag.Agent.load("eve", w)
    text = " ".join(_board_context(adam, eve, w, "T-2").split())
    assert "Eve Smith's tasks: T-2, T-3, T-4." in text
    assert "offering to work on one together, which Eve Smith can accept or turn down" in text


def test_the_react_prompt_says_whose_tasks_are_whose(fake_llm, isolated_data, w):
    _run_board(w)
    adam = w["agents"]["adam"]
    rules = " ".join(Planner(adam)._board_rules(adam, w).split())
    assert "Each task belongs to its assignee" in rules


# --- the board -------------------------------------------------------------------
def test_a_task_in_another_area_is_left_for_its_owner(w):
    _run_board(w)
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
    _run_board(w)
    pb.create_task(w, "adam", "Deploy the API", area="backend")                   # T-5
    adam = w["agents"]["adam"]
    with pytest.raises(pb.BoardError, match="backend work isn't Eve Smith's area"):
        pb.apply_action(w, adam, {"action": "assign_task", "task_id": "T-5", "assignee": "Eve Smith",
                                  "note": "Can you deploy it?"})
    pb.apply_action(w, adam, {"action": "assign_task", "task_id": "T-5", "assignee": "Adam Beck"})
    assert pb.get_task(w, "T-5")["assignee"] == "adam"


def test_general_work_is_anyones(w):
    _run_board(w)
    line = pb.apply_action(w, w["agents"]["adam"], {"action": "create_task", "title": "Write the README",
                                                    "area": "general", "assign_to_self": True})
    assert "took it on" in line


# --- the seeded identity -----------------------------------------------------------
@pytest.mark.parametrize("agent_id", ["adam", "eve"])
def test_profiles_say_tasks_belong_to_their_assignee(agent_id):
    profile = " ".join(persona.profile(agent_id).split())
    assert "update their status" not in profile           # "anyone can ... update their status"
    assert "A task belongs to the person it is assigned to" in profile
    assert "offers to work on it together with" in profile


def test_seeds_say_tasks_belong_to_their_assignee():
    from test_simulation_smoke import _seed_module
    seeds = _seed_module()
    for memories in (seeds.adam_seed_memories, seeds.eve_seed_memories):
        text = " ".join(memories)
        assert "belongs to the person it is assigned to" in text
        assert "take them on, and update their status" not in text


# --- reflection on the paper's trigger -----------------------------------------------
def test_reflection_waits_for_enough_importance(fake_llm, isolated_data, w, monkeypatch):
    rt = ag.Agent.load("adam", w)
    calls = []
    monkeypatch.setattr(rt.reflector, "reflect", lambda: calls.append(1) or [])
    monkeypatch.setattr(ag, "REFLECTION_THRESHOLD", 150)
    rt.stream.add_observations([f"seen thing {i}" for i in range(29)])       # 29 x 5 = 145
    rt.reflect("9:00 am")
    assert calls == []
    rt.stream.add_observations(["one more thing"])                           # 150
    rt.reflect("9:05 am")
    assert calls == [1]
    rt.reflect("9:10 am")                                       # counted: starts over
    assert calls == [1]
    assert rt.importance_since_reflection() == 0
