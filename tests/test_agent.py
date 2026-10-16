"""
agent.py: one agent's step. Following the plan (lazy decomposition, gaps),
deciding whether to react (the filters that drop reactions which only repeat
the plan), acting (where the step happens, board visits), replanning after a
reaction, a talk or a board visit, and reflecting on the paper's trigger.
"""

import pytest

import agent as ag
import llm
import main
import project_board as pb
import world as world_module
from conftest import fake_generate


def short_hours(prompt, options=None, format=None):
    if "Create EXACTLY" in prompt:          # the model returns one hour, whatever we ask
        return "1. 10:05 am: Review progress and adjust task order"
    return fake_generate(prompt, options, format)


EVE_INITIAL = [   # Eve's initial plan in the run
    "clock in and start work at 7:00 am",
    "attend the daily stand-up with the whole team from 8:00 am to 8:15 am",
    "check the project board in the conference room from 8:15 am to 9:00 am",
    "review and update frontend tasks on the project board from 9:00 am to 10:00 am",
    "collaborate with Adam Beck on technical challenges from 10:00 am to 11:00 am",
    "work on implementing a new page layout for the company website from 11:00 am to 12:00 pm",
]


def _plan(p, lines, agent, w):
    return p.create_chunks(lines, agent, w)


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


def _at_board(w, rt):
    rt.agent["current_location"] = "conference_room"
    rt.agent["interacting_with"] = pb.BOARD_OBJECT


def _runtimes(w, room="office_2"):
    adam, eve = ag.Agent.load("adam", w), ag.Agent.load("eve", w)
    for rt in (adam, eve):
        rt.agent["current_location"] = room
    return adam, eve


def _office_chunk(activity, **extra):
    return {"activity": activity, "location": "office_2", "start_time": "11:00 am",
            "duration": 60, "target_agent": None, **extra}


def _replan_eve(w, monkeypatch, prompts=None):
    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            if prompts is not None:
                prompts.append(" ".join(prompt.split()))
            return EVE_REPLAN
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("eve", w)
    rt.broad_chunks = rt.planner.create_chunks(EVE_DAY, rt.agent, w)
    return rt


# Eve's replan in the run: no stand-up, no 10:15 work with Adam.
EVE_REPLAN = ("1. work on the website's homepage layout from 7:25 am to 10:00 am\n"
              "2. work on styling and accessibility improvements for the about page from 10:00 am to 1:00 pm\n"
              "3. take a lunch break in the break room from 1:00 pm to 2:00 pm\n"
              "4. work on the website's navigation bar from 2:00 pm to 5:00 pm")


EVE_DAY = [
    "clock in at reception from 7:00 am to 7:05 am",
    "check the project board in the conference room from 7:05 am to 7:20 am",
    "review and update any frontend tasks from 7:20 am to 8:00 am",
    "attend a stand-up meeting with Adam Beck to discuss progress and next steps from 8:00 am to 8:15 am",
    "work on implementing a new component for the homepage layout from 8:15 am to 10:00 am",
    "collaborate with Adam Beck on integrating the backend API for the contact form from 10:15 am to 11:30 am",
    "take a lunch break in the break room from 1:00 pm to 2:00 pm",
]


def _runtime(world, monkeypatch, prompts):
    def capture(prompt, options=None, format=None):
        if "Revise" in prompt:
            prompts.append(prompt)
            return ("1. work on T-1 'Set up the API' in the office from 9:05 am to 11:00 am\n"
                    "2. update T-1 on the project board from 11:00 am to 11:15 am\n"
                    "3. clock out at 5:00 pm")
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", capture)
    rt = ag.Agent.load("adam", world)
    ag.initial_planning(rt, world)
    return rt


def _revise_returns_board_visit(prompts):
    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            prompts.append(prompt)
            return ("1. update the project board from 11:40 am to 11:50 am\n"
                    "2. work on the home page from 11:50 am to 12:00 pm")
        return fake_generate(prompt, options, format)
    return gen


def _adam_with_backend_tasks(w):
    for title in ("Backend API", "Database schema"):
        pb.create_task(w, "adam", title, area="backend", assign_to_self=True)


def test_gap_after_a_short_hour_decomposition_is_filled(
    fake_llm, isolated_data, world, monkeypatch
):
    monkeypatch.setattr(llm, "generate", short_hours)
    rt = ag.Agent.load("adam", world)
    rt.broad_chunks = [{"activity": "work on T-1 'Set up the API' from 10:05 am to 12:00 pm",
                        "location": "office_1", "start_time": "10:05 am", "duration": 115,
                        "target_agent": None}]

    early = ag.get_chunk_lazy(rt, "10:30 am")
    late = ag.get_chunk_lazy(rt, "11:30 am")
    later = ag.get_chunk_lazy(rt, "11:50 am")

    assert rt.planner.covers(early, "10:30 am")
    assert rt.planner.covers(late, "11:30 am")         # not a stale 10:xx action
    assert rt.planner.covers(later, "11:50 am")
    gap_hours = [k for k in rt.minute_cache if k[0] == "11:05 am"]
    assert len(gap_hours) == 1                          # the gap is expanded once


def test_gap_in_the_broad_plan_is_filled_with_the_task_in_progress(
    fake_llm, isolated_data, w
):
    rt = ag.Agent.load("adam", w)
    pb.create_task(w, "adam", "Set up the API", assign_to_self=True)
    pb.update_status(w, "adam", "T-1", "in_progress")
    rt.broad_chunks = [
        {"activity": "update the project board from 11:15 am to 11:20 am",
         "location": "conference room", "start_time": "11:15 am", "duration": 5,
         "target_agent": None, "board_work": True},
        {"activity": "lunch in the break room from 12:30 pm to 1:30 pm",
         "location": "break room", "start_time": "12:30 pm", "duration": 60,
         "target_agent": None, "board_work": False},
    ]
    chunk = ag.get_chunk_lazy(rt, "11:40 am", w)
    assert rt.planner.covers(chunk, "11:40 am")
    assert "board" not in chunk["activity"]                     # not stuck on the 11:15 step

    gap = ag.gap_block(rt, "11:40 am", w)
    assert gap["activity"] == ("keep working on T-1 'Set up the API' in the office "
                               "from 11:20 am to 12:30 pm")
    assert (gap["start_time"], gap["duration"]) == ("11:20 am", 70)


def test_no_gap_filling_after_the_workday(fake_llm, isolated_data, w):
    rt = ag.Agent.load("adam", w)
    rt.broad_chunks = [{"activity": "clock out at 4:00 pm", "location": "reception",
                        "start_time": "4:00 pm", "duration": 60, "target_agent": None}]
    assert ag.gap_block(rt, "5:10 pm", w) is None


def test_the_gap_filler_names_the_next_task(fake_llm, isolated_data, w):
    pb.get_project(w)["tasks"].clear()
    pb.create_task(w, "adam", "Design and implement the API for the company website",
                   area="backend", assign_to_self=True)
    rt = ag.Agent.load("adam", w)
    rt.broad_chunks = rt.planner.create_chunks([
        "clock in at reception from 7:00 am to 7:05 am",
        "check the project board in the conference room from 7:05 am to 7:20 am",
        "work on the backend from 7:40 am to 10:00 am"], rt.agent, w)
    gap = ag.gap_block(rt, "7:20 am", w)
    assert gap["activity"].startswith(
        "prepare for T-1 'Design and implement the API for the company website'")
    assert not pb.mentions_board_explicitly(gap["activity"])   # it isn't a board visit


def test_initial_plans_get_the_same_cleanup_as_replans(fake_llm, isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if "Continue the plan as a numbered list" in prompt:
            return "\n".join(f"{i + 2}. {line}" for i, line in enumerate(EVE_INITIAL[1:]))
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("eve", w)
    ag.initial_planning(rt, w)
    board = [c for c in rt.broad_chunks if c.get("board_work")]
    assert board and all(c["duration"] <= 15 for c in board)


def test_a_board_block_is_one_step_not_many(fake_llm, isolated_data, w, monkeypatch):
    rt = ag.Agent.load("eve", w)
    calls = []

    def gen(prompt, options=None, format=None):
        if "Break it into actions" in prompt or "Create EXACTLY" in prompt:
            calls.append(prompt)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    p = rt.planner
    # 80 minutes: long enough to have been split into hours, then 5-minute steps
    rt.broad_chunks = _plan(p, ["visit the project board to add frontend tasks from 7:10 am to 8:30 am"],
                            rt.agent, w)
    first = ag.get_chunk_lazy(rt, "7:10 am", w)
    later = ag.get_chunk_lazy(rt, "8:20 am", w)
    assert calls == []                                    # no decomposition LLM calls
    assert first is later and first["board_work"]         # same chunk: one board call


def test_short_chunks_are_not_sent_for_minute_decomposition(fake_llm, isolated_data, w,
                                                            monkeypatch):
    rt = ag.Agent.load("adam", w)
    step = {"activity": "move T-1 to in progress on the project board from 9:00 am to 9:10 am",
            "start_time": "9:00 am", "duration": 10, "target_agent": None, "board_work": True}
    monkeypatch.setattr(llm, "generate", lambda *a, **k: pytest.fail("no LLM call expected"))
    assert ag.expand_minutes(rt, step) == [step]


def test_decomposition_knows_the_task_and_drops_board_steps(fake_llm, isolated_data, w, monkeypatch):
    _board_api_pages_and_db(w)
    prompts = []

    def gen(prompt, options=None, format=None):
        if "Break it into actions" in prompt:
            prompts.append(" ".join(prompt.split()))
            return ("1. 7:45 am: Update task status in project management tool\n"
                    "2. 7:55 am: Sketch the hero section\n"
                    "3. 8:10 am: Update T-2 on the project board\n"
                    "4. 8:20 am: Build the navigation bar")
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("eve", w)
    hour = {"activity": "7:45 am: review and update any frontend tasks", "location": "office_2",
            "start_time": "7:45 am", "duration": 60, "target_agent": None}
    steps = ag.expand_minutes(rt, hour, w)
    assert "Context: Eve Smith is working on T-2 'Design and implement the homepage layout'" in prompts[0]
    assert [(s["start_time"], s["activity"]) for s in steps] == [
        ("7:45 am", "7:45 am: Sketch the hero section"),
        ("8:20 am", "8:20 am: Build the navigation bar")]
    assert [s["duration"] for s in steps] == [35, 25]
    assert not any(s["board_work"] for s in steps)


def test_finer_steps_are_held_to_the_same_rule(fake_llm, isolated_data, w, monkeypatch):
    _board_api_and_pages(w)

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


@pytest.mark.parametrize("reaction,planned,same", [
    ("go to the conference room to check the project board",
     "check the project board in the conference room from 7:05 am to 7:20 am", True),
    ("Ask Eve which fields the contact form sends", "Outline the API structure", False),
    ("Go to the conference room to review the project board with Eve Smith",
     "Review pending backend tasks", False),
])
def test_is_the_plan(reaction, planned, same):
    assert ag._is_the_plan(reaction, planned) is same


def test_reacting_with_the_planned_step_is_ignored(fake_llm, isolated_data, w):
    rt = ag.Agent.load("adam", w)
    rt.chunk = {"activity": "check the project board in the conference room from 7:05 am to 7:20 am",
                "board_work": True}
    assert ag.pointless_reaction(rt, "go to the conference room to check the project board",
                                   "7:05 am", w) == "that is the plan already"


def test_going_to_the_board_while_the_plan_is_a_board_visit_is_the_plan(fake_llm, isolated_data, w):
    rt = ag.Agent.load("adam", w)
    rt.agent["current_location"] = "reception"
    rt.chunk = {"activity": "review any new backend tasks added to the project board from "
                            "7:05 am to 7:20 am", "board_work": True}
    assert ag.pointless_reaction(rt, "go to the conference room to review the project board",
                                   "7:05 am", w) == "that is the plan already"


@pytest.mark.parametrize("reaction,ignored", [
    ("continue working on the homepage layout", True),
    ("keep going with the API endpoints", True),
    ("continue the discussion with Eve about the API", False),     # involves a teammate
])
def test_continue_reactions_are_the_plan(fake_llm, isolated_data, w, reaction, ignored):
    rt = ag.Agent.load("adam", w)
    rt.chunk = {"activity": "7:30 am: Sketch the homepage layout"}
    assert bool(ag.pointless_reaction(rt, reaction, "7:35 am", w)) is ignored


def test_no_board_reaction_while_at_the_board(fake_llm, isolated_data, w):
    rt = ag.Agent.load("eve", w)
    _at_board(w, rt)
    rt.chunk = {"activity": "7:25 am: Add new frontend tasks", "board_work": True}
    assert ag.pointless_reaction(rt, "Move to the project_board to update tasks",
                                   "7:35 am", w) == "already at the board"


def test_no_board_reaction_right_after_using_the_board(fake_llm, isolated_data, w):
    rt = ag.Agent.load("adam", w)
    rt.chunk = {"activity": "8:30 am: Summarize key points", "board_work": False}
    rt.last_board_at = rt.planner._parse_time("8:10 am")
    assert ag.pointless_reaction(rt, "Move to the project_board to update task T-4",
                                   "8:35 am", w) == "used the board at 8:10 am"
    assert ag.pointless_reaction(rt, "Move to the project_board to update task T-4",
                                   "8:45 am", w) == ""    # 35 minutes later: fine


def test_the_same_reaction_twice_is_ignored(fake_llm, isolated_data, w):
    rt = ag.Agent.load("eve", w)
    reaction = "Move to the project_board to update task T-2 and check on T-4's status"
    rt.chunk = {"activity": f"8:30 am: {reaction}", "is_reaction": True, "board_work": True}
    assert ag.pointless_reaction(rt, reaction, "8:35 am", w) == "already doing this"


def test_a_real_reaction_still_goes_through(fake_llm, isolated_data, w):
    rt = ag.Agent.load("adam", w)
    rt.chunk = {"activity": "9:10 am: Outline the core functionality of the API"}
    assert ag.pointless_reaction(rt, "Ask Eve which fields the contact form sends",
                                   "9:15 am", w) == ""


@pytest.mark.parametrize("reaction,ignored", [
    ("check the time_clock in the reception", True),
    ("go to the reception to check the time_clock with Adam Beck", True),
    ("look at the time clock", True),
    ("fix the broken coffee_machine", False),          # does something
    ("use the computer to write the API endpoints", False),
    ("talk with Eve Smith about the event schedule", False),   # no object
])
def test_reactions_only_about_an_object_are_ignored(w, reaction, ignored):
    assert bool(ag.object_only(reaction, w["agents"]["eve"], w)) is ignored


def test_board_use_is_remembered_only_when_using_the_board(fake_llm, isolated_data, w):
    rt = ag.Agent.load("eve", w)
    rt.agent["interacting_with"] = "desk"
    ag.note_board_use(rt, "7:05 am")
    assert rt.last_board_at is None
    rt.agent["interacting_with"] = pb.BOARD_OBJECT
    ag.note_board_use(rt, "7:10 am")
    assert rt.planner._fmt(rt.last_board_at) == "7:10 am"


def test_decide_phase_drops_a_pointless_reaction(fake_llm, isolated_data, w, monkeypatch):
    rt = ag.Agent.load("eve", w)
    _at_board(w, rt)
    rt.prev_company = set()

    def gen(prompt, options=None, format=None):
        if "DECISION" in prompt:
            return "DECISION: 2\nREACTION: Move to the project_board to update tasks"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "7:25 am: Add new frontend tasks based on current priorities",
        "location": "conference room", "start_time": "7:25 am", "duration": 15,
        "target_agent": None, "board_work": True})
    rt.new_obs = ["Task T-2 on the project board is todo and assigned to Eve Smith."]
    rt.decide(8, "7:35 am", w)
    assert (rt.decision, rt.reaction) == ("1", "")
    assert rt.last_react_step < 0                         # and no cooldown started


def test_no_replan_for_a_reaction_that_is_the_plan(fake_llm, isolated_data, w, monkeypatch):
    replans = []

    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            replans.append(prompt)
        if "DECISION" in prompt:
            return "DECISION: 2\nREACTION: go to the conference room to check the project board"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("adam", w)
    rt.prev_company = set()
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "check the project board in the conference room from 7:05 am to 7:20 am",
        "location": "conference room", "start_time": "7:05 am", "duration": 15,
        "target_agent": None, "board_work": True})
    rt.new_obs = ["Eve Smith is currently clocking in at reception."]
    rt.decide(2, "7:05 am", w)
    assert rt.decision == "1" and replans == []


def test_the_start_reaction_from_the_run_is_ignored(fake_llm, isolated_data, w, monkeypatch):
    _board_api_pages_and_db(w)

    def gen(prompt, options=None, format=None):
        if "DECISION" in prompt:
            return "DECISION: 2\nREACTION: Start working on task T-5 'Set up database for contact form'"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("adam", w)
    rt.agent["current_location"] = "conference_room"
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "review the project board and any new backend tasks added from 7:25 am to 7:40 am",
        "location": "conference room", "start_time": "7:25 am", "duration": 15,
        "target_agent": None, "board_work": True})
    rt.new_obs = ["Task T-5 on the project board, 'Set up database for contact form', is todo."]
    rt.decide(7, "7:30 am", w)
    assert rt.decision == "1"


def test_no_conversation_about_the_time_clock(fake_llm, isolated_data, w, monkeypatch):
    asked = []

    def gen(prompt, options=None, format=None):
        if "DECISION" in prompt:
            asked.append("DECISION")
            return "DECISION: 2\nREACTION: go to the reception to check the time_clock with Adam Beck"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("eve", w)
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "check the project board in the conference room from 7:05 am to 7:20 am",
        "location": "conference room", "start_time": "7:05 am", "duration": 15,
        "target_agent": None, "board_work": True})
    rt.new_obs = ["Adam Beck is interacting with the time_clock.", "The time_clock in the reception is off."]
    rt.decide(2, "7:05 am", w)
    assert rt.decision == "1" and asked == ["DECISION"]


def test_a_reaction_about_a_teammates_task_goes_to_them(fake_llm, isolated_data, w, monkeypatch):
    _board_api_pages_and_db(w)

    def gen(prompt, options=None, format=None):
        if "DECISION" in prompt:
            return "DECISION: 2\nREACTION: Check on T-2 progress and confirm if it affects T-1 or T-5."
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("adam", w)
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "work on T-1 'Design and implement the API for the company website'",
        "location": "office_1", "start_time": "8:05 am", "duration": 60,
        "target_agent": None, "board_work": False})
    rt.new_obs = ["Task T-2 on the project board ... The latest note on it, from Eve Smith, says: ..."]
    rt.decide(14, "8:05 am", w)
    assert rt.decision == "2"
    assert rt.reaction.startswith("talk with Eve Smith about T-2: ")
    assert rt.planner.extract_target_agent(rt.reaction, rt.agent, w) == "eve"


@pytest.mark.parametrize("reaction", [
    "ask Eve Smith how T-2 is going",                  # names her already
    "approve T-2 on the project board",                # board work
    "keep working on T-1",                             # own task
])
def test_other_reactions_are_left_alone(w, reaction):
    _board_api_pages_and_db(w)

    class RT:
        agent = w["agents"]["adam"]

    assert ag.aim_at_owner(RT, reaction, w) == reaction


def test_planned_collaboration_with_a_teammate_here_starts_a_conversation(
        fake_llm, isolated_data, w, monkeypatch):
    adam, _ = _runtimes(w)
    adam.last_react_step = 50                       # mid-cooldown, as at 11:25 in the log
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: _office_chunk(
        "11:00 am: discuss integration requirements with Eve", target_agent="eve"))

    adam.decide(52, "11:05 am", w)
    assert adam.decision == "2"
    assert adam.reaction == "talk with Eve Smith about: discuss integration requirements with Eve"


def test_planned_collaboration_waits_for_the_conversation_cooldown(
        fake_llm, isolated_data, w, monkeypatch):
    adam, eve = _runtimes(w)
    main.remember_conversation(adam, eve, "10:45 am", "integration")
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: _office_chunk(
        "11:00 am: discuss integration requirements with Eve", target_agent="eve"))

    adam.decide(52, "11:05 am", w)
    assert adam.decision == "1"


def test_a_teammate_walking_in_is_considered_even_during_the_react_cooldown(
        fake_llm, isolated_data, w, monkeypatch):
    monkeypatch.setattr(ag, "REACT_COOLDOWN_STEPS", 6)   # the window this test is about
    adam, _ = _runtimes(w, room="conference_room")
    adam.last_react_step = 52                       # reacted at 11:15
    adam.prev_company = set()                       # Eve wasn't here last step
    asked = []

    def gen(prompt, options=None, format=None):
        if "DECISION" in prompt:
            asked.append(prompt)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    monkeypatch.setattr(ag, "get_chunk_lazy",
                        lambda rt, t, world=None: _office_chunk("11:25 am: test API endpoints"))
    adam.new_obs = ["Eve Smith is currently reporting progress on T-4."]

    adam.decide(54, "11:25 am", w)
    assert len(asked) == 1                          # the cooldown didn't swallow it

    asked.clear()
    adam.decide(55, "11:30 am", w)      # she's still here: cooldown applies
    assert asked == []


def test_cooldown_is_the_reactions_own_ten_minutes():
    assert ag.REACT_COOLDOWN_STEPS * main.STEP_MINUTES == 10


def test_before_talking_the_step_still_goes_to_the_teammate(fake_llm, isolated_data, w, monkeypatch):
    eve = ag.Agent.load("eve", w)
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "7:30 am: Start collaborating with Adam Beck",
        "location": "conference room", "start_time": "7:30 am", "duration": 10,
        "target_agent": "adam"})
    eve.decide(8, "7:30 am", w)
    assert eve.chunk["target_agent"] == "adam"


def test_after_talking_a_planned_collaboration_step_happens_in_your_own_office(
        fake_llm, isolated_data, w, monkeypatch):
    eve, adam = ag.Agent.load("eve", w), ag.Agent.load("adam", w)
    eve.agent["current_location"] = "conference_room"
    adam.agent["current_location"] = "office_1"
    main.remember_conversation(eve, adam, "7:20 am", "T-3 and T-4")
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "7:30 am: Start collaborating with Adam Beck on technical challenges",
        "location": "conference room", "start_time": "7:30 am", "duration": 10,
        "target_agent": "adam"})

    eve.decide(8, "7:30 am", w)
    assert eve.chunk["target_agent"] is None
    assert eve.chunk["location"] == "office_2" and eve.chunk["pinned"]
    ag.apply_chunk(eve, w)
    assert eve.agent["current_location"] == "office_2"        # not Adam's office


def test_after_the_planned_talk_the_agent_goes_back_to_their_task(fake_llm, isolated_data, w, monkeypatch):
    _board_api_pages_and_db(w)
    eve, adam = ag.Agent.load("eve", w), ag.Agent.load("adam", w)
    main.remember_conversation(eve, adam, "8:05 am", "the stand-up")
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "attend a stand-up meeting with Adam Beck from 8:00 am to 8:15 am",
        "location": "conference room", "start_time": "8:00 am", "duration": 15,
        "target_agent": "adam"})
    eve.decide(15, "8:10 am", w)
    assert eve.chunk["activity"] == "8:00 am: keep working on T-2 'Design and implement the homepage layout'"
    assert eve.chunk["location"] == "office_2" and eve.chunk["target_agent"] is None


def test_a_move_only_reaction_relocates_without_a_cooldown(fake_llm, isolated_data, w, monkeypatch):
    adam, _ = _runtimes(w, room="office_2")
    w["agents"]["eve"]["current_location"] = "conference_room"
    adam.prev_company = set()

    def gen(prompt, options=None, format=None):
        if "DECISION" in prompt:
            return ("DECISION: 2\nREACTION: Move to office_1 to set up the development "
                    "environment as planned")
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    monkeypatch.setattr(ag, "get_chunk_lazy",
                        lambda rt, t, world=None: rt.override or _office_chunk(
                            "11:10 am: set up development environment"))
    adam.new_obs = ["The office_2 is currently quiet."]

    adam.decide(52, "11:15 am", w)
    assert (adam.decision, adam.reaction) == ("1", "")
    assert adam.last_react_step < 0                 # no cooldown started
    assert adam.chunk["location"] == "office_1" and adam.chunk["pinned"]
    assert adam.chunk["activity"] == "11:10 am: set up development environment"


@pytest.mark.parametrize("reaction,expected", [
    ("Move to office_1 to continue debugging integration issues", "office_1"),
    ("head back to my office to keep coding", "office_1"),
    ("go to the conference room to update T-1", None),    # board work: a real reaction
    ("Move to office_2 to talk with Eve", None),          # about a teammate: a real reaction
    ("Move to office_2 to check something", None),        # already there
    ("Let's quickly check in on T-4 progress", None),     # not a move
    ("ask Eve about the API", None),
])
def test_relocation_target(w, reaction, expected):
    adam = w["agents"]["adam"]
    adam["current_location"] = "office_2"
    assert ag.relocation_target(reaction, adam, w) == expected


@pytest.mark.parametrize("reaction,current,expected", [
    ("Walk to the conference room to update the project board",
     "Review the day's tasks and priorities", None),
    ("Move to the conference room to update the project board",
     "Begin designing the homepage layout", None),
    ("Move to office_1 to write the deployment docs", "Debug integration issues", None),
    ("Move to office_1 to set up the development environment as planned",
     "Set up the development environment", "office_1"),
    ("Move to office_1 to continue debugging integration issues",
     "Debug integration issues", "office_1"),
    ("Move to office_1", "Debug integration issues", "office_1"),
])
def test_relocation_only_when_the_purpose_is_the_current_work(w, reaction, current, expected):
    adam = w["agents"]["adam"]
    adam["current_location"] = "office_2"
    assert ag.relocation_target(reaction, adam, w, current) == expected


def test_board_activity_goes_to_the_board_without_asking_the_llm(
    fake_llm, isolated_data, w, monkeypatch
):
    rt = ag.Agent.load("eve", w)
    monkeypatch.setattr(rt.planner, "choose_location",
                        lambda *a, **k: pytest.fail("board work should not ask the LLM"))
    rt.chunk = {"activity": "11:40 am: reviewing the project board for updates",
                "start_time": "11:40 am", "duration": 10, "target_agent": None}
    ag.apply_chunk(rt, w)
    assert rt.agent["current_location"] == "conference_room"
    assert rt.agent["interacting_with"] == "project_board"


def test_inherited_board_work_goes_to_the_board(fake_llm, isolated_data, w, monkeypatch):
    rt = ag.Agent.load("adam", w)
    monkeypatch.setattr(rt.planner, "choose_location",
                        lambda *a, **k: pytest.fail("board work should not ask the LLM"))
    rt.chunk = {"activity": "7:20 am: checking all frontend tasks for updates",
                "start_time": "7:20 am", "duration": 10, "target_agent": None,
                "board_work": True}
    ag.apply_chunk(rt, w)
    assert (rt.agent["current_location"], rt.agent["interacting_with"]) == \
        ("conference_room", "project_board")


def test_apply_chunk_skips_the_location_llm_for_solo_work(fake_llm, isolated_data, w, monkeypatch):
    rt = ag.Agent.load("adam", w)
    rt.agent["current_location"] = "office_1"
    monkeypatch.setattr(rt.planner, "choose_location",
                        lambda *a, **k: pytest.fail("choose_location should not be called"))
    rt.chunk = {"activity": "7:45 am: Outline database models", "location": "office_1",
                "start_time": "7:45 am", "duration": 15, "target_agent": None}
    ag.apply_chunk(rt, w)
    assert rt.agent["current_location"] == "office_1"


def test_the_second_step_of_a_board_visit_shows_what_happened(fake_llm, isolated_data, w):
    _board_api_pages_and_db(w)
    rt = ag.Agent.load("eve", w)
    rt.chunk = rt.last_chunk = {
        "activity": "start T-2 (move it to in progress) on the project board from 7:25 am to 7:35 am",
        "location": "conference room", "start_time": "7:25 am", "duration": 10,
        "target_agent": None, "board_work": True}
    world_module.update_agent_state(rt.agent, rt.chunk, w, location="conference_room")
    rt.agent["interacting_with"] = pb.BOARD_OBJECT
    rt.decision = "1"
    rt.act(8, "7:30 am", w)                       # same chunk: no board call
    assert rt.agent["current_action"].startswith("reporting progress on T-2 (already in progress)")
    assert rt.agent["interacting_with"] == pb.BOARD_OBJECT


def test_a_reaction_replans_the_rest_of_the_day(fake_llm, isolated_data, w, monkeypatch):
    replans = []

    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            replans.append(" ".join(prompt.split()))
            return "1. work on the API from 9:25 am to 5:00 pm"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("adam", w)
    ag.initial_planning(rt, w)
    rt.reaction = "Ask Eve which fields the contact form sends"
    rt.react(30, "9:15 am", w)

    assert len(replans) == 1
    assert "it is now 9:25 am" in replans[0]              # after the 10-minute reaction
    assert "decided to: Ask Eve which fields the contact form sends" in replans[0]
    assert rt.override["activity"] == "9:15 am: Ask Eve which fields the contact form sends"
    assert any("work on the API" in c["activity"] for c in rt.broad_chunks)


def test_the_old_detour_is_still_available(fake_llm, isolated_data, w, monkeypatch):
    monkeypatch.setattr(ag, "REACT_REPLANS", False)
    calls = []

    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            calls.append(prompt)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("adam", w)
    ag.initial_planning(rt, w)
    rt.reaction = "Ask Eve which fields the contact form sends"
    rt.react(30, "9:15 am", w)
    assert calls == []


def test_a_replan_keeps_plans_with_teammates(fake_llm, isolated_data, w, monkeypatch):
    prompts = []
    rt = _replan_eve(w, monkeypatch, prompts)
    assert ag.replan_rest_of_day(rt, "7:25 am", "updated the board", w, "test")
    with_adam = [(c["start_time"], c["duration"]) for c in rt.broad_chunks
                 if c.get("target_agent") == "adam"]
    assert with_adam == [("8:00 am", 15), ("10:15 am", 75)]
    assert "Keep these plans with teammates at the same time and place" in prompts[0]
    # What overlapped them was cut around them, not dropped.
    times = [(c["start_time"], c["duration"]) for c in rt.broad_chunks
             if rt.planner._parse_time(c["start_time"]) >= rt.planner._parse_time("7:25 am")]
    assert times[:5] == [("7:25 am", 35), ("8:00 am", 15), ("8:15 am", 105),
                         ("10:00 am", 15), ("10:15 am", 75)]


def test_a_block_right_after_a_talk_is_not_kept(fake_llm, isolated_data, w, monkeypatch):
    """They just talked; replan_after_conversation already says to leave out
    talking to the same teammate within the hour (conversation cooldown)."""
    rt = _replan_eve(w, monkeypatch)
    adam = ag.Agent.load("adam", w)
    main.remember_conversation(rt, adam, "7:20 am", "the homepage")
    ag.replan_rest_of_day(rt, "7:25 am", "talked", w, "test")
    assert [c["start_time"] for c in rt.broad_chunks
            if c.get("target_agent") == "adam"] == ["10:15 am"]          # 8:00 stand-up dropped


def test_replans_around_tasks_after_leaving_the_board(
    fake_llm, isolated_data, world, monkeypatch
):
    prompts = []
    rt = _runtime(world, monkeypatch, prompts)
    pb.clear_workload_changed()
    pb.create_task(world, "adam", "Set up the API", area="backend", assign_to_self=True)

    rt.agent["interacting_with"] = pb.BOARD_OBJECT
    ag.replan_after_board(rt, "9:00 am", world)
    assert prompts == []                              # still at the board: wait

    rt.agent["interacting_with"] = "computer"
    ag.replan_after_board(rt, "9:00 am", world)

    [prompt] = prompts
    assert "T-1 'Set up the API' (todo)" in prompt
    assert "Starting and finishing tasks at the board is scheduled automatically" in " ".join(prompt.split())
    assert "before lunch and before" in prompt
    assert any("T-1 on the project board" in c["activity"] for c in rt.broad_chunks)
    assert not pb.workload_changed("adam")

    ag.replan_after_board(rt, "9:05 am", world)
    assert len(prompts) == 1                          # one visit, one replan


def test_no_replan_without_a_workload_change(fake_llm, isolated_data, world, monkeypatch):
    prompts = []
    rt = _runtime(world, monkeypatch, prompts)
    pb.clear_workload_changed()
    pb.create_task(world, "adam", "Write docs")       # created, not taken on
    ag.replan_after_board(rt, "9:00 am", world)
    assert prompts == []


def test_replan_after_the_board_adds_checkpoints(fake_llm, isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            return "1. work on T-1 'Set up the API' in the office from 9:05 am to 11:00 am"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("adam", w)
    ag.initial_planning(rt, w)
    pb.create_task(w, "adam", "Set up the API", area="backend", assign_to_self=True)
    ag.replan_after_board(rt, "9:00 am", w)

    new = [c for c in rt.broad_chunks if c["start_time"] in ("9:05 am", "9:15 am", "10:50 am")]
    assert [(c["start_time"], bool(c.get("board_work"))) for c in new] == [
        ("9:05 am", True), ("9:15 am", False), ("10:50 am", True)]


def test_assignee_is_notified_and_replans(fake_llm, isolated_data, w, monkeypatch):
    prompts = []
    monkeypatch.setattr(llm, "generate", _revise_returns_board_visit(prompts))
    rt = ag.Agent.load("adam", w)
    ag.initial_planning(rt, w)
    pb.create_task(w, "eve", "Contact form backend", area="backend")
    pb.assign_task(w, "eve", "T-1", "adam")

    ag.replan_after_board(rt, "11:35 am", w)

    notice = "Eve Smith assigned task T-1 'Contact form backend' to Adam Beck on the project board."
    assert notice in [m.description for m in rt.stream.memories]
    assert notice in prompts[0]
    assert "T-1 'Contact form backend' (todo)" in prompts[0]


def test_a_conversation_replan_absorbs_the_pending_board_replan(
        fake_llm, isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            return "1. work on T-1 'Backend API' in the office from 7:10 am to 12:00 pm"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("adam", w)
    ag.initial_planning(rt, w)
    _adam_with_backend_tasks(w)                       # claiming marks the workload changed
    pb.assign_task(w, "eve", pb.create_task(w, "eve", "Contact endpoint", area="backend")["id"],
                   "adam", note="the form needs POST /contact")
    assert pb.workload_changed("adam")

    assert ag.replan_after_conversation(rt, "I will start on T-1 today", "7:05 am", w)
    assert not pb.workload_changed("adam")            # replan_after_board won't fire again
    memories = [m.description for m in rt.stream.memories]
    assert any("assigned task T-3 'Contact endpoint' to Adam Beck" in m for m in memories)


def test_replan_prompt_asks_for_no_gaps_and_no_reaction_wording(fake_llm, isolated_data, w,
                                                                monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            prompts.append(prompt)
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("eve", w)
    ag.initial_planning(rt, w)
    ag.replan_after_conversation(rt, "build T-4 first", "9:00 am", w)
    assert "What just happened: Eve Smith agreed in a conversation to: build T-4 first" in prompts[0]
    assert "decided to react" not in prompts[0]
    assert "(no gaps)" in prompts[0]


def test_the_replan_after_a_talk_knows_they_just_talked(fake_llm, isolated_data, w, monkeypatch):
    prompts = []

    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            prompts.append(" ".join(prompt.split()))
            return "1. work on T-2 in the office from 7:25 am to 5:00 pm"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("eve", w)
    ag.initial_planning(rt, w)
    ag.replan_after_conversation(rt, "I will keep working on T-2", "7:20 am", w, partner="Adam Beck")
    assert ("Eve Smith and Adam Beck just talked, so leave out any block in the next hour that "
            "is only about talking or collaborating with Adam Beck") in prompts[0]


def test_a_repeated_promise_does_not_replan_again(fake_llm, isolated_data, w, monkeypatch):
    calls = []

    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            calls.append(prompt)
            return "1. work on the API from 9:05 am to 5:00 pm"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("adam", w)
    ag.initial_planning(rt, w)

    assert ag.replan_after_conversation(
        rt, "I will review the integration issues and task list with Eve", "9:00 am", w)
    assert not ag.replan_after_conversation(
        rt, "I will review the integration issues and the task list with Eve Smith",
        "10:30 am", w)
    assert ag.replan_after_conversation(rt, "I will deploy T-2 to staging", "11:00 am", w)
    assert len(calls) == 2


def test_board_promise_at_the_board_is_kept_right_away(fake_llm, isolated_data, w, monkeypatch):
    prompts = []
    monkeypatch.setattr(llm, "generate", _revise_returns_board_visit(prompts))
    rt = ag.Agent.load("eve", w)
    ag.initial_planning(rt, w)
    rt.agent["current_location"] = "conference_room"
    promise = "I will assign T-4 to Adam and note the dependencies on the project board"

    ag.replan_after_conversation(rt, promise, "11:35 am", w)
    ag.queue_board_visit(rt, promise, "11:35 am", w)

    assert pb.commitments_for("eve") == [promise]
    assert "can only keep this promise at the project board" in prompts[0]
    assert rt.queued["location"] == "conference_room" and rt.queued["pinned"]
    assert rt.queued["start_time"] == "11:40 am"


def test_no_immediate_visit_when_the_talk_was_elsewhere(fake_llm, isolated_data, w, monkeypatch):
    monkeypatch.setattr(llm, "generate", _revise_returns_board_visit([]))
    rt = ag.Agent.load("eve", w)
    rt.agent["current_location"] = "break_room"
    ag.queue_board_visit(rt, "assign T-4 to Adam", "12:30 pm", w)
    assert rt.queued is None


def test_keep_working_promise_does_not_send_anyone_to_the_board(fake_llm, isolated_data, w, monkeypatch):
    def gen(prompt, options=None, format=None):
        if "Revise" in prompt:
            return "1. work on T-1 in the office from 7:25 am to 5:00 pm"
        return fake_generate(prompt, options, format)

    monkeypatch.setattr(llm, "generate", gen)
    rt = ag.Agent.load("adam", w)
    ag.initial_planning(rt, w)
    rt.agent["current_location"] = "conference_room"
    promise = "Adam Beck will continue working on T-1 and keep Eve Smith updated"
    assert ag.replan_after_conversation(rt, promise, "7:20 am", w, partner="Eve Smith")
    assert pb.commitments_for("adam") == []
    ag.queue_board_visit(rt, promise, "7:20 am", w)
    assert rt.queued is None


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
