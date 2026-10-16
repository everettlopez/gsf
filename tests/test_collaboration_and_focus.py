"""
Fixes for what the 30-step run.log after "done_when is optional" showed:

1. Adam reacted "Start working on task T-5" while T-1 was in progress: a replan
   around work the board then refused to start (one task at a time).
2. Replans dropped plans with teammates (Eve's 8:00 stand-up and her 10:15
   work with Adam); the run had one conversation in 30 steps.
3. "Check on T-2 progress and confirm if it affects T-1 or T-5" (T-2 is Eve's)
   named nobody, so Adam did it alone in his office for 15 minutes.
4. Both agents reacted to the time clock at 7:05, and in the conversation that
   followed Adam claimed to have done what he had promised a line earlier.
5. Work blocks broke down into "prioritize task list", "update task status in
   project management tool", "address high-priority bugs" (there were none).
6. Eve's note "Eve is currently addressing high-priority bugs or issues ...".
7. A plan named "T-1 'Backend data structure'" for T-1 'Design and implement
   the API ...', and planned to approve it while it was in progress.
8. The shown action read "starting T-2 (move it to in progress)" after the
   board step had been brought up to date.
Plus: an agent can hold at most three open tasks.
"""

import copy

import pytest

import llm
import agent as ag
import main
import project_board as pb
import world as world_module
from conftest import fake_generate
from conversation import claims_promised_work, run_conversation
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


# --- 1. one task at a time, in reactions too ------------------------------------
@pytest.mark.parametrize("reaction,why", [
    ("Start working on task T-5 'Set up database for contact form'", "T-1 is still in progress"),
    ("switch to T-5", "T-1 is still in progress"),
    ("keep working on T-1", ""),                    # the task in progress: fine
    ("Check on T-2 progress", ""),                  # not starting anything
])
def test_a_reaction_cannot_start_a_second_task(w, reaction, why):
    _run_board(w)
    assert pb.start_conflict(w, "adam", reaction).startswith(why) if why \
        else pb.start_conflict(w, "adam", reaction) == ""


def test_a_reaction_cannot_start_a_task_that_waits(w):
    pb.get_project(w)["tasks"].clear()
    pb.create_task(w, "adam", "API", area="backend", assign_to_self=True)
    pb.create_task(w, "eve", "Contact form", area="frontend", depends_on=["T-1"], assign_to_self=True)
    assert pb.start_conflict(w, "eve", "start working on T-2") == "T-2 waits on T-1"


def test_the_start_reaction_from_the_run_is_ignored(fake_llm, isolated_data, w, monkeypatch):
    _run_board(w)

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


def test_the_react_prompt_says_one_task_at_a_time(fake_llm, isolated_data, w):
    _run_board(w)
    rules = Planner(w["agents"]["adam"])._board_rules(w["agents"]["adam"], w)
    assert "T-1 is in progress, so Adam Beck's other tasks wait until it goes to review" in rules


# --- 2. plans with teammates survive a replan -----------------------------------
EVE_DAY = [
    "clock in at reception from 7:00 am to 7:05 am",
    "check the project board in the conference room from 7:05 am to 7:20 am",
    "review and update any frontend tasks from 7:20 am to 8:00 am",
    "attend a stand-up meeting with Adam Beck to discuss progress and next steps from 8:00 am to 8:15 am",
    "work on implementing a new component for the homepage layout from 8:15 am to 10:00 am",
    "collaborate with Adam Beck on integrating the backend API for the contact form from 10:15 am to 11:30 am",
    "take a lunch break in the break room from 1:00 pm to 2:00 pm",
]
# Eve's replan in the run: no stand-up, no 10:15 work with Adam.
EVE_REPLAN = ("1. work on the website's homepage layout from 7:25 am to 10:00 am\n"
              "2. work on styling and accessibility improvements for the about page from 10:00 am to 1:00 pm\n"
              "3. take a lunch break in the break room from 1:00 pm to 2:00 pm\n"
              "4. work on the website's navigation bar from 2:00 pm to 5:00 pm")


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


def test_a_teammate_block_the_replan_kept_is_not_doubled(fake_llm, isolated_data, w):
    eve = w["agents"]["eve"]
    p = Planner(eve)
    kept = p.create_chunks(["sync with Adam Beck from 10:15 am to 11:00 am"], eve, w)
    new = p.create_chunks(["review the API with Adam Beck from 10:00 am to 10:30 am",
                           "work on the about page from 10:30 am to 1:00 pm"], eve, w)
    assert p.keep_blocks(new, kept) == new


# --- 3. a question about a teammate's task goes to the teammate -----------------
def test_a_reaction_about_a_teammates_task_goes_to_them(fake_llm, isolated_data, w, monkeypatch):
    _run_board(w)

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
    _run_board(w)

    class RT:
        agent = w["agents"]["adam"]

    assert ag.aim_at_owner(RT, reaction, w) == reaction


# --- 4. the time clock, and work done mid-conversation ---------------------------
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
    # The 7:05 conversation from the run, word for word.
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


# --- 5. work steps are about the task --------------------------------------------
def test_decomposition_knows_the_task_and_drops_board_steps(fake_llm, isolated_data, w, monkeypatch):
    _run_board(w)
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


def test_breaks_and_board_visits_get_no_task_focus(fake_llm, isolated_data, w):
    _run_board(w)
    p = Planner(w["agents"]["eve"])
    focus = p.work_focus(w["agents"]["eve"], w)
    assert p._focus_text({"activity": "take a lunch break in the break room"}, focus) == ""
    assert p._focus_text({"activity": "check the board", "board_work": True}, focus) == ""
    assert "T-2" in p._focus_text({"activity": "work on the homepage"}, focus)


def test_board_blocks_keep_their_board_steps(fake_llm, isolated_data, w):
    p = Planner(w["agents"]["eve"])
    steps = [{"activity": "8:00 am: Update T-2 on the project board", "start_time": "8:00 am"}]
    assert p._drop_board_detours(steps, {"board_work": True}) == steps


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


# --- 6. notes are for teammates ---------------------------------------------------
@pytest.mark.parametrize("note,ok", [
    ("Eve is currently addressing high-priority bugs or issues. Please check if this affects T-2 progress.", False),
    ("I'm working on the hero section", False),
    ("Eve Smith is still designing the layout", False),
    ("Eve is waiting on the API: Adam, can you share the endpoint list?", True),   # asks Adam
    ("I'm building against mock data until T-1 is ready, is the JSON shape settled?", True),
    ("The homepage needs the API's /pages endpoint before it can show real content", True),
])
def test_notes_that_only_narrate_are_refused(w, note, ok):
    _run_board(w)
    action = {"action": "add_note", "task_id": "T-2", "note": note}
    if ok:
        pb.apply_action(w, w["agents"]["eve"], action)
    else:
        with pytest.raises(pb.BoardError, match="for a teammate to act on"):
            pb.apply_action(w, w["agents"]["eve"], action)


# --- 7. plans name tasks as they really are ---------------------------------------
def test_a_wrong_title_in_a_plan_drops_the_id(w):
    """The id is what's wrong, not the title: putting T-1's title there would
    point the agent at different work than the plan described."""
    _run_board(w)
    eve = w["agents"]["eve"]
    chunk = Planner(eve).create_chunks(
        ["review and approve T-1 'Backend data structure' with Adam Beck from 11:30 am to 11:45 am"],
        eve, w)[0]
    assert chunk["activity"] == ("review and approve 'Backend data structure' with Adam Beck "
                                 "from 11:30 am to 11:45 am")


def test_a_short_title_is_left_alone(w):
    _run_board(w)
    assert pb.fix_task_refs(w, "work on T-2 'homepage layout'") == "work on T-2 'homepage layout'"


def test_approving_a_task_not_in_review_becomes_a_check(w):
    _run_board(w)
    step = "review and approve T-1 'Backend data structure' with Adam Beck on the project board"
    assert pb.reconcile_activity(w, "eve", step) == (
        "check on T-1 (it is in progress, not in review, so there is nothing to approve yet) "
        "with Adam Beck on the project board")
    pb.get_task(w, "T-1")["status"] = "review"
    assert pb.reconcile_activity(w, "eve", step) == step              # now there is


# --- 8. what is shown is what the agent does --------------------------------------
def test_the_shown_action_follows_the_reconciled_board_step(fake_llm, isolated_data, w):
    _run_board(w)
    eve = w["agents"]["eve"]
    chunk = {"activity": "start T-2 (move it to in progress) on the project board from 8:00 am to 8:10 am",
             "location": "conference room", "start_time": "8:00 am", "duration": 10,
             "target_agent": None, "board_work": True}
    world_module.update_agent_state(eve, chunk, w, location="conference_room")
    assert eve["current_action"].startswith("starting T-2")
    world_module.generate_interactions(eve, w, [], Planner(eve), chunk=chunk)
    assert eve["current_action"] == ("reporting progress on T-2 (already in progress) on the "
                                     "project board from 8:00 am to 8:10 am")


def test_a_shown_action_keeps_text_after_a_colon(w):
    eve = w["agents"]["eve"]
    chunk = {"activity": "8:05 am: talk with Adam Beck about T-1: is the API ready",
             "location": "office_2", "start_time": "8:05 am", "duration": 10}
    world_module.update_agent_state(eve, chunk, w)
    assert eve["current_action"] == "talking with Adam Beck about T-1: is the API ready"


# --- sharing the work --------------------------------------------------------------
def test_a_fourth_task_is_left_unassigned(w):
    _run_board(w)
    eve = w["agents"]["eve"]                                  # T-2, T-3, T-4
    line = pb.apply_action(w, eve, {"action": "create_task", "title": "Footer", "area": "frontend",
                                    "assign_to_self": True})
    assert "left it unassigned, since Eve Smith already has 3 open tasks" in line
    assert pb.get_task(w, "T-6")["assignee"] is None
    with pytest.raises(pb.BoardError, match="already has 3 open tasks"):
        pb.apply_action(w, eve, {"action": "claim_task", "task_id": "T-6"})


def test_the_board_shows_who_has_how_much(w):
    _run_board(w)
    assert "Open tasks per person: Adam Beck 2, Eve Smith 3." in pb.status_for_agent(w, "adam")


@pytest.mark.parametrize("plan,shown", [
    ("keep working on T-2 'Homepage'", "still working on T-2 'Homepage'"),
    ("continue working on the API", "still working on the API"),
    ("start building the navigation bar", "starting to build the navigation bar"),
    ("work on T-1", "working on T-1"),
    ("update the project board", "updating the project board"),
])
def test_shown_actions_read_naturally(plan, shown):
    assert world_module.action_to_present(plan) == shown


def test_the_second_step_of_a_board_visit_shows_what_happened(fake_llm, isolated_data, w):
    _run_board(w)
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


def test_after_the_planned_talk_the_agent_goes_back_to_their_task(fake_llm, isolated_data, w, monkeypatch):
    _run_board(w)
    eve, adam = ag.Agent.load("eve", w), ag.Agent.load("adam", w)
    main.remember_conversation(eve, adam, "8:05 am", "the stand-up")
    monkeypatch.setattr(ag, "get_chunk_lazy", lambda rt, t, world=None: {
        "activity": "attend a stand-up meeting with Adam Beck from 8:00 am to 8:15 am",
        "location": "conference room", "start_time": "8:00 am", "duration": 15,
        "target_agent": "adam"})
    eve.decide(15, "8:10 am", w)
    assert eve.chunk["activity"] == "8:00 am: keep working on T-2 'Design and implement the homepage layout'"
    assert eve.chunk["location"] == "office_2" and eve.chunk["target_agent"] is None
