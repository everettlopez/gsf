"""
executor.py and the hands in agent.py: when an agent's step is work on a task
it owns and has in progress, the executor does the work (once per plan block),
and the outcome comes back as memories. A finished task becomes a promise to
move it to review; a blocked one sends the agent to the board to say so. The
executor never works on a teammate's task, at the board, in a talk or on a break.
"""

import pytest

import agent as ag
import project_board as pb
from conftest import board_api_and_pages
from executor import ExecResult, FakeExecutor

PROGRESS = ExecResult("Built the hero section and the nav bar.", ["index.html", "styles.css"], None)
DONE = ExecResult("The home page is complete.", ["index.html"], tests_passed=True)
STUCK = ExecResult("Couldn't wire the contact form.", [], tests_passed=False,
                   blocked_reason="the /contact endpoint doesn't exist yet")


def _eve_at_her_desk(w, results, activity="9:00 am: Build the hero section"):
    """Eve, with T-2 in progress, in office_2, on a work step of a 9-11 block."""
    board_api_and_pages(w)
    executor = FakeExecutor(results)
    rt = ag.Agent.load("eve", w, executor=executor, workspace="workspaces/eve")
    rt.agent["current_location"] = "office_2"
    rt.broad_chunks = [{"activity": "work on T-2 'Design and implement the home page layout' "
                                    "from 9:00 am to 11:00 am", "location": "office_2",
                        "start_time": "9:00 am", "duration": 120, "target_agent": None}]
    rt.chunk = {"activity": activity, "location": "office_2", "start_time": "9:00 am",
                "duration": 15, "target_agent": None}
    return rt, executor


def _memories(rt):
    return [m.description for m in rt.stream.memories]


# --- when the hands work ---------------------------------------------------------
def test_work_on_an_own_task_in_progress_goes_to_the_executor(fake_llm, isolated_data, w):
    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    assert ag.work_on_task(rt, "9:00 am", w) is PROGRESS
    assert executor.calls == [("eve", "T-2", "workspaces/eve")]
    memories = _memories(rt)
    assert ("Eve Smith worked on T-2 'Design and implement the home page layout' at 9:00 am: "
            "Built the hero section and the nav bar.") in memories
    assert "Eve Smith changed index.html, styles.css for T-2." in memories
    assert not any("tests for T-2" in m for m in memories)            # none were run


def test_one_session_per_plan_block(fake_llm, isolated_data, w):
    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    ag.work_on_task(rt, "9:00 am", w)
    rt.chunk = dict(rt.chunk, activity="9:15 am: Style the hero section", start_time="9:15 am")
    assert ag.work_on_task(rt, "9:15 am", w) is None                  # same 9-11 block
    rt.broad_chunks.append({"activity": "work on T-2 from 11:00 am to 12:00 pm", "location": "office_2",
                            "start_time": "11:00 am", "duration": 60, "target_agent": None})
    rt.chunk = dict(rt.chunk, activity="11:00 am: Add the footer", start_time="11:00 am")
    ag.work_on_task(rt, "11:00 am", w)
    assert len(executor.calls) == 2


@pytest.mark.parametrize("change", [
    {"activity": "9:00 am: Review T-1's API responses"},                      # Adam's task
    {"activity": "9:00 am: take a coffee break"},
    {"activity": "9:00 am: move T-2 to review on the project board", "board_work": True},
    {"activity": "9:00 am: pair with Adam Beck on the API", "target_agent": "adam"},
    {"location": "conference_room"},                                           # not at her desk
])
def test_the_hands_stay_in_their_lane(fake_llm, isolated_data, w, change):
    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    if change.get("location"):
        rt.agent["current_location"] = change["location"]
    rt.chunk = {**rt.chunk, **{k: v for k, v in change.items() if k != "location"}}
    assert ag.work_on_task(rt, "9:00 am", w) is None
    assert executor.calls == []


def test_only_a_task_in_progress_is_worked_on(fake_llm, isolated_data, w):
    rt, executor = _eve_at_her_desk(w, [PROGRESS], activity="9:00 am: work on T-3")
    assert pb.get_task(w, "T-3")["status"] == "todo"                   # not started at the board
    assert ag.work_on_task(rt, "9:00 am", w) is None
    rt.executor = None
    rt.chunk["activity"] = "9:00 am: work on T-2"
    assert ag.work_on_task(rt, "9:00 am", w) is None                   # no hands: the paper's agents
    assert executor.calls == []


def test_acting_on_a_work_step_uses_the_hands_but_talking_does_not(fake_llm, isolated_data, w):
    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    rt.decision = "1"
    rt.act(20, "9:00 am", w)
    assert len(executor.calls) == 1
    rt.decision = "3"                                                  # in a conversation
    rt.chunk = dict(rt.chunk, start_time="11:00 am")
    rt.act(21, "11:00 am", w)
    assert len(executor.calls) == 1


# --- what comes of it ---------------------------------------------------------------
def test_a_finished_task_becomes_a_promise_to_move_it_to_review(fake_llm, isolated_data, w):
    rt, executor = _eve_at_her_desk(w, [DONE])
    ag.work_on_task(rt, "9:00 am", w)
    assert "The tests for T-2 pass." in _memories(rt)
    assert pb.commitments_for("eve") == [
        "move T-2 to review on the project board, noting for the reviewer: The home page is "
        "complete. Changed: index.html. Tests pass on branch task/T-2."]
    assert pb.is_board_commitment(pb.commitments_for("eve")[0])

    rt.broad_chunks[0]["start_time"] = "11:00 am"                      # a later block
    assert ag.work_on_task(rt, "11:00 am", w) is None                  # nothing new to do
    pb.add_note(w, "adam", "T-2", "The hero image is missing alt text")
    ag.work_on_task(rt, "11:00 am", w)                                 # the board changed
    assert len(executor.calls) == 2


def test_a_blocked_task_sends_the_agent_to_the_board_with_the_reason(fake_llm, isolated_data, w):
    rt, _ = _eve_at_her_desk(w, [STUCK])
    ag.work_on_task(rt, "9:00 am", w)
    assert "Eve Smith is stuck on T-2: the /contact endpoint doesn't exist yet." in _memories(rt)
    assert "The tests for T-2 fail." in _memories(rt)
    promise = "mark T-2 blocked on the project board: the /contact endpoint doesn't exist yet"
    assert pb.commitments_for("eve") == [promise]

    visit = ag.get_chunk_lazy(rt, "9:05 am", w)                         # the very next step
    assert visit["board_work"] and visit["location"] == "conference_room"
    assert visit["activity"] == f"9:05 am: {promise}"
    prompt = pb.build_prompt(rt.agent, w, visit["activity"], "summary", "",
                             commitments=pb.commitments_for("eve"))
    assert f"- {promise}" in prompt                                     # the board visit knows why


def test_a_blocked_report_waits_for_a_step_already_under_way(fake_llm, isolated_data, w):
    rt, _ = _eve_at_her_desk(w, [STUCK])
    rt.override = {"activity": "9:00 am: Build the hero section", "location": "office_2",
                   "start_time": "9:00 am", "duration": 15, "target_agent": None}
    ag.work_on_task(rt, "9:00 am", w)
    assert rt.override["activity"] == "9:00 am: Build the hero section"
    assert rt.queued["board_work"]


# --- the executor interface ----------------------------------------------------------
def test_the_fake_executor_plays_its_script_and_repeats_the_last_result():
    class A:
        agent = {"id": "eve"}

    fake = FakeExecutor([PROGRESS, DONE])
    task = {"id": "T-2"}
    assert [fake.run(A, task, None) for _ in range(3)] == [PROGRESS, DONE, DONE]
    assert FakeExecutor(lambda a, t, ws: STUCK).run(A, task, None) is STUCK


def test_a_result_is_finished_only_when_tests_pass_and_nothing_blocks_it():
    assert DONE.finished and not DONE.blocked
    assert not PROGRESS.finished                                        # no tests run
    assert STUCK.blocked and not STUCK.finished
    assert not ExecResult("x", tests_passed=True, blocked_reason="no key").finished


def test_many_changed_files_are_summarized(fake_llm, isolated_data, w):
    files = [f"page{i}.html" for i in range(9)]
    rt, _ = _eve_at_her_desk(w, [ExecResult("Added pages.", files)])
    ag.work_on_task(rt, "9:00 am", w)
    assert ("Eve Smith changed page0.html, page1.html, page2.html, page3.html, page4.html, "
            "page5.html and 3 more for T-2.") in _memories(rt)


# --- choosing the hands, and shipping approved work ------------------------------
def test_the_environment_chooses_the_hands(monkeypatch, tmp_path):
    import executor
    import workspace

    monkeypatch.delenv("GA_EXECUTOR", raising=False)
    assert executor.hands_for({"id": "eve"}) == (None, None)            # the paper's agents
    monkeypatch.setenv("GA_EXECUTOR", "fake")
    hands, place = executor.hands_for({"id": "eve"})
    assert isinstance(hands.inner, FakeExecutor) and place is None
    monkeypatch.setenv("GA_EXECUTOR", "local")
    monkeypatch.setenv("GA_SANDBOX", "local")
    monkeypatch.setattr(workspace, "WORKSPACES", tmp_path)
    hands, place = executor.hands_for({"id": "eve", "name": "Eve Smith"})
    assert isinstance(hands.inner, executor.LocalExecutor) and place.path == tmp_path / "eve"
    monkeypatch.setenv("GA_EXECUTOR", "robot")
    with pytest.raises(ValueError, match="unknown GA_EXECUTOR"):
        executor.hands_for({"id": "eve"})


def test_a_done_task_is_merged_into_main_once(fake_llm, isolated_data, w, tmp_path):
    from workspace import Workspace

    rt, _ = _eve_at_her_desk(w, [DONE])
    rt.workspace = Workspace(rt.agent, tmp_path / "workspaces", runner=object())
    rt.workspace.start_task("T-2")
    rt.workspace.write_file("index.html", "<h1>Welcome</h1>")
    rt.workspace.commit("T-2: home page")
    rt.workspace.publish("T-2")
    rt.work_log["T-2"] = [{"block": "9:00 am", "result": DONE, "history": 0}]

    ag.ship_approved_work(rt, w)                                         # still in progress
    assert rt.shipped == set()

    pb.get_task(w, "T-2")["status"] = "done"
    ag.ship_approved_work(rt, w)
    ag.ship_approved_work(rt, w)
    assert rt.shipped == {"T-2"}
    assert "index.html" in rt.workspace.git("ls-tree", "--name-only", "origin/main").split()
    merged = [m for m in _memories(rt) if "merged T-2" in m]
    assert merged == ["Eve Smith merged T-2 'Design and implement the home page layout' "
                      "into the website's main branch."]


def test_a_failed_merge_is_remembered(fake_llm, isolated_data, w, tmp_path):
    from workspace import Workspace

    rt, _ = _eve_at_her_desk(w, [DONE])
    rt.workspace = Workspace(rt.agent, tmp_path / "workspaces", runner=object())
    rt.work_log["T-2"] = [{"block": "9:00 am", "result": DONE, "history": 0}]
    pb.get_task(w, "T-2")["status"] = "done"                             # but never published
    ag.ship_approved_work(rt, w)
    assert any(m.startswith("Eve Smith couldn't merge T-2") for m in _memories(rt))


# --- reviewing a teammate's task ----------------------------------------------------
def _adams_task_in_review(w):
    """T-1 (Adam's) sent to review once."""
    pb.update_status(w, "adam", "T-1", "review", note="the content file and build are ready")
    return pb.get_task(w, "T-1")


def test_a_teammates_task_in_review_is_reviewed_before_own_work(fake_llm, isolated_data, w):
    from executor import ReviewResult

    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    _adams_task_in_review(w)
    result = ag.work_on_task(rt, "9:00 am", w)
    assert isinstance(result, ReviewResult) and executor.review_calls == [("eve", "T-1")]
    assert executor.calls == []                                         # her own work waits
    memories = _memories(rt)
    assert any(m.startswith("Eve Smith reviewed Adam Beck's T-1") for m in memories)
    assert "Eve Smith thinks T-1 is ready to approve." in memories
    assert pb.commitments_for("eve") == [
        "approve T-1 on the project board: The branch does what the task says"]
    assert pb.is_board_commitment(pb.commitments_for("eve")[0])

    rt.broad_chunks[0]["start_time"] = "11:00 am"                       # next block: her own work
    ag.work_on_task(rt, "11:00 am", w)
    assert executor.review_calls == [("eve", "T-1")] and len(executor.calls) == 1


def test_changes_requested_become_a_note_promise_and_a_new_round_is_reviewed_again(
        fake_llm, isolated_data, w):
    from executor import ReviewResult

    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    executor.reviews = [ReviewResult("changes", "The build works but misses a check.",
                                     ["build.py doesn't escape the tagline",
                                      "no test for a missing key"], tests_passed=True)]
    task = _adams_task_in_review(w)
    ag.work_on_task(rt, "9:00 am", w)
    promise = pb.commitments_for("eve")[0]
    assert promise == ("leave a note on T-1 on the project board asking Adam Beck for changes: "
                       "build.py doesn't escape the tagline; no test for a missing key")
    assert pb.is_board_commitment(promise)

    pb.add_note(w, "eve", "T-1", "build.py doesn't escape the tagline")  # at the board
    pb.update_status(w, "adam", "T-1", "in_progress")                    # Adam reopens it
    pb.update_status(w, "adam", "T-1", "review", note="fixed")            # second round
    assert task["status"] == "review"
    rt.broad_chunks[0]["start_time"] = "1:00 pm"
    ag.work_on_task(rt, "1:00 pm", w)
    assert len(executor.review_calls) == 2


def test_no_review_away_from_the_desk_or_of_an_approved_task(fake_llm, isolated_data, w):
    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    task = _adams_task_in_review(w)
    rt.chunk = dict(rt.chunk, board_work=True, activity="9:00 am: approve T-1 on the project board")
    assert ag.review_at_hand(rt, w) is None
    rt.chunk = dict(rt.chunk, board_work=False, activity="9:00 am: Build the hero section")
    pb.approve_task(w, "eve", "T-1")
    assert ag.review_at_hand(rt, w) is None                              # she already approved it
    task["approved_by"].clear()
    assert ag.review_at_hand(rt, w) is task


def test_an_unpublished_branch_is_retried_later(fake_llm, isolated_data, w):
    from executor import ReviewResult

    rt, executor = _eve_at_her_desk(w, [PROGRESS])
    executor.reviews = [ReviewResult("none", "T-1 has no published branch to review yet.")]
    _adams_task_in_review(w)
    ag.work_on_task(rt, "9:00 am", w)
    assert "Eve Smith couldn't review Adam Beck's T-1" in _memories(rt)[-1]
    assert pb.commitments_for("eve") == [] and ag.review_at_hand(rt, w) is not None


# --- the run's session budget -------------------------------------------------------
def test_the_agents_share_one_session_budget(fake_llm, isolated_data, w):
    from executor import Budgeted, SessionBudget

    budget = SessionBudget(1)
    rt, fake = _eve_at_her_desk(w, [PROGRESS])
    rt.executor = Budgeted(fake, budget)
    adam = ag.Agent.load("adam", w, executor=Budgeted(FakeExecutor(), budget))
    assert ag.work_on_task(rt, "9:00 am", w) is PROGRESS
    assert not adam.executor.available()                                # Eve spent it
    rt.broad_chunks[0]["start_time"] = "11:00 am"
    assert ag.work_on_task(rt, "11:00 am", w) is None and len(fake.calls) == 1


def test_the_budget_comes_from_the_environment(monkeypatch):
    import executor

    monkeypatch.delenv("GA_EXECUTOR_MAX_SESSIONS", raising=False)
    monkeypatch.setenv("GA_EXECUTOR", "fake")
    assert executor.run_budget().left is None                           # fake: no limit
    monkeypatch.setenv("GA_EXECUTOR", "local")
    assert executor.run_budget().left == executor.DEFAULT_MAX_SESSIONS  # real hands: capped
    monkeypatch.setenv("GA_EXECUTOR_MAX_SESSIONS", "5")
    assert executor.run_budget().left == 5
