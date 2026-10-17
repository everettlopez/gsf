"""
The website's data: the chronicle keeps the story of the run, snapshot.py
turns the office into the document the site is built against, and export()
commits it to the sandbox's main, where every task branch started after it
finds it (and nobody but the host can change it).
"""

import json
import shutil

import pytest

import chronicle
import clock
import log
import snapshot
import workspace as ws
from memory_stream import MemoryStream
from workspace import Workspace, WorkspaceError


@pytest.fixture
def story(world):
    chronicle.start(world)
    yield chronicle
    chronicle.stop()


# --- the chronicle -------------------------------------------------------------------
def test_events_are_kept_with_their_kind_and_the_agents_they_are_about(story):
    clock.set_time("9:05 am")
    log.event("📋 Eve Smith moved task T-1 'Layout' to review on the project board.")
    log.event("🔍 ADAM reviews Eve Smith's T-1 'Layout'")
    log.event("📋✗ Eve Smith couldn't claim_task: Eve Smith already has 3 open tasks")
    log.event("EVE committed: I will start T-2 once T-1 is approved")
    events = story.data()["events"]
    assert [e["kind"] for e in events] == ["board", "review", "board_refused", "promise"]
    assert events[0] == {"day": 1, "time": "9:05 am", "kind": "board", "agents": ["eve"],
                         "text": "Eve Smith moved task T-1 'Layout' to review on the project board."}
    assert events[1]["agents"] == ["adam", "eve"]


def test_nothing_is_kept_once_the_run_stops(story):
    story.stop()
    log.event("📋 Eve Smith added task T-9")
    assert story.data()["events"] == []


def test_steps_conversations_and_days_are_kept(story, world):
    clock.set_time("7:30 am")
    story.note_day({"adam": ["clock in at reception from 7:00 am to 7:05 am"]})
    story.note_step(world)
    story.note_conversation("conference_room", "sync on T-1",
                            [("Adam Beck", "Morning, Eve."), ("Eve Smith", "Hi Adam.")],
                            {"meeting": None, "commitments": {"Eve Smith": "Review T-4"}})
    data = story.data()
    assert data["days"] == [{"day": 1, "name": "Thursday", "date": clock.now().date().isoformat(),
                             "plans": {"adam": ["clock in at reception from 7:00 am to 7:05 am"]}}]
    assert {s["agent"] for s in data["steps"]} == {"adam", "eve"}
    assert data["steps"][0]["time"] == "7:30 am" and data["steps"][0]["location"] == "reception"
    talk = data["conversations"][0]
    assert talk["agents"] == ["adam", "eve"] and talk["commitments"] == {"eve": "Review T-4"}
    assert talk["lines"][1] == {"agent": "eve", "text": "Hi Adam."}


def test_the_chronicle_is_saved_beside_the_world(story, isolated_data):
    log.event("🚢 Eve Smith merged T-1 'Layout' into the website's main branch.")
    story.save()
    saved = chronicle.load()
    assert saved["events"][0]["kind"] == "ship"
    assert (isolated_data / "chronicle.json").is_file()


# --- the snapshot --------------------------------------------------------------------
def _snapshot(world, story, isolated_data):
    import project_board as pb

    pb.create_task(world, "eve", "Board page", area="frontend", assign_to_self=True)
    clock.set_time("8:15 am")
    stream = MemoryStream("eve")
    stream.memories = []
    stream.add_observation("Eve Smith is sketching the board page.").importance = 4
    log.event("📋 Eve Smith added task T-1 'Board page' to the project board and took it on.")
    story.note_step(world)
    return snapshot.build(world, story.data(), streams={"eve": stream},
                          plans={"eve": [{"start_time": "8:00 am", "duration": 60,
                                          "activity": "work on T-1", "location": "office_2"}]})


def test_the_snapshot_has_the_documented_shape(fake_llm, world, story, isolated_data):
    shot = _snapshot(world, story, isolated_data)
    assert shot["schema"] == snapshot.SCHEMA == 1
    assert shot["generated"] == {"day": 1, "day_name": "Thursday", "time": "8:15 am"}
    assert shot["company"]["name"] == "Beck & Smith"
    assert shot["company"]["email"] == "hello@becksmith.example"
    assert set(shot) == {"schema", "generated", "company", "days", "agents", "rooms", "board",
                         "timeline", "events", "conversations"}
    eve = next(a for a in shot["agents"] if a["id"] == "eve")
    assert eve["name"] == "Eve Smith" and eve["area"] == "frontend" and "shy" in eve["traits"]
    assert eve["now"] == {"location": "reception", "action": "idle", "object": None}
    assert eve["plan"] == [{"start": "8:00 am", "minutes": 60, "activity": "work on T-1",
                            "location": "office_2"}]
    assert eve["memories"] == [{"day": 1, "time": "8:15 am", "importance": 4,
                                "text": "Eve Smith is sketching the board page."}]
    reception = next(r for r in shot["rooms"] if r["id"] == "reception")
    assert set(reception["occupants"]) == {"adam", "eve"} and reception["name"] == "Reception"
    assert shot["board"]["tasks"][0]["title"] == "Board page"
    assert shot["board"]["tasks"][0]["history"][0]["day"] == 1
    assert shot["timeline"][0]["time"] == "8:15 am" and shot["events"][0]["kind"] == "board"
    json.loads(snapshot.dumps(shot))                                    # plain JSON


def test_a_snapshot_from_disk_is_as_of_the_runs_last_step(world, story, isolated_data):
    clock.set_time("4:55 pm")
    story.note_step(world)
    story.save()
    clock.reset("7:00 am")                                              # outside the run
    shot = snapshot.from_disk(isolated_data / "world" / "world.json")
    assert shot["generated"] == {"day": 1, "day_name": "Thursday", "time": "4:55 pm"}


# --- into the sandbox ----------------------------------------------------------------
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


class Recorder:
    def run(self, path, command, timeout=None):
        return ws.CommandResult(0, "ok")


@needs_git
def test_the_snapshot_is_committed_to_main_and_new_tasks_start_with_it(fake_llm, world, story,
                                                                       isolated_data, tmp_path):
    root = tmp_path / "workspaces"
    shot = _snapshot(world, story, isolated_data)
    assert snapshot.export(shot, root) is True
    assert snapshot.export(shot, root) is False                         # nothing new
    eve = Workspace({"id": "eve", "name": "Eve Smith"}, root, runner=Recorder())
    eve.start_task("T-1")
    on_branch = json.loads(eve.read_file("data/snapshot.json"))
    assert on_branch["generated"]["time"] == "8:15 am"
    assert "NO TESTS FOUND" in eve.read_file("run_tests.py")
    log_line = eve.git("log", "-1", "--format=%an: %s", "origin/main")
    assert log_line == "The office: Snapshot of the office: Thursday 8:15 am"


@needs_git
def test_the_hands_cant_change_the_offices_files(tmp_path):
    root = tmp_path / "workspaces"
    eve = Workspace({"id": "eve", "name": "Eve Smith"}, root, runner=Recorder())
    eve.start_task("T-1")
    for path in ("run_tests.py", "data/snapshot.json", "./data/snapshot.json"):
        with pytest.raises(WorkspaceError, match="kept by the simulation"):
            eve.write_file(path, "{}")
    (eve.path / "run_tests.py").write_text("print('all good')\n")    # a command did it
    (eve.path / "data").mkdir()
    (eve.path / "data" / "snapshot.json").write_text("{}")
    eve.write_file("site/pages/index.html", "<h1>Hi</h1>")
    assert eve.commit("T-1: the home page")
    assert eve.changed_files() == ["site/pages/index.html"]
    assert "NO TESTS FOUND" in eve.read_file("run_tests.py")
    assert not (eve.path / "data" / "snapshot.json").exists()


@needs_git
def test_a_task_merges_cleanly_after_newer_snapshots(fake_llm, world, story, isolated_data,
                                                     tmp_path):
    root = tmp_path / "workspaces"
    snapshot.export(_snapshot(world, story, isolated_data), root)
    eve = Workspace({"id": "eve", "name": "Eve Smith"}, root, runner=Recorder())
    eve.start_task("T-1")
    eve.write_file("site/pages/index.html", "<h1>Hi</h1>")
    eve.commit("T-1: the home page")
    eve.publish("T-1")
    clock.set_time("9:00 am")
    snapshot.export(snapshot.build(world, story.data()), root)          # main moves on
    eve.integrate("T-1")
    eve.git("fetch", "origin")
    files = eve.git("ls-tree", "-r", "--name-only", "origin/main").splitlines()
    assert "site/pages/index.html" in files
    newest = json.loads(eve.git("show", "origin/main:data/snapshot.json"))
    assert newest["generated"]["time"] == "9:00 am"
