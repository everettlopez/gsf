"""
workspace.py: each agent's checkout of the sandbox repository, with file access
confined to it, a branch per task started from the latest main, publish for
review and merge into main, and commands run by a runner (Docker by default;
never quietly on this machine).
"""

import shutil
import subprocess

import pytest

import workspace as ws
from workspace import CommandResult, DockerRunner, LocalRunner, Workspace, WorkspaceError

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

EVE = {"id": "eve", "name": "Eve Smith"}
ADAM = {"id": "adam", "name": "Adam Beck"}


class Recorder:
    """A runner that records commands instead of running them."""

    def __init__(self, result=CommandResult(0, "ok")):
        self.result, self.calls = result, []

    def run(self, path, command, timeout=ws.COMMAND_TIMEOUT):
        self.calls.append((path, command, timeout))
        return self.result


@pytest.fixture
def root(tmp_path):
    return tmp_path / "workspaces"


def test_a_workspace_is_a_checkout_of_the_sandbox_repo(root):
    eve = Workspace(EVE, root, runner=Recorder())
    assert eve.path == root / "eve" and (eve.path / ".git").is_dir()
    assert eve.list_files() == ["README.md"]
    assert "sandbox repository" in eve.read_file("README.md")
    assert Workspace(ADAM, root, runner=Recorder()).origin == eve.origin   # one shared origin


@pytest.mark.parametrize("path", ["../adam/notes.txt", "/etc/passwd", "C:/Windows/win.ini",
                                  ".git/config", "sub/../../escape.txt"])
def test_file_access_stays_inside_the_checkout(root, path):
    eve = Workspace(EVE, root, runner=Recorder())
    with pytest.raises(WorkspaceError):
        eve.write_file(path, "x")
    with pytest.raises(WorkspaceError):
        eve.read_file(path)


def test_files_are_written_read_and_listed(root):
    eve = Workspace(EVE, root, runner=Recorder())
    eve.write_file("site/index.html", "<h1>Hi</h1>")
    eve.write_file("site/css/styles.css", "h1 { color: navy; }")
    assert eve.read_file("site/index.html") == "<h1>Hi</h1>"
    assert eve.list_files("site") == ["site/css/styles.css", "site/index.html"]
    with pytest.raises(WorkspaceError, match="no file"):
        eve.read_file("site/missing.html")
    with pytest.raises(WorkspaceError, match="directory"):
        eve.write_file("site", "x")


def test_commands_go_to_the_runner_in_the_checkout(root):
    runner = Recorder(CommandResult(1, "1 failed"))
    eve = Workspace(EVE, root, runner=runner)
    result = eve.run("python -m pytest -q", timeout=30)
    assert runner.calls == [(eve.path, "python -m pytest -q", 30)]
    assert not result.ok and result.output == "1 failed"


def test_a_task_branch_starts_from_main_and_carries_a_teammates_merged_work(root):
    adam = Workspace(ADAM, root, runner=Recorder())
    assert adam.start_task("T-1") == "task/T-1"
    adam.write_file("api.json", "{}")
    assert adam.changed_files() == ["api.json"]
    assert adam.commit("T-1: the API contract") and not adam.commit("nothing new")
    adam.publish("T-1")
    adam.integrate("T-1")

    eve = Workspace(EVE, root, runner=Recorder())
    eve.start_task("T-2")
    assert "api.json" in eve.list_files()                       # Adam's merged work is there
    assert eve.changed_files() == []
    eve.write_file("index.html", "<h1>Hi</h1>")
    eve.commit("T-2: home page")
    eve.start_task("T-3")
    assert "index.html" not in eve.list_files()                 # T-3 starts clean from main
    eve.start_task("T-2")
    assert "index.html" in eve.list_files()                     # and T-2's work is kept


def test_commits_carry_the_agents_name(root):
    eve = Workspace(EVE, root, runner=Recorder())
    eve.start_task("T-2")
    eve.write_file("index.html", "<h1>Hi</h1>")
    eve.commit("T-2: home page")
    assert eve.git("log", "-1", "--format=%an <%ae>") == "Eve Smith <eve@gsf.local>"


# --- runners --------------------------------------------------------------------
def test_the_local_runner_runs_in_the_directory_and_times_out(tmp_path):
    (tmp_path / "hello.txt").write_text("hi")
    runner = LocalRunner()
    listed = runner.run(tmp_path, "git --version")
    assert listed.ok and "git version" in listed.output
    assert runner.run(tmp_path, "git not-a-command").exit_code != 0


def test_the_docker_runner_isolates_the_command(tmp_path):
    line = DockerRunner(image="node:22-slim").command_line(tmp_path, "npm test", "gsf-x")
    assert line[:3] == ["docker", "run", "--rm"]
    assert line[line.index("--network") + 1] == "none"
    assert f"{tmp_path.resolve()}:/work" in line and line[-3:] == ["sh", "-c", "npm test"]
    assert "node:22-slim" in line


def test_the_docker_runner_kills_a_command_that_runs_too_long(tmp_path, monkeypatch):
    seen = []

    def fake_run(args, **kw):
        seen.append(args)
        if args[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(args, kw.get("timeout"))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(ws.subprocess, "run", fake_run)
    result = DockerRunner().run(tmp_path, "sleep 999", timeout=1)
    assert result.exit_code == 124
    assert seen[1][:3] == ["docker", "rm", "-f"]                 # the container is removed


def test_without_docker_the_model_never_runs_commands_here_unasked(monkeypatch):
    monkeypatch.delenv("GA_SANDBOX", raising=False)
    monkeypatch.setattr(ws.shutil, "which", lambda name: None)
    with pytest.raises(WorkspaceError, match="GA_SANDBOX=local"):
        ws.default_runner()
    monkeypatch.setenv("GA_SANDBOX", "local")
    assert isinstance(ws.default_runner(), LocalRunner)
    monkeypatch.setenv("GA_SANDBOX", "vm")
    with pytest.raises(WorkspaceError, match="unknown GA_SANDBOX"):
        ws.default_runner()


def test_long_output_is_trimmed_in_the_middle():
    text = "a" * 15_000 + "b" * 15_000
    trimmed = ws._trim(text)
    assert len(trimmed) < len(text) and trimmed.startswith("a") and trimmed.endswith("b")
    assert "characters cut" in trimmed
