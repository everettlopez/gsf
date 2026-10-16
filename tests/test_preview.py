"""
scripts/preview_site.py: the agents' work put together from the sandbox's main
and every task branch (published, or still in an agent's checkout), then built
and tested. Recording runner, temp workspaces: no Docker.
"""

import importlib.util
import json
import shutil

import pytest

from conftest import ROOT
from workspace import CommandResult, Workspace

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

spec = importlib.util.spec_from_file_location("preview_site", ROOT / "scripts" / "preview_site.py")
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class Runner:
    def __init__(self):
        self.commands = []

    def run(self, path, command, timeout=None):
        self.commands.append(command)
        return CommandResult(0, "ok")


@pytest.fixture
def sandbox(tmp_path):
    """Adam's T-1 published; Eve's T-2 committed in her checkout only."""
    root = tmp_path / "workspaces"
    adam = Workspace({"id": "adam", "name": "Adam Beck"}, root, runner=Runner())
    adam.start_task("T-1")
    adam.write_file("build.py", "print('built')\n")
    adam.write_file("content/site.json", "{}")
    adam.commit("T-1: the build")
    adam.publish("T-1")
    eve = Workspace({"id": "eve", "name": "Eve Smith"}, root, runner=Runner())
    eve.start_task("T-2")
    eve.write_file("site/template.html", "<h1>{{company_name}}</h1>")
    eve.commit("T-2: the layout")
    world = tmp_path / "world.json"
    world.write_text(json.dumps({"project": {"tasks": [
        {"id": "T-1", "title": "Build", "status": "review", "assignee": "adam"},
        {"id": "T-2", "title": "Layout", "status": "in_progress", "assignee": "eve"}]}}))
    return root, world


def test_both_agents_work_is_combined_on_top_of_main(sandbox, tmp_path):
    root, world = sandbox
    out = tmp_path / "preview"
    report = preview.combine(root, out, world)
    assert (out / "build.py").exists() and (out / "content" / "site.json").exists()
    assert (out / "site" / "template.html").exists() and (out / "README.md").exists()
    assert report[1] == "T-1 [review, adam] from published: T-1: the build"
    assert report[2] == "T-2 [in progress, eve] from eve's checkout: T-2: the layout"


def test_the_newest_commit_of_a_task_wins(sandbox, tmp_path):
    root, world = sandbox
    adam = Workspace({"id": "adam", "name": "Adam Beck"}, root, runner=Runner())
    adam.start_task("T-1")
    adam.write_file("build.py", "print('built v2')\n")
    adam.commit("T-1: v2, not published yet")
    out = tmp_path / "preview"
    report = preview.combine(root, out, world)
    assert (out / "build.py").read_text() == "print('built v2')\n"
    assert "T-1 [review, adam] from adam's checkout: T-1: v2, not published yet" in report


def test_the_combined_site_is_built_and_tested(sandbox, tmp_path):
    root, world = sandbox
    out = tmp_path / "preview"
    preview.combine(root, out, world)
    (out / "tests").mkdir()
    runner = Runner()
    results = preview.build_and_test(out, runner)
    assert runner.commands == ["python build.py", "python -m unittest discover -s tests"]
    assert set(results) == {"build", "tests"}
    assert preview.page(out) == out / "site" / "template.html"          # not built: the layout
    (out / "dist").mkdir()
    (out / "dist" / "index.html").write_text("<h1>Beck</h1>")
    assert preview.page(out) == out / "dist" / "index.html"


def test_without_a_sandbox_it_says_how_to_get_one(tmp_path, capsys):
    assert preview.main(["--workspaces", str(tmp_path / "none"), "--no-open"]) == 1
    assert "GA_EXECUTOR=local" in capsys.readouterr().out
