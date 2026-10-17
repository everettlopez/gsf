"""
The sandbox the hands work in: its image has Python and Node, its run_tests.py
runs both kinds of tests (and is the host's word on whether they pass), and a
reset sets the last project's sandbox aside instead of deleting it.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import executor
import workspace as ws
from conftest import seed_module

ROOT = Path(__file__).resolve().parent.parent
RUN_TESTS = ROOT / "sandbox" / "run_tests.py"

PY_TEST = "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n"
JS_PASS = ('const test = require("node:test");\nconst assert = require("node:assert/strict");\n'
           'test("adds", () => assert.equal(1 + 1, 2));\n')
JS_FAIL = JS_PASS.replace("2));", "3));")


def _repo(tmp_path, files):
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    shutil.copy(RUN_TESTS, tmp_path / "run_tests.py")
    return subprocess.run([sys.executable, "run_tests.py"], cwd=tmp_path, capture_output=True,
                          text=True)


def test_no_tests_is_its_own_answer(tmp_path):
    done = _repo(tmp_path, {})
    assert done.returncode == 5 and "NO TESTS FOUND" in done.stdout


def test_python_tests_alone(tmp_path):
    done = _repo(tmp_path, {"tests/test_a.py": PY_TEST})
    assert done.returncode == 0 and "all tests passed" in done.stdout


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


@needs_node
def test_python_and_javascript_tests_both_count(tmp_path):
    done = _repo(tmp_path, {"tests/test_a.py": PY_TEST, "tests/js/a.test.js": JS_PASS})
    assert done.returncode == 0 and "JavaScript tests" in done.stdout
    done = _repo(tmp_path, {"tests/js/b.test.js": JS_FAIL})
    assert done.returncode == 1 and "FAILED: JavaScript" in done.stdout


def test_the_host_runs_the_sandboxs_own_test_command():
    assert executor.TEST_COMMAND == f"python {ws.RUN_TESTS}"
    assert ws.host_file_contents()[ws.RUN_TESTS] == RUN_TESTS.read_text()


def test_the_image_has_python_and_node_and_is_the_default():
    dockerfile = (ROOT / "sandbox" / "Dockerfile").read_text()
    assert "FROM python:3.12" in dockerfile and "/usr/local/bin/node" in dockerfile
    assert ws.DockerRunner().image == ws.DOCKER_IMAGE == "gsf-sandbox:py3.12-node22"


def test_the_image_is_built_once_when_missing(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args[:3])
        return subprocess.CompletedProcess(args, 1 if args[1] == "image" else 0, "", "")

    monkeypatch.setattr(ws.subprocess, "run", fake_run)
    ws.ensure_image()
    assert calls == [["docker", "image", "inspect"], ["docker", "build", "-t"]]


def test_a_javascript_test_counts_as_tests_for_the_host(tmp_path):
    class Files:
        def __init__(self, files, output):
            self.files, self.output, self.ran = files, output, []

        def list_files(self, path):
            return self.files

        def run(self, command):
            self.ran.append(command)
            return ws.CommandResult(0, self.output)

    js_only = Files(["tests/js/board.test.js"], "== all tests passed ==")
    assert executor.LocalExecutor()._final_tests(js_only) is True
    assert js_only.ran == ["python run_tests.py"]
    assert executor.LocalExecutor()._final_tests(Files(["tests/fixtures/snapshot.json"], "")) is None


def test_a_reset_sets_the_last_sandbox_aside(tmp_path):
    seed = seed_module()
    (tmp_path / "workspaces" / "origin.git").mkdir(parents=True)
    dest = seed.archive_sandbox(tmp_path)
    assert not (tmp_path / "workspaces").exists()
    assert dest.parent == tmp_path / "workspaces-archive" and (dest / "origin.git").is_dir()
    assert seed.archive_sandbox(tmp_path) is None                       # nothing to set aside
