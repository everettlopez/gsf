"""
workspace.py - where an agent's hands work.

Each agent gets its own git checkout, workspaces/<agent>/, cloned from a
sandbox repository on disk, workspaces/origin.git (nothing here talks to
GitHub). Work on a task happens on its own branch, task/T-3, started from the
latest main, so teammates' finished work is there when a new task begins:

    origin.git (main) ──clone──> workspaces/eve   task/T-2 ──publish──> origin.git
                                                                 ──integrate──> main

File access is confined to the checkout. Commands the model asks for run in a
runner: DockerRunner (default: a throwaway container, the checkout mounted at
/work, no network) or LocalRunner (on this machine; only when asked for with
GA_SANDBOX=local, since the commands come from a model).
"""

import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACES = ROOT / "workspaces"

MAX_READ_BYTES = 200_000
COMMAND_TIMEOUT = 120          # seconds
MAX_OUTPUT = 20_000            # characters of command output kept

DOCKER_IMAGE = "python:3.12-slim"

SEED_README = """# Company website

The team's sandbox repository. Each task is built on its own branch
(task/T-<n>) and merged into main once a teammate has approved it.
"""


class WorkspaceError(Exception):
    """A request the workspace refuses (a path outside it, a missing file, ...)."""


@dataclass
class CommandResult:
    exit_code: int
    output: str                 # stdout and stderr together, trimmed to MAX_OUTPUT

    @property
    def ok(self):
        return self.exit_code == 0


def _trim(text):
    if len(text) <= MAX_OUTPUT:
        return text
    half = MAX_OUTPUT // 2
    return f"{text[:half]}\n... ({len(text) - MAX_OUTPUT} characters cut) ...\n{text[-half:]}"


class LocalRunner:
    """Runs commands on this machine, in the checkout. Not a sandbox."""

    def run(self, path, command, timeout=COMMAND_TIMEOUT):
        try:
            done = subprocess.run(command, shell=True, cwd=path, capture_output=True, text=True,
                                  timeout=timeout, errors="replace")
        except subprocess.TimeoutExpired:
            return CommandResult(124, f"timed out after {timeout}s")
        return CommandResult(done.returncode, _trim(done.stdout + done.stderr))


class DockerRunner:
    """Runs each command in a fresh container: the checkout at /work, no
    network, capped memory, CPU and processes, removed afterwards."""

    def __init__(self, image=None):
        self.image = image or os.environ.get("GA_SANDBOX_IMAGE") or DOCKER_IMAGE

    def command_line(self, path, command, name):
        return ["docker", "run", "--rm", "--name", name, "--network", "none",
                "--memory", "1g", "--cpus", "1", "--pids-limit", "256",
                "-v", f"{Path(path).resolve()}:/work", "-w", "/work",
                self.image, "sh", "-c", command]

    def run(self, path, command, timeout=COMMAND_TIMEOUT):
        name = f"gsf-{uuid.uuid4().hex[:12]}"
        try:
            done = subprocess.run(self.command_line(path, command, name), capture_output=True,
                                  text=True, timeout=timeout, errors="replace")
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            return CommandResult(124, f"timed out after {timeout}s")
        return CommandResult(done.returncode, _trim(done.stdout + done.stderr))


def default_runner():
    """GA_SANDBOX=docker (default) or local. Without Docker, refuse rather
    than quietly run a model's commands on this machine."""
    choice = os.environ.get("GA_SANDBOX", "docker")
    if choice == "local":
        return LocalRunner()
    if choice != "docker":
        raise WorkspaceError(f"unknown GA_SANDBOX {choice!r} (docker or local)")
    if shutil.which("docker") is None:
        raise WorkspaceError("Docker isn't available; install it, or set GA_SANDBOX=local "
                             "to run the executor's commands on this machine")
    return DockerRunner()


def _git(cwd, *args, check=True):
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, errors="replace")
    if check and done.returncode != 0:
        raise WorkspaceError(f"git {' '.join(args)} failed: {(done.stderr or done.stdout).strip()}")
    return done.stdout.strip()


def ensure_origin(root=WORKSPACES):
    """The sandbox repository: a bare repo with a main branch to start from."""
    origin = Path(root) / "origin.git"
    if (origin / "HEAD").exists():
        return origin
    origin.mkdir(parents=True, exist_ok=True)
    _git(origin, "init", "--bare", "--initial-branch=main")
    seed = Path(root) / ".seed"
    shutil.rmtree(seed, ignore_errors=True)
    _git(root, "clone", str(origin), str(seed))
    (seed / "README.md").write_text(SEED_README, encoding="utf-8")
    _git(seed, "add", "README.md")
    _git(seed, "-c", "user.name=Sandbox", "-c", "user.email=sandbox@gsf.local",
         "commit", "-m", "Start the company website")
    _git(seed, "push", "origin", "HEAD:main")
    shutil.rmtree(seed, ignore_errors=True)
    return origin


class Workspace:
    """One agent's checkout of the sandbox repository."""

    def __init__(self, agent, root=WORKSPACES, runner=None):
        self.agent_id = agent["id"]
        self.author = (agent["name"], f"{agent['id']}@gsf.local")
        self.root = Path(root)
        self.origin = ensure_origin(self.root)
        self.path = self.root / self.agent_id
        self.runner = runner if runner is not None else default_runner()
        if not (self.path / ".git").exists():
            _git(self.root, "clone", str(self.origin), str(self.path))

    def __repr__(self):
        return f"Workspace({self.path})"

    def git(self, *args, check=True):
        name, email = self.author
        return _git(self.path, "-c", f"user.name={name}", "-c", f"user.email={email}", *args,
                    check=check)

    # -- files -------------------------------------------------------------
    def resolve(self, relative):
        """The path inside the checkout, or WorkspaceError for anything outside
        it (absolute paths, .., links out) or inside .git."""
        target = (self.path / (relative or ".")).resolve()
        base = self.path.resolve()
        if target != base and base not in target.parents:
            raise WorkspaceError(f"{relative!r} is outside the workspace")
        if ".git" in target.relative_to(base).parts:
            raise WorkspaceError("the .git directory is off limits")
        return target

    def list_files(self, relative="."):
        base = self.resolve(relative)
        if not base.is_dir():
            raise WorkspaceError(f"{relative!r} is not a directory")
        root = self.path.resolve()
        files = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != ".git")
            files += [Path(dirpath, f).relative_to(root).as_posix() for f in sorted(filenames)]
        return sorted(files)

    def read_file(self, relative):
        path = self.resolve(relative)
        if not path.is_file():
            raise WorkspaceError(f"there is no file {relative!r}")
        data = path.read_bytes()
        if len(data) > MAX_READ_BYTES:
            raise WorkspaceError(f"{relative!r} is {len(data)} bytes; too big to read whole")
        return data.decode("utf-8", errors="replace")

    def write_file(self, relative, contents):
        path = self.resolve(relative)
        if path == self.path.resolve() or path.is_dir():
            raise WorkspaceError(f"{relative!r} is a directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8", newline="\n")

    def run(self, command, timeout=COMMAND_TIMEOUT):
        return self.runner.run(self.path, command, timeout)

    # -- branches ------------------------------------------------------------
    @staticmethod
    def branch(task_id):
        return f"task/{task_id}"

    def start_task(self, task_id):
        """Check out the task's branch: the existing one if work already began
        (here or published), else a new one from the latest main."""
        self.git("fetch", "--quiet", "origin")
        branch = self.branch(task_id)
        if self.git("rev-parse", "--verify", "--quiet", branch, check=False):
            self.git("checkout", "--quiet", branch)
        elif self.git("rev-parse", "--verify", "--quiet", f"origin/{branch}", check=False):
            self.git("checkout", "--quiet", "-b", branch, f"origin/{branch}")
        else:
            self.git("checkout", "--quiet", "-b", branch, "origin/main")
        return branch

    def changed_files(self):
        """Files the task's branch changes against main, committed or not."""
        committed = self.git("diff", "--name-only", "origin/main...HEAD").splitlines()
        pending = [line[3:].strip('"') for line in
                   self.git("status", "--porcelain", "--untracked-files=all").splitlines()]
        return sorted({f for f in committed + pending if f})

    def commit(self, message):
        """Commit everything in the checkout; returns False if nothing changed."""
        self.git("add", "-A")
        if not self.git("status", "--porcelain"):
            return False
        self.git("commit", "--quiet", "-m", message)
        return True

    def publish(self, task_id):
        """Push the task's branch to the sandbox repository for review."""
        self.git("push", "--quiet", "--force-with-lease", "origin", self.branch(task_id))

    def integrate(self, task_id):
        """Merge the task's published branch into main (after approval)."""
        branch = self.branch(task_id)
        self.git("fetch", "--quiet", "origin")
        self.git("checkout", "--quiet", "--detach", "origin/main")
        self.git("merge", "--quiet", "--no-ff", "-m", f"Merge {branch}", f"origin/{branch}")
        self.git("push", "--quiet", "origin", "HEAD:main")
