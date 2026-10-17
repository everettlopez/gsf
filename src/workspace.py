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
GA_SANDBOX=local, since the commands come from a model). The container image
(sandbox/Dockerfile: Python 3.12 and Node 22) is built the first time it is
needed.

Some files in the sandbox belong to the simulation, not the team (HOST_FILES):
the snapshot of the office the website shows (snapshot.py) and the command
that runs the tests. The host commits them to main (update_host_files); the
hands can read them but not change them: a write is refused, and anything a
command changes in them is put back before a commit.
"""

import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent.parent
WORKSPACES = ROOT / "workspaces"

MAX_READ_BYTES = 200_000
COMMAND_TIMEOUT = 120          # seconds
MAX_OUTPUT = 20_000            # characters of command output kept

SANDBOX_DIR = ROOT / "sandbox"
DOCKER_IMAGE = "gsf-sandbox:py3.12-node22"      # built from SANDBOX_DIR/Dockerfile

# The files only the host writes: the test command (executor.TEST_COMMAND runs
# it) and the snapshot of the office.
RUN_TESTS = "run_tests.py"
SNAPSHOT_FILE = "data/snapshot.json"
HOST_FILES = (RUN_TESTS, SNAPSHOT_FILE)

# dist/ is built, never committed: both halves of the team rebuild it, and
# committed builds conflicted on every merge.
SEED_GITIGNORE = "__pycache__/\n*.pyc\ndist/\nnode_modules/\n"

SEED_README = """# Beck & Smith: the studio's website

The team's sandbox repository. Each task is built on its own branch
(task/T-<n>) and merged into main once a teammate has approved it.

data/snapshot.json and run_tests.py are kept by the simulation: the first is
the office as it is now, the second runs every test (`python run_tests.py`).
"""

HOST_AUTHOR = ("The office", "office@gsf.local")


def host_file_contents():
    """The host-owned files every sandbox main has (the snapshot is added by
    snapshot.export when there is one)."""
    return {RUN_TESTS: (SANDBOX_DIR / RUN_TESTS).read_text(encoding="utf-8")}


def is_host_file(relative):
    return Path(relative).as_posix() in HOST_FILES


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
                                  timeout=timeout, errors="replace",
                                  env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
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
                "-e", "PYTHONDONTWRITEBYTECODE=1",
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


def ensure_image(image=DOCKER_IMAGE):
    """Build the sandbox image from sandbox/Dockerfile if Docker doesn't have
    it yet (once: a minute or two, with network, on the host). An image named
    by GA_SANDBOX_IMAGE is the user's to provide."""
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0:
        return
    print(f"Building the sandbox image {image} from {SANDBOX_DIR / 'Dockerfile'} "
          f"(first run only)...", flush=True)
    done = subprocess.run(["docker", "build", "-t", image, str(SANDBOX_DIR)],
                          capture_output=True, text=True, errors="replace")
    if done.returncode != 0:
        raise WorkspaceError(f"couldn't build the sandbox image {image}: "
                             f"{(done.stderr or done.stdout).strip()[-800:]}")


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
    runner = DockerRunner()
    if runner.image == DOCKER_IMAGE:
        ensure_image(runner.image)
    return runner


def _git(cwd, *args, check=True):
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, errors="replace")
    if check and done.returncode != 0:
        raise WorkspaceError(f"git {' '.join(args)} failed: {(done.stderr or done.stdout).strip()}")
    return done.stdout.strip()


def ensure_origin(root=None):
    """The sandbox repository: a bare repo with a main branch to start from."""
    root = Path(root or WORKSPACES)
    origin = root / "origin.git"
    if (origin / "HEAD").exists():
        return origin
    origin.mkdir(parents=True, exist_ok=True)
    _git(origin, "init", "--bare", "--initial-branch=main")
    seed = root / ".seed"
    shutil.rmtree(seed, ignore_errors=True)
    _git(root, "-c", "core.autocrlf=false", "clone", str(origin), str(seed))
    (seed / "README.md").write_text(SEED_README, encoding="utf-8")
    (seed / ".gitignore").write_text(SEED_GITIGNORE, encoding="utf-8")
    for relative, contents in host_file_contents().items():
        _write(seed / relative, contents)
    _git(seed, "add", "-A")
    _git(seed, "-c", f"user.name={HOST_AUTHOR[0]}", "-c", f"user.email={HOST_AUTHOR[1]}",
         "commit", "-m", "Start the studio's website")
    _git(seed, "push", "origin", "HEAD:main")
    shutil.rmtree(seed, ignore_errors=True)
    return origin


def _write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8", newline="\n")


def update_host_files(files, root=None, message="Update the office's files"):
    """Commit host-owned files to the sandbox's main, from the host's own
    checkout (workspaces/.host), with the host files every main has. `files`:
    path -> contents. Returns True if main changed."""
    root = Path(root or WORKSPACES)
    origin = ensure_origin(root)
    host = root / ".host"
    if not (host / ".git").exists():
        shutil.rmtree(host, ignore_errors=True)
        _git(root, "-c", "core.autocrlf=false", "clone", "--quiet", str(origin), str(host))
        _git(host, "config", "core.autocrlf", "false")
    _git(host, "fetch", "--quiet", "origin")
    _git(host, "checkout", "--quiet", "-B", "main", "origin/main")
    for relative, contents in {**host_file_contents(), **files}.items():
        _write(host / relative, contents)
    _git(host, "add", "-A", "--", *{**host_file_contents(), **files})
    if not _git(host, "status", "--porcelain"):
        return False
    _git(host, "-c", f"user.name={HOST_AUTHOR[0]}", "-c", f"user.email={HOST_AUTHOR[1]}",
         "commit", "--quiet", "-m", message)
    _git(host, "push", "--quiet", "origin", "HEAD:main")
    return True


class Workspace:
    """One agent's checkout of the sandbox repository."""

    def __init__(self, agent, root=None, runner=None):
        self.agent_id = agent["id"]
        self.author = (agent["name"], f"{agent['id']}@gsf.local")
        self.root = Path(root or WORKSPACES)
        self.origin = ensure_origin(self.root)
        self.path = self.root / self.agent_id
        self.runner = runner if runner is not None else default_runner()
        if not (self.path / ".git").exists():
            # Files exactly as written: the sandbox is Linux, and Windows git
            # otherwise turns them into CRLF on every checkout.
            _git(self.root, "-c", "core.autocrlf=false", "clone", str(self.origin), str(self.path))
            _git(self.path, "config", "core.autocrlf", "false")

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
        # A drive or root ("C:/...", "\\\\server\\...") is absolute on Windows; on
        # Linux it would quietly become a folder named "C:" in the checkout.
        if PureWindowsPath(relative or ".").anchor:
            raise WorkspaceError(f"{relative!r} is outside the workspace")
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
        if is_host_file(path.relative_to(self.path.resolve())):
            raise WorkspaceError(f"{relative!r} is kept by the simulation, not the team: read it, "
                                 f"but don't change it")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8", newline="\n")

    def run(self, command, timeout=COMMAND_TIMEOUT):
        return self.runner.run(self.path, command, timeout)

    # -- branches ------------------------------------------------------------
    @staticmethod
    def branch(task_id):
        return f"task/{task_id}"

    def start_task(self, task_id, build_on=()):
        """Check out the task's branch: the existing one if work already began
        (here or published), else a new one from the latest main.

        `build_on`: the agent's own other tasks whose published work isn't on
        main yet (in review, say). Their branches are merged in, so this task
        builds on that work instead of redoing it: Adam's T-1 and T-2 both
        wrote build.py from scratch, and T-1 then conflicted with main. A merge
        that conflicts is left out. Returns the branch."""
        self.git("fetch", "--quiet", "origin")
        branch = self.branch(task_id)
        if self.git("rev-parse", "--verify", "--quiet", branch, check=False):
            self.git("checkout", "--quiet", branch)
        elif self.git("rev-parse", "--verify", "--quiet", f"origin/{branch}", check=False):
            self.git("checkout", "--quiet", "-b", branch, f"origin/{branch}")
        else:
            self.git("checkout", "--quiet", "-b", branch, "origin/main")
        self.built_on = []
        for other in build_on:
            ref = f"origin/{self.branch(other)}"
            if other == task_id or not self.git("rev-parse", "--verify", "--quiet", ref, check=False):
                continue
            if subprocess.run(["git", "merge-base", "--is-ancestor", ref, "HEAD"],
                              cwd=self.path, capture_output=True).returncode == 0:
                continue                                    # already in
            try:
                self.git("merge", "--quiet", "--no-edit", ref)
                self.built_on.append(other)
            except WorkspaceError:
                self.git("merge", "--abort", check=False)
        return branch

    def changed_files(self):
        """Files the task's branch changes against main, committed or not."""
        committed = self.git("diff", "--name-only", "origin/main...HEAD").splitlines()
        pending = [line[3:].strip('"') for line in
                   self.git("status", "--porcelain", "--untracked-files=all").splitlines()]
        return sorted({f for f in committed + pending if f})

    def restore_host_files(self):
        """Put the host-owned files back as the branch has them (a command may
        have changed them): they only ever change on main, by the host, so a
        task branch never conflicts with the next snapshot."""
        for relative in HOST_FILES:
            if self.git("ls-files", "--error-unmatch", relative, check=False):
                self.git("checkout", "--quiet", "HEAD", "--", relative, check=False)
            else:
                (self.path / relative).unlink(missing_ok=True)

    def commit(self, message, body=""):
        """Commit everything in the checkout but the host-owned files; returns
        False if nothing changed."""
        self.restore_host_files()
        self.git("add", "-A")
        if not self.git("status", "--porcelain"):
            return False
        self.git("commit", "--quiet", "-m", message, *(["-m", body] if body else []))
        return True

    def diff(self):
        """What the checked-out branch changes against main: the summary, then
        the full diff (trimmed like command output)."""
        stat = self.git("diff", "--stat", "origin/main...HEAD")
        return _trim(f"{stat}\n\n{self.git('diff', 'origin/main...HEAD')}" if stat else "")

    def checkout_review(self, task_id):
        """Check out a teammate's published task branch to review it (detached:
        nothing done here goes anywhere). False if it isn't published."""
        self.git("fetch", "--quiet", "origin")
        remote = f"origin/{self.branch(task_id)}"
        if not self.git("rev-parse", "--verify", "--quiet", remote, check=False):
            return False
        self.git("checkout", "--quiet", "--detach", remote)
        return True

    def discard_changes(self):
        """Throw away anything changed in the checkout since the last commit."""
        self.git("reset", "--quiet", "--hard")
        self.git("clean", "-fdq")

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
