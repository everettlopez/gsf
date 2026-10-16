"""
Preview the website the agents have built so far.

The agents' work lives on separate task branches (task/T-n) in their own
checkouts under workspaces/, and only meets on main once a task is approved
and done. This puts it all together now: the sandbox's main, then the newest
commit of every task branch (published or still in an agent's checkout),
laid over it in task order, in preview/. Then it builds the site and runs
every test in the Docker sandbox, and opens the page.

    python scripts/preview_site.py             # combine, build, test, open
    python scripts/preview_site.py --no-open   # ... without opening a browser
    python scripts/preview_site.py --local     # run build and tests here, not in Docker

The board's tasks are listed alongside, so you can see whose work is where.
Nothing in workspaces/ is changed.
"""

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from workspace import DockerRunner, LocalRunner  # noqa: E402

TEST_COMMAND = "python -m unittest discover -s tests"


def git(repo, *args):
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    return done.stdout.decode("utf-8", "replace").strip() if done.returncode == 0 else ""


def repos(workspaces):
    """The sandbox repo and every agent checkout, as (label, path, ref prefix)."""
    found = []
    origin = workspaces / "origin.git"
    if (origin / "HEAD").exists():
        found.append(("published", origin, ""))
    for path in sorted(workspaces.iterdir()):
        if (path / ".git").exists():
            found.append((f"{path.name}'s checkout", path, ""))
    return found


def task_branches(workspaces):
    """task id -> (label, repo, branch, commit time, subject) of its newest commit."""
    newest = {}
    for label, repo, _ in repos(workspaces):
        for line in git(repo, "for-each-ref", "--format=%(refname:short)|%(committerdate:unix)|%(subject)",
                        "refs/heads/task/").splitlines():
            branch, when, subject = line.split("|", 2)
            task_id = branch.split("/", 1)[1]
            if task_id not in newest or int(when) > newest[task_id][3]:
                newest[task_id] = (label, repo, branch, int(when), subject)
    return newest


def extract(repo, ref, dest):
    data = subprocess.run(["git", "-C", str(repo), "archive", ref], capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")


def board(world_path):
    try:
        return json.loads(Path(world_path).read_text(encoding="utf-8"))["project"]["tasks"]
    except (OSError, KeyError, ValueError):
        return []


def task_order(task_id):
    try:
        return int(task_id.split("-")[1])
    except (IndexError, ValueError):
        return 10_000


def combine(workspaces, dest, world_path=None):
    """Put main plus every task branch's newest work into `dest`. Returns the
    lines describing what went in."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    report = []
    origin = workspaces / "origin.git"
    if git(origin, "rev-parse", "--verify", "--quiet", "main"):
        extract(origin, "main", dest)
        report.append(f"main: {git(origin, 'log', '-1', '--format=%s', 'main')}")
    tasks = {t["id"]: t for t in board(world_path)} if world_path else {}
    for task_id, (label, repo, branch, _, subject) in sorted(task_branches(workspaces).items(),
                                                             key=lambda kv: task_order(kv[0])):
        extract(repo, branch, dest)
        task = tasks.get(task_id)
        on_board = f" [{task['status'].replace('_', ' ')}, {task['assignee']}]" if task else ""
        report.append(f"{task_id}{on_board} from {label}: {subject}")
    return report


def build_and_test(dest, runner):
    results = {}
    if (dest / "build.py").exists():
        results["build"] = runner.run(dest, "python build.py")
    if (dest / "tests").is_dir():
        results["tests"] = runner.run(dest, TEST_COMMAND)
    return results


def page(dest):
    for candidate in ("dist/index.html", "site/template.html", "index.html"):
        if (dest / candidate).exists():
            return dest / candidate
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workspaces", default=str(ROOT / "workspaces"))
    parser.add_argument("--out", default=str(ROOT / "preview"))
    parser.add_argument("--world", default=str(ROOT / "data" / "world" / "world.json"))
    parser.add_argument("--local", action="store_true", help="run build and tests here, not in Docker")
    parser.add_argument("--no-open", action="store_true", help="don't open the page in a browser")
    args = parser.parse_args(argv)

    workspaces, dest = Path(args.workspaces), Path(args.out)
    if not (workspaces / "origin.git").exists():
        print(f"No sandbox at {workspaces}: run the simulation with GA_EXECUTOR=local first.")
        return 1

    print("The board:")
    for task in board(args.world):
        print(f"  {task['id']} [{task['status'].replace('_', ' ')}] {task['title']} "
              f"({task['assignee'] or 'unassigned'})")
    print("\nCombined into", dest, ":")
    for line in combine(workspaces, dest, args.world):
        print("  " + line)

    results = build_and_test(dest, LocalRunner() if args.local else DockerRunner())
    for name, result in results.items():
        print(f"\n{name}: {'ok' if result.ok else f'failed (exit {result.exit_code})'}")
        print("  " + "\n  ".join(result.output.strip().splitlines()[-6:]))

    shown = page(dest)
    if shown is None:
        print("\nNo page yet: nobody has written site/template.html or built dist/.")
        return 0
    note = "" if shown.parts[-2] == "dist" else "  (not built yet: placeholders unfilled)"
    print(f"\nThe page: {shown}{note}")
    if not args.no_open:
        webbrowser.open(shown.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
