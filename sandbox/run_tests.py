"""
Run every test in the repository: the Python tests (unittest) and the
JavaScript tests (node --test). This is the command the host runs to decide
whether a task's tests pass, so it is kept by the simulation, not the team:
changes to this file are not kept.

    python run_tests.py

Python tests: tests/test_*.py. JavaScript tests: tests/**/*.test.js
(also .mjs, .cjs). Exit code 0 only if every suite that exists passes;
"NO TESTS FOUND" and exit code 5 if there are none at all.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"
JS_SUFFIXES = (".test.js", ".test.mjs", ".test.cjs")


def python_tests():
    return sorted(TESTS.glob("test_*.py"))


def js_tests():
    return sorted(p for p in TESTS.rglob("*") if p.is_file() and p.name.endswith(JS_SUFFIXES)
                  and "node_modules" not in p.parts)


def main():
    if not TESTS.is_dir() or not (python_tests() or js_tests()):
        print("NO TESTS FOUND (add tests/test_*.py or tests/**/*.test.js)")
        return 5
    failed = []
    if python_tests():
        print("== Python tests (unittest) ==", flush=True)
        done = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                              cwd=ROOT)
        if done.returncode != 0:
            failed.append("Python")
    if js_tests():
        print("== JavaScript tests (node --test) ==", flush=True)
        node = shutil.which("node")
        if node is None:
            print("node isn't installed here, so the JavaScript tests can't run")
            failed.append("JavaScript")
        else:
            files = [str(p.relative_to(ROOT)) for p in js_tests()]
            if subprocess.run([node, "--test", *files], cwd=ROOT).returncode != 0:
                failed.append("JavaScript")
    print(f"== {'FAILED: ' + ', '.join(failed) if failed else 'all tests passed'} ==")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
