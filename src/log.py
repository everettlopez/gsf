"""
log.py - simulation logging with three verbosity levels.

    GA_LOG=quiet   warnings, reflections, conversations and the end-of-run summary
    GA_LOG=info    (default) one compact line per agent per step, plus events
    GA_LOG=debug   everything: observations, per-call timings, raw LLM decisions

Set it per run, e.g.  GA_LOG=debug python src/main.py

Output layout at the default level:

    ━━ STEP 7/30 · 7:30 am ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      ADAM  office_1    typing code                  [computer]
      EVE   reception   clocking in                  [time_clock]
      💬 ADAM ↔ EVE  (reception)
          ADAM: "Morning Eve!"
          EVE : "Hi Adam."
"""

import os
import sys
import time
from contextlib import contextmanager

QUIET, INFO, DEBUG = 0, 1, 2
_LEVELS = {"quiet": QUIET, "info": INFO, "debug": DEBUG}

level = _LEVELS.get(os.environ.get("GA_LOG", "info").lower(), INFO)

WIDTH = 64


def _emit(text, min_level, stderr=False):
    if level >= min_level:
        # Look the stream up per call so redirection (tests, pipes) is respected.
        print(text, file=sys.stderr if stderr else sys.stdout, flush=True)


def set_level(name):
    global level
    level = _LEVELS[name]


# --- Generic ---------------------------------------------------------------
def info(text):
    _emit(text, INFO)


def debug(text):
    _emit(f"    · {text}", DEBUG)


def warn(text):
    _emit(f"  ⚠ {text}", QUIET, stderr=True)


# Called with every event's text, whatever the verbosity (chronicle.py keeps
# them for the website).
listeners = []


def event(text, min_level=INFO):
    """A notable thing that happened during the step (indented one level)."""
    _emit(f"  ▸ {text}", min_level)
    for listener in listeners:
        listener(text)


# --- Structure -------------------------------------------------------------
def banner(text):
    _emit("\n" + text, QUIET)


def step_header(step, total, time_str):
    label = f"━━ STEP {step}/{total} · {time_str} "
    _emit("\n" + label + "━" * max(4, WIDTH - len(label)), INFO)


def agent_line(tag, location, action, obj=None):
    """The one-line-per-agent summary printed at the end of every step."""
    obj_text = f"[{obj}]" if obj else ""
    _emit(f"  {tag:<5} {location:<16} {action:<44} {obj_text}".rstrip(), INFO)


def conversation_start(a_tag, b_tag, location, reaction):
    _emit(f"  💬 {a_tag} ↔ {b_tag}  ({location})  — {reaction}", QUIET)


def dialogue(tag, line):
    _emit(f'      {tag:<5}: "{line}"', QUIET)


def reflection_header(tag, time_str):
    _emit(f"  🧠 {tag} reflecting ({time_str})", QUIET)


def reflection_item(text):
    _emit(f"      → {text}", QUIET)


# ---------------------------------------------------------------------------
# TIMING (where is the time going?)
# ---------------------------------------------------------------------------
timings = {}  # label -> (total_seconds, call_count)


@contextmanager
def timed(label):
    """Time a block, print it, and accumulate it for the end-of-run summary."""
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        total, count = timings.get(label, (0.0, 0))
        timings[label] = (total + dt, count + 1)
        debug(f"timing {label}: {dt:.1f}s")


def print_timing_summary():
    if not timings:
        return
    banner("TIMING SUMMARY (slowest first)")
    for label, (total, count) in sorted(timings.items(), key=lambda kv: -kv[1][0]):
        info(f"  {total:7.1f}s total | {count:3d} calls | {total / count:6.1f}s avg | {label}")
