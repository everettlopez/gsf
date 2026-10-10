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


def event(text, min_level=INFO):
    """A notable thing that happened during the step (indented one level)."""
    _emit(f"  ▸ {text}", min_level)


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
