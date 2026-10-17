---
name: review
description: Reviewing a teammate's task branch before approving it. Loaded for review sessions.
---

# Review: a teammate's task, before approval

You are reviewing, not building: read and run, but change no files.

1. See what changed: `git diff origin/main...HEAD --stat`, then the full diff of each file.
2. Run the tests: `python run_tests.py` (Python and JavaScript).
3. If the branch has the build, run `python build.py --check`.
4. Check the task's "done when" and description against what the branch actually does.
5. Check the contract in the shared conventions:
   - placeholders and `content/site.json` match; nothing else is a placeholder;
   - data from the snapshot is escaped: `html.escape` in Python, `textContent` /
     `createElement` in the browser, never `innerHTML` with the data;
   - the site works from disk: no `fetch`, no `type="module"`, nothing from the internet,
     relative paths only;
   - empty or missing data shows a friendly message, not an error;
   - nothing outside the task's lane changed, and `data/snapshot.json` and `run_tests.py`
     are untouched.
6. Report a verdict:
   - **approve** when the tests pass and the task does what it says;
   - **changes** otherwise, with each change as one concrete, checkable sentence
     ("board.js puts task titles into innerHTML"), most important first.

Small style preferences are not reasons to block an approval; mention them as optional.
