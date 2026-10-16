---
name: review
description: Reviewing a teammate's task branch before approving it. Loaded for review sessions.
---

# Review: a teammate's task, before approval

You are reviewing, not building: read and run, but change no files.

1. See what changed: `git diff origin/main...HEAD --stat`, then the full diff of each file.
2. Run the tests: `python -m unittest discover -s tests -v`.
3. Check the task's "done when" and description against what the branch actually does.
4. Check the contract in the shared conventions: placeholders and content keys match, text
   is escaped, nothing outside the task's lane was changed.
5. Report a verdict:
   - **approve** when the tests pass and the task does what it says;
   - **changes** otherwise, with each change as one concrete, checkable sentence
     ("build.py doesn't escape the tagline"), most important first.

Small style preferences are not reasons to block an approval; mention them as optional.
