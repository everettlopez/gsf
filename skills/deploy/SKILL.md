---
name: deploy
description: Shipping the site - a complete, checked build anyone can reproduce. Loaded for deploy, build and release tasks.
---

# Deploy: shipping the site

There is no server: shipping means `python build.py` from a clean checkout of `main`
produces a complete `dist/` that works opened straight from disk, and nothing about it
depends on the machine that built it.

- `dist/` is built, never committed. A release is the commit on `main` that builds cleanly.
- `python build.py --check` passes: every page builds, every local link and script
  resolves, the data loads, no placeholder is left.
- Every page in `content/site.json` has its file in `site/pages/`, and every file in
  `site/pages/` is listed: an unlisted page is a mistake the check reports.
- `CHANGELOG.md` gets one line per shipped task: its id, what a visitor can now see or do.
- `tests/test_deploy.py` runs the `--check` path against the fixture snapshot and checks
  that a page missing from the list (or listed but missing) fails it.
