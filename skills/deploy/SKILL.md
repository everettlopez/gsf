---
name: deploy
description: Shipping the site - the build output and its final checks. Loaded for deploy, build and release tasks.
---

# Deploy: shipping the site

There is no server: shipping means `dist/` is complete, correct, and reproducible from a
clean checkout.

- `python build.py` from a clean checkout produces `dist/index.html` and `dist/styles.css`,
  and nothing else.
- `python build.py --check` builds into a temporary directory and verifies the result
  without touching `dist/`: the HTML parses, every local link and the stylesheet resolve,
  the `mailto:` link is well formed, and no `{{` is left. It exits non-zero on any failure.
- `dist/` is committed, so the shipped site is what the repository's `main` shows.
- A `tests/test_deploy.py` runs the `--check` path.
