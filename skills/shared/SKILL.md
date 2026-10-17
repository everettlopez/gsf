---
name: shared
description: Team conventions for the company website repository. Loaded for every task.
---

# The company website: how the repository fits together

One static page, built from two halves that meet at a contract:

| Path | Owner | What it is |
|---|---|---|
| `site/template.html` | frontend | The page layout, with `{{placeholders}}` where the text goes |
| `site/styles.css` | frontend | The page's styles |
| `content/site.json` | backend | The page's text, one key per placeholder |
| `build.py` | backend | Fills the template with the content, checks the page, writes `dist/` |
| `dist/` | backend (generated) | The finished site: `index.html` and `styles.css`. Not edited by hand |
| `tests/` | both | `unittest` tests, one file per area (`test_page.py`, `test_build.py`, ...) |

## The company

These are settled; use them as they are, and don't ask a teammate for them.

- Name: Beck & Smith (`company_name`)
- Contact email: hello@becksmith.example (`contact_email`)
- Team: Adam Beck, backend engineer; Eve Smith, frontend engineer (`team`)
- The tagline and the about copy are yours to write: short and plain, about a small
  software studio that builds simple, reliable websites.

## The contract between template and content

- A placeholder is `{{key}}`, lowercase with underscores: `{{company_name}}`, `{{about}}`.
- Every placeholder in the template has a key in `content/site.json`, and every key is used.
- Text values are plain text; the build escapes them. The one list, `team`, is a list of
  `{"name": ..., "role": ...}` objects; the build renders it as `<li>` items into `{{team}}`.
- A new placeholder or key is a change to the contract: say so in the report, so the
  teammate on the other half can match it.

## Working rules

- Python 3.12 standard library only (no network, nothing to install).
- Run every test with `python -m unittest discover -s tests -v` and keep it passing.
- Stay in your lane's files. If the other half doesn't exist yet, build against a small
  stand-in (a sample `site.json`, a minimal template) in your own tests, not in their files.
