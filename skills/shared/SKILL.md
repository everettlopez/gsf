---
name: shared
description: Team conventions for the studio's website repository. Loaded for every task.
---

# The studio's website: a window into the office

Beck & Smith's website shows visitors the studio from the inside: who Adam and Eve are,
where they are and what they're doing, the project board, their conversations, plans and
thoughts. That data is real: it comes from `data/snapshot.json`, which the office keeps up
to date. The site shows it; it never invents it.

## The company

These are settled; use them as they are, and don't ask a teammate for them.

- Name: Beck & Smith. Contact email: hello@becksmith.example
- Team: Adam Beck, backend engineer; Eve Smith, frontend engineer
- The tagline and the intro copy are yours to write: short and plain, about a small
  software studio that builds simple, reliable websites.

## The repository

| Path | Owner | What it is |
|---|---|---|
| `data/snapshot.json` | the office | The office right now (shape below). Read it; never change it |
| `run_tests.py` | the office | Runs every test: `python run_tests.py`. Never change it |
| `content/site.json` | backend | The site's own words and its list of pages |
| `build.py` | backend | Builds the finished site into `dist/` and checks it |
| `site/layout.html` | frontend | The layout every page shares: head, header and nav, footer |
| `site/pages/<page>.html` | frontend | One page's main content |
| `site/assets/styles.css` | frontend | The styles |
| `site/assets/js/<name>.js` | frontend | The pages' JavaScript |
| `tests/test_*.py` | both | Python tests (`unittest`), one file per area |
| `tests/js/<name>.test.js` | frontend | JavaScript tests (`node:test`) |
| `tests/fixtures/snapshot.json` | backend | A small hand-written snapshot every test uses |
| `dist/` | built | The finished site. Never committed (it's in `.gitignore`) |

Pages, by name (`site/pages/<name>.html` becomes `dist/<name>.html`): `index` (home),
`team`, `board`, `day` (a day at the office), `office` (the office map), `minds` (behind the
scenes), and `404`. Build whichever exist; a page is added by adding its file and its
entry in `content/site.json`.

## How the pieces meet

- `site/layout.html` has these placeholders, which the build fills: `{{title}}` (the page's
  title), `{{company_name}}`, `{{tagline}}`, `{{nav}}` (the menu, the current page marked
  with `aria-current="page"`), `{{content}}` (the page's own file), `{{page}}` (its name).
  A page's own file may use `{{company_name}}`, `{{tagline}}` and `{{contact_email}}`.
  Nothing else is a placeholder, and no `{{` is left in a built page.
- `content/site.json`:
  `{"company_name", "tagline", "intro", "contact_email", "pages": [{"name", "title", "nav"}]}`;
  `nav` is the menu label, or `null` for a page that isn't in the menu (404). The menu
  follows the list's order.
- The data reaches the browser as a script, never a request: the build writes
  `dist/assets/site-data.js`, which sets `window.SITE_DATA = {"snapshot": ..., "content": ...}`
  (the snapshot and `content/site.json` as they are). The layout loads it before the page
  scripts. The site is opened straight from disk (`file://`), where `fetch`, ES modules
  (`type="module"`) and anything from the internet don't work: plain `<script src>` only, no
  CDNs, no web fonts from outside, nothing that needs a server.

## The snapshot (`data/snapshot.json`, `"schema": 1`)

```
{
  "schema": 1,
  "generated": {"day": 1, "day_name": "Thursday", "time": "9:00 am"},
  "company": {"name", "email", "project", "goal", "status"},
  "days": [{"day": 1, "name": "Thursday"}],
  "agents": [{
    "id": "adam", "name": "Adam Beck", "age", "area": "backend", "office": "office_1",
    "traits": [...], "about": [...], "work_style": [...], "communication_style": [...],
    "aspirations": [...],
    "now": {"location": "office_1", "action": "writing the build", "object": "computer" | null},
    "plan": [{"start": "9:00 am", "minutes": 60, "activity", "location"}],
    "memories": [{"day", "time", "text", "importance"}],            // latest, oldest first
    "reflections": [{"day", "time", "text", "importance"}],
    "stats": {"memories", "reflections"}
  }],
  "rooms": [{"id": "office_1", "name": "Office 1", "state", "objects": [{"name", "state"}],
             "occupants": ["adam"]}],
  "board": {"tasks": [{"id": "T-1", "title", "description", "area", "priority",
                       "status": "todo|in_progress|review|blocked|done", "assignee": "eve" | null,
                       "depends_on": [...], "approved_by": [...], "done_when",
                       "created_at", "updated_at",
                       "history": [{"day", "time", "agent", "event", "note"?}]}],
            "notes": [{"day", "time", "agent", "note"}]},
  "timeline": [{"day", "time", "agent", "location", "action", "object"}],  // every 5 minutes
  "events": [{"day", "time", "kind", "agents": [...], "text"}],
  "conversations": [{"day", "time", "location", "topic", "agents": [...],
                     "lines": [{"agent", "text"}], "meeting": {...} | null,
                     "commitments": {"adam": "..."}}]
}
```

- `kind` is one of: `board`, `board_refused`, `work`, `review`, `ship`, `cost`, `question`,
  `meeting`, `promise`, `board_trip`, `reaction`, `plan`, `hands`, `event`.
- It grows through the day and soon is too big to read whole: look at it with a short
  command instead, e.g. `python -c "import json; d = json.load(open('data/snapshot.json'));
  print({k: len(v) if isinstance(v, list) else v for k, v in d.items()})"`.
- Times are written like `9:05 am`; within a day, order by the list, not by the text.
- Any list can be empty and any field can be missing early in a run: every view needs a
  friendly empty state, never an error.
- Everything in it is text the office wrote: escape it. In Python, `html.escape`; in the
  browser, `textContent` or `createElement`, never `innerHTML` with the data.
- `build.py` checks `"schema": 1` and fails clearly on anything else.

## Working rules

- Python 3.12 standard library and Node 22's built-ins only (no network, nothing to install,
  no `package.json` dependencies).
- `python run_tests.py` runs every test; keep it passing.
- Test against `tests/fixtures/snapshot.json` or small inline data, never against
  `data/snapshot.json`: the real one changes every hour.
- Stay in your lane's files. If the other half doesn't exist yet, build against a small
  stand-in in your own tests, not in their files.
- A change to the contract above (a new placeholder, a new field you rely on): say so in the
  report, so the teammate on the other half can match it.
