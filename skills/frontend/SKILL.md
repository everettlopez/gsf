---
name: frontend
description: The site's pages, layout, styles and JavaScript. Loaded for frontend tasks.
---

# Frontend: the pages and what visitors can do on them

You own `site/layout.html`, `site/pages/*.html`, `site/assets/` and `tests/js/`.

## The pages

Each page is a view of the snapshot (`window.SITE_DATA.snapshot`). What each one is for:

- **index** (home): the studio in a sentence, then "right now": where each person is and
  what they're doing, how far the project is (tasks done of all), and the latest events.
- **team**: a card per person (name, role, traits); choosing one shows their profile: what
  they're doing now, today's plan, recent memories and reflections, their open tasks.
- **board**: the project board in columns by status; filter by person and by area; choosing
  a task shows its description, who has it, what it depends on, and its history and notes.
- **day** (a day at the office): replay the day. A time control (a range input over the
  timeline's times, plus previous/next buttons) shows where everyone was and what they were
  doing at that moment; below it, the events and conversations of the day, newest first,
  each conversation's lines in a `<details>`. With several days, choose the day.
- **office**: the five rooms as a simple map (CSS grid is plenty), each with its objects
  and who is in it; the same time control as the day page.
- **minds** (behind the scenes): how the agents think: reflections as a feed (filter by
  person), and each person's plan for the day.
- **404**: a short note and a link home.

## Markup and styles

- `site/layout.html`: `<!doctype html>`, `<html lang="en">`, `<meta charset="utf-8">`, a
  viewport meta, `<title>{{title}} · {{company_name}}</title>`, the stylesheet
  (`assets/styles.css`), a header with `{{company_name}}`, `{{tagline}}` and
  `<nav aria-label="Main">{{nav}}</nav>`, `<main id="main">{{content}}</main>`, a footer,
  then `<script src="assets/site-data.js"></script>` and the page's own scripts. Paths are
  relative (`assets/...`), so the site works from any folder.
- Semantic HTML: one `h1` per page, headings in order, landmarks, real `<button>`s for
  actions and real links for navigation. A "skip to content" link.
- Accessible: everything works with the keyboard, focus is visible, text contrast is at least
  4.5:1, filter and time changes are announced (`aria-live="polite"` on the region that
  changes), and `prefers-reduced-motion` is respected.
- Plain CSS, no frameworks, no outside fonts. Custom properties for colors and spacing at
  the top of `styles.css`. Mobile first; it should work on a phone and a wide screen.

## JavaScript

- Plain scripts (no modules, no `fetch`, no libraries): one file per area, e.g.
  `assets/js/board.js`, plus a small `assets/js/common.js` for what every page shares.
- Keep the logic in pure functions of the data (filter tasks, the state of the office at a
  time, the latest events) and the DOM work thin. Export the pure functions so Node can test
  them, with this pattern:

  ```js
  (function (root, factory) {
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    else root.Board = api;
  })(this, function () {
    function tasksByStatus(snapshot) { /* ... */ }
    return { tasksByStatus: tasksByStatus };
  });
  ```

  and do the page wiring only when there is a page:
  `if (typeof document !== "undefined") { document.addEventListener("DOMContentLoaded", ...) }`.
- Build elements with `document.createElement` and `textContent`: the data is text the
  office wrote, and must never be read as HTML.
- Every view handles empty or missing data with a short, friendly message.

## Testing it

- `tests/js/<name>.test.js` with `node:test` and `node:assert/strict`, requiring the file
  under test (`require("../../site/assets/js/board.js")`) and checking the pure functions
  with small inline snapshots. Run them with `python run_tests.py` (or `node --test tests/js`).
- `tests/test_pages.py` parses the layout and each page with `html.parser`: one `h1`, the
  landmarks, the scripts and stylesheet linked with relative paths, only the contract's
  placeholders. Test the files themselves, not the built site.
