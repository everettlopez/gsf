---
name: backend
description: The build that turns the pages and the office's data into the finished site, and its checks. Loaded for backend tasks.
---

# Backend: the build and the data

You own `build.py`, `content/site.json`, `tests/fixtures/snapshot.json` and the Python
tests of the build.

## The content

- `content/site.json` holds the site's own words and its pages, in the shape the shared
  conventions give: company name, tagline, a short intro, the contact email, and the list
  of pages with their titles and menu labels. Real, short copy for a small studio; no lorem
  ipsum. The company's facts are the settled ones.

## The build

`python build.py` writes the finished site to `dist/`:

1. Load and check `content/site.json` and `data/snapshot.json`: `"schema": 1`, the top-level
   keys present and of the right type (lists are lists). Anything wrong stops the build
   with a message that says what and where, and a non-zero exit code.
2. For each page in `content/site.json` whose `site/pages/<name>.html` exists: put it into
   `site/layout.html` and fill the placeholders (escaping text with `html.escape`), with the
   menu built from the pages list, the current page marked `aria-current="page"`. Write
   `dist/<name>.html`.
3. Write `dist/assets/site-data.js`: `window.SITE_DATA = {...};` with the snapshot and the
   content. Embed the JSON safely: `json.dumps(..., ensure_ascii=False)` with every `</`
   written as `<\/`, so no text in the data can end the script.
4. Copy `site/assets/` into `dist/assets/`.
5. Check the result: every built page parses, no `{{` is left, and every local `href` and
   `src` points at a file in `dist/` (anchors `#...` and `mailto:` links aside).

`python build.py --check` does all of that in a temporary directory and leaves `dist/`
alone. Start `dist/` empty on every build, so nothing stale survives.

Keep `build.py` importable: the work in functions (`load_snapshot`, `load_content`,
`render_page`, `site_data_js`, `check_site`, `build`), the command line in
`if __name__ == "__main__":`.

## The fixture

`tests/fixtures/snapshot.json` is a small, hand-written snapshot in the documented shape:
two agents, a few rooms, three or four tasks in different states, a dozen timeline steps,
a few events, one short conversation. Every test (yours and the frontend's) builds against
it, so keep it valid and small, and extend it when a new view needs something it lacks.

## Testing it

`tests/test_build.py` runs the build on small inline pages and the fixture in a temporary
directory: placeholders filled and escaped, the menu and `aria-current`, `site-data.js`
loads the data (and a `</script>` inside a text doesn't break it), the link check catches a
broken link, and each bad input fails with a clear message.
